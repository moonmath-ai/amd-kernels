"""MXFP4 MoE GEMMs against a dense torch reference.

The reference dequantizes the STOCK weight layout and uses a plain matmul, so
it shares no code with the repack the kernels consume -- a bug in the repack
cannot hide behind a matching bug in the reference.
"""

import pytest
import torch

import moonmath_amd as ma

#  E2M1: sign bit plus a 3-bit magnitude index into these values.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

#  Kimi-K3 TP8 shard, which is the shape these kernels are tuned for.
E, H, INTER, TOPK = 64, 3584, 384, 8


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    return torch.device("cuda")


def _dequant(B, Bs):
    """[E, N, K/2] uint8 + [E, N, K/32] uint8 -> [E, N, K] float32, stock layout."""
    e, n, kh = B.shape
    tbl = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=B.device)
    lo = tbl[(B & 0x0F).long()]                       # even k
    hi = tbl[(B >> 4).long()]                         # odd k
    vals = torch.stack([lo, hi], dim=-1).reshape(e, n, kh * 2)
    scale = torch.exp2(Bs.to(torch.float32) - 127.0).repeat_interleave(32, dim=2)
    return vals * scale


def _align(topk_ids, block_m, num_experts):
    """The usual moe_align_block_size: sort rows by expert, pad each to block_m."""
    flat = topk_ids.flatten()
    order = torch.argsort(flat.float(), stable=True)
    sorted_experts = flat[order]
    rows = flat.numel()

    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        sel = order[sorted_experts == e]
        if sel.numel() == 0:
            continue
        pad = (-sel.numel()) % block_m
        blk = torch.cat([sel, torch.full((pad,), rows, dtype=torch.long, device=sel.device)])
        sorted_ids.append(blk)
        expert_ids.append(torch.full((blk.numel() // block_m,), e, dtype=torch.long,
                                     device=sel.device))

    sorted_ids = torch.cat(sorted_ids).to(torch.int32)
    expert_ids = torch.cat(expert_ids).to(torch.int32)
    ntpp = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, ntpp, expert_ids.numel()


def _route(T, n_active, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    pool = torch.randperm(E, device=device, generator=g)[:n_active]
    return pool[torch.randint(0, n_active, (T, TOPK), device=device, generator=g)].int()


def _weights(n, k, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    B = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=device, generator=g)
    #  A narrow exponent band keeps the reference matmul away from fp32 overflow
    #  while still exercising a different scale in most 32-k groups.
    Bs = torch.randint(120, 134, (E, n, k // 32), dtype=torch.uint8, device=device, generator=g)
    return B, Bs


def _reference(A, W, sorted_ids, expert_ids, block_m, rows, N, top_k):
    """Dense reference over the same padded block list the kernel walks.

    The kernel reads ``A[slot // top_k]`` and writes ``C[slot]``: gate/up takes
    per-TOKEN activations with top_k = TOPK, down takes per-(token, expert)
    rows with top_k = 1.
    """
    out = torch.zeros(rows, N, dtype=torch.float32, device=A.device)
    for blk in range(expert_ids.numel()):
        e = int(expert_ids[blk])
        slots = sorted_ids[blk * block_m:(blk + 1) * block_m].long()
        live = slots[slots < rows]
        if live.numel() == 0:
            continue
        out[live] = A[live // top_k].float() @ W[e].float().T
    return out


def _rel_err(ref, got, live):
    a, b = ref[live].float(), got[live].float()
    return ((a - b).abs().max() / a.abs().max().clamp_min(1e-9)).item()


@pytest.mark.parametrize("T,n_active", [(8, 32), (64, 48), (256, 64)])
def test_gateup_epi_none(device, T, n_active):
    rows = T * TOPK
    B, Bs = _weights(INTER, H, device, seed=11)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)

    #  gate/up consumes per-TOKEN hidden states and fans them out over top_k.
    A = (torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1)
    C = torch.zeros(rows, INTER, device=device, dtype=torch.bfloat16)

    bm = ma.mxfp4_moe_gateup_block_m(rows, n_active)
    assert bm in (16, 32, 48)
    ids = _route(T, n_active, device, seed=3)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, bm, E)

    ma.mxfp4_moe_gateup(A, Br, Bsr, C, None, sorted_ids, expert_ids, ntpp,
                        nblk, bm, rows, TOPK, epilogue=ma.EPI_NONE)
    torch.cuda.synchronize()

    ref = _reference(A, _dequant(B, Bs), sorted_ids, expert_ids, bm, rows, INTER, TOPK)
    live = torch.zeros(rows, dtype=torch.bool, device=device)
    live[sorted_ids[sorted_ids < rows].long()] = True
    assert _rel_err(ref, C, live) < 2e-2


@pytest.mark.parametrize("T,n_active", [(8, 32), (64, 48)])
def test_down(device, T, n_active):
    rows = T * TOPK
    B, Bs = _weights(H, INTER, device, seed=17)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)

    #  down consumes the gate/up output, which is already one row per
    #  (token, expert) pair, so its top_k is 1.
    A = (torch.randn(rows, INTER, device=device, dtype=torch.bfloat16) * 0.1)
    C = torch.zeros(rows, H, device=device, dtype=torch.bfloat16)

    bm = ma.mxfp4_moe_down_block_m(rows, n_active, INTER)
    assert bm in (16, 32, 64)
    ids = _route(T, n_active, device, seed=5)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, bm, E)
    _, nt, n_steps = ma.mxfp4_moe_down_plan(rows, n_active, H, INTER, nblk)

    ma.mxfp4_moe_down(A, Br, Bsr, C, None, sorted_ids, expert_ids, ntpp,
                      nblk, bm, n_steps, nt, rows, top_k=1)
    torch.cuda.synchronize()

    ref = _reference(A, _dequant(B, Bs), sorted_ids, expert_ids, bm, rows, H, 1)
    live = torch.zeros(rows, dtype=torch.bool, device=device)
    live[sorted_ids[sorted_ids < rows].long()] = True
    assert _rel_err(ref, C, live) < 2e-2


def _situ_glu(g, u, beta, linear_beta):
    """Kimi-K3 SituGLU, using tanh(x) = 2*sigmoid(2x) - 1 as the kernel does."""
    tg = 2.0 * torch.sigmoid(2.0 * g / beta) - 1.0
    tu = 2.0 * torch.sigmoid(2.0 * u / linear_beta) - 1.0
    return (beta * tg * torch.sigmoid(g)) * (linear_beta * tu)


@pytest.mark.parametrize("T,n_active", [(64, 48), (256, 64)])
def test_gateup_situ(device, T, n_active):
    """The fused epilogue: B carries gate in columns [0, INTER) and up in
    [INTER, 2*INTER), and C comes back INTER wide."""
    rows = T * TOPK
    beta, linear_beta = 4.0, 25.0
    B, Bs = _weights(2 * INTER, H, device, seed=13)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)

    A = (torch.randn(T, H, device=device, dtype=torch.bfloat16) * 0.1)
    C = torch.zeros(rows, INTER, device=device, dtype=torch.bfloat16)

    bm = ma.mxfp4_moe_gateup_block_m(rows, n_active)
    ids = _route(T, n_active, device, seed=7)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, bm, E)

    ma.mxfp4_moe_gateup(A, Br, Bsr, C, None, sorted_ids, expert_ids, ntpp,
                        nblk, bm, rows, TOPK, epilogue=ma.EPI_SITU,
                        situ_beta=beta, situ_linear_beta=linear_beta)
    torch.cuda.synchronize()

    both = _reference(A, _dequant(B, Bs), sorted_ids, expert_ids, bm, rows,
                      2 * INTER, TOPK)
    ref = _situ_glu(both[:, :INTER], both[:, INTER:], beta, linear_beta)
    live = torch.zeros(rows, dtype=torch.bool, device=device)
    live[sorted_ids[sorted_ids < rows].long()] = True
    assert _rel_err(ref, C, live) < 2e-2


@pytest.mark.parametrize("n_steps", [1, 2, 3, 7, 14])
def test_down_n_steps_does_not_change_the_result(device, n_steps):
    """`n_steps` is a scheduling knob -- how many column chunks one workgroup
    sweeps against its staged A tile -- so every legal value must agree."""
    T, n_active = 64, 48
    rows = T * TOPK
    B, Bs = _weights(H, INTER, device, seed=17)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)

    A = (torch.randn(rows, INTER, device=device, dtype=torch.bfloat16) * 0.1)
    C = torch.zeros(rows, H, device=device, dtype=torch.bfloat16)

    bm = ma.mxfp4_moe_down_block_m(rows, n_active, INTER)
    ids = _route(T, n_active, device, seed=5)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, bm, E)
    nt = ma.mxfp4_moe_down_nt(rows, n_active)

    ma.mxfp4_moe_down(A, Br, Bsr, C, None, sorted_ids, expert_ids, ntpp,
                      nblk, bm, n_steps, nt, rows, top_k=1)
    torch.cuda.synchronize()

    ref = _reference(A, _dequant(B, Bs), sorted_ids, expert_ids, bm, rows, H, 1)
    live = torch.zeros(rows, dtype=torch.bool, device=device)
    live[sorted_ids[sorted_ids < rows].long()] = True
    assert _rel_err(ref, C, live) < 2e-2


def test_down_plan_widens_the_thin_decode_grid():
    """At decode the m-blocks are about the live expert count, so one chunk per
    workgroup leaves the A stage unamortised. The rule is off where the grid is
    too thin to give the columns up."""
    assert ma.mxfp4_moe_down_n_steps(384, H, 304, 16, 1) >= 4
    assert ma.mxfp4_moe_down_n_steps(48, H, 32, 16, 2) >= 3
    assert ma.mxfp4_moe_down_n_steps(48, H, 304, 16, 2) == 1


def test_repack_is_a_permutation(device):
    """The repack must move bytes, never change values.

    The one exception is MXFP4's negative zero: nibble 0x8 comes out as 0x0, so
    the magnitude nibble stays a plain table index. That changes the sign of a
    zero and nothing else, so the comparison below is on magnitudes.
    """
    B, Bs = _weights(64, 512, device, seed=23)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)

    assert Br.shape == (E, 512 // 32, 64, 16)
    assert Bsr.shape == (E, 512 // 32, 64)
    #  Same multiset of magnitudes, and the scales are a pure transpose.
    assert torch.equal(torch.bincount((B & 0x07).flatten(), minlength=8)
                       + torch.bincount(((B >> 4) & 0x07).flatten(), minlength=8),
                       torch.bincount((Br & 0x07).flatten(), minlength=8)
                       + torch.bincount(((Br >> 4) & 0x07).flatten(), minlength=8))
    #  Every sign survives except the ones sitting on a zero.
    def _signs(x):
        nz = (x & 0x07) != 0
        return torch.bincount((x >> 3)[nz].flatten(), minlength=2)
    assert torch.equal(_signs(B & 0x0F) + _signs(B >> 4),
                       _signs(Br & 0x0F) + _signs(Br >> 4))
    assert torch.equal(Bsr, Bs.permute(0, 2, 1).contiguous())


def test_rejects_unrepacked_weights(device):
    """A stock-layout B has the wrong rank, and must be refused rather than run."""
    rows = 64
    B, Bs = _weights(INTER, H, device, seed=29)
    A = torch.zeros(rows // TOPK, H, device=device, dtype=torch.bfloat16)
    C = torch.zeros(rows, INTER, device=device, dtype=torch.bfloat16)
    ids = _route(rows // TOPK, 16, device, seed=7)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, 16, E)

    with pytest.raises(Exception):
        ma.mxfp4_moe_gateup(A, B, Bs, C, None, sorted_ids, expert_ids, ntpp,
                            nblk, 16, rows, TOPK, epilogue=ma.EPI_NONE)


def test_rejects_bad_tile(device):
    rows = 64
    B, Bs = _weights(INTER, H, device, seed=31)
    Br, Bsr = ma.repack_mxfp4(B), ma.repack_mxfp4_scales(Bs)
    A = torch.zeros(rows // TOPK, H, device=device, dtype=torch.bfloat16)
    C = torch.zeros(rows, INTER, device=device, dtype=torch.bfloat16)
    ids = _route(rows // TOPK, 16, device, seed=9)
    sorted_ids, expert_ids, ntpp, nblk = _align(ids, 16, E)

    with pytest.raises(Exception):
        ma.mxfp4_moe_gateup(A, Br, Bsr, C, None, sorted_ids, expert_ids, ntpp,
                            nblk, 24, rows, TOPK, epilogue=ma.EPI_NONE)
