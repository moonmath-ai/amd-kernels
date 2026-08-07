"""MLA absorbed-decode ops vs an fp32 reference (sanity).

The reference dequantizes the SAME fp8 KV tensor the kernel reads, so the only
error under test is the kernel's own bf16 QK/PV arithmetic, not the quantization.
Queries are peaked (x QPEAK) so the softmax is far from uniform and a wrong
causal limit or a wrong query row cannot hide behind a near-average output.

Covers the q_len=1 ops (mla_decode_a16w8*) and the multi-query ops
(mla_decode_a16w8_multiq*), contiguous and device-driven paged.
"""

from itertools import accumulate

import pytest
import torch

import moonmath_attention as ma

LAT, ROPE = 512, 64
FUSED = LAT + ROPE  # 576
SCALE = FUSED**-0.5
KV_SCALE = 1.0 / 32.0
QPEAK = 3.0
REL_L2 = 3e-3  # bf16 QK/PV against an fp32 reference over the same dequantized KV


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")
    return torch.device("cuda")


def _rand_kv(B, S, device, seed):
    """Fused [B, S, 576] fp8 e4m3-fnuz KV at per-tensor KV_SCALE."""
    torch.manual_seed(seed)
    kv = torch.randn(B, S, FUSED, device=device) * KV_SCALE
    return (kv / KV_SCALE).to(torch.float8_e4m3fnuz)


def _rand_q(B, q_len, H, device, seed):
    torch.manual_seed(seed + 1)
    shape = (B, H) if q_len is None else (B, q_len, H)
    q_lat = (torch.randn(*shape, LAT, device=device) * QPEAK).to(torch.bfloat16)
    q_pe = (torch.randn(*shape, ROPE, device=device) * QPEAK).to(torch.bfloat16)
    o_lat = torch.zeros(*shape, LAT, dtype=torch.bfloat16, device=device)
    return q_lat, q_pe, o_lat


def _reference(q_lat, q_pe, kv, seq_lens):
    """fp32 absorbed-decode MLA. q_* are [B, q_len, H, *]; kv is the fp8 slab, seq_lens a list.

    Draft position t attends kv[b, : seq_lens[b] - q_len + t + 1] (end-aligned causal).
    """
    B, q_len, H, _ = q_lat.shape
    out = torch.empty(B, q_len, H, LAT, dtype=torch.float32, device=q_lat.device)
    for b in range(B):
        k = kv[b].float() * KV_SCALE  # [S, 576]
        k_lat, k_pe = k[:, :LAT], k[:, LAT:]
        for t in range(q_len):
            end = seq_lens[b] - q_len + t + 1
            score = (
                q_lat[b, t].float() @ k_lat[:end].T + q_pe[b, t].float() @ k_pe[:end].T
            ) * SCALE
            out[b, t] = torch.softmax(score, dim=-1) @ k_lat[:end]
    return out


def _rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _paged_pool(kv, seq_lens, device, seed):
    """Scatter the per-request KV rows into a shared pool; return (pool, indices, indptr).

    Slots are a random permutation, so a kernel that assumed contiguity would fail.
    """
    B = kv.shape[0]
    total = sum(seq_lens)
    torch.manual_seed(seed + 2)
    slots = torch.randperm(total, device=device, dtype=torch.int32)
    pool = torch.zeros(total, 1, FUSED, dtype=torch.float8_e4m3fnuz, device=device)
    off = 0
    for b in range(B):
        sel = slots[off : off + seq_lens[b]].long()
        pool[sel, 0] = kv[b, : seq_lens[b]]
        off += seq_lens[b]
    indptr = torch.tensor(
        [0, *accumulate(seq_lens)], dtype=torch.int32, device=device
    )
    return pool, slots, indptr


# ---- q_len = 1 -------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("H", [12, 16])
def test_single_query_matches_reference(device, H):
    B, S = 2, 2048
    kv = _rand_kv(B, S, device, seed=11)
    q_lat, q_pe, o_lat = _rand_q(B, None, H, device, seed=11)
    ma.mla_decode_a16w8(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)
    ref = _reference(q_lat.unsqueeze(1), q_pe.unsqueeze(1), kv, [S] * B)
    assert _rel_l2(o_lat.unsqueeze(1), ref) < REL_L2


# ---- multi-query, contiguous ----------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("q_len", [4, 5, 8])
@pytest.mark.parametrize("H", [12, 16])
def test_multiq_contiguous_matches_reference(device, q_len, H):
    B, S = 2, 2048
    kv = _rand_kv(B, S, device, seed=23)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, H, device, seed=23)
    ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)
    assert _rel_l2(o_lat, _reference(q_lat, q_pe, kv, [S] * B)) < REL_L2


@pytest.mark.gpu
def test_multiq_causal_window_is_end_aligned(device):
    """Position t must see exactly S - q_len + t + 1 tokens, not the whole slab.

    Poisoning the tail rows changes the last draft position's output and leaves
    the first one bit-identical; if the causal limit were flat, both would move.
    """
    B, S, H, q_len = 1, 2048, 16, 4
    kv = _rand_kv(B, S, device, seed=31)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, H, device, seed=31)
    ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)
    base = o_lat.clone()

    kv[:, S - q_len + 1 :] = _rand_kv(B, q_len - 1, device, seed=99)  # only pos > 0 sees these
    ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)
    assert torch.equal(o_lat[:, 0], base[:, 0]), "position 0 read past its causal limit"
    assert not torch.equal(o_lat[:, -1], base[:, -1]), "last position missed the tail tokens"


# ---- multi-query, device-driven paged --------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("q_len", [4, 8])
def test_multiq_paged_matches_reference(device, q_len):
    """Ragged batch over a scattered pool (MLA page_size=1)."""
    B, H = 3, 16
    seq_lens = [2048, 2064, 1024]
    S = max(seq_lens)
    kv = _rand_kv(B, S, device, seed=43)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, H, device, seed=43)
    pool, kv_indices, kv_indptr = _paged_pool(kv, seq_lens, device, seed=43)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    parts = ma.mla_decode_a16w8_multiq_plan_parts_q(B, S, q_len, H)
    assert parts >= 1
    ma.mla_decode_a16w8_multiq_paged_dev(
        q_lat, q_pe, pool, o_lat, lens, None, kv_indices, kv_indptr, parts, SCALE, KV_SCALE
    )
    assert _rel_l2(o_lat, _reference(q_lat, q_pe, kv, seq_lens)) < REL_L2


@pytest.mark.gpu
def test_multiq_paged_rows_remaps_query_and_output(device):
    """`rows[b]` picks the q/out batch row for KV-request b: reversing it reverses the batch."""
    B, H, q_len = 4, 16, 4
    seq_lens = [2048] * B
    kv = _rand_kv(B, seq_lens[0], device, seed=57)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, H, device, seed=57)
    pool, kv_indices, kv_indptr = _paged_pool(kv, seq_lens, device, seed=57)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    parts = ma.mla_decode_a16w8_multiq_plan_parts_q(B, seq_lens[0], q_len, H)

    ma.mla_decode_a16w8_multiq_paged_dev(
        q_lat, q_pe, pool, o_lat, lens, None, kv_indices, kv_indptr, parts, SCALE, KV_SCALE
    )
    identity = o_lat.clone()

    # KV-request b now serves query row B-1-b; feeding the reversed queries must
    # reproduce the identity result, reversed.
    rows = torch.arange(B - 1, -1, -1, dtype=torch.int32, device=device)
    o_rev = torch.zeros_like(o_lat)
    ma.mla_decode_a16w8_multiq_paged_dev(
        q_lat.flip(0).contiguous(), q_pe.flip(0).contiguous(), pool, o_rev,
        lens, rows, kv_indices, kv_indptr, parts, SCALE, KV_SCALE,
    )
    assert torch.equal(o_rev.flip(0), identity)


# ---- supported domain ------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("q_len", [1, 3, 9])
def test_multiq_rejects_q_len_out_of_domain(device, q_len):
    B, S, H = 1, 2048, 16
    kv = _rand_kv(B, S, device, seed=71)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, H, device, seed=71)
    with pytest.raises(ValueError, match="q_len"):
        ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)


@pytest.mark.gpu
def test_multiq_rejects_too_many_heads(device):
    B, S, q_len = 1, 2048, 4
    kv = _rand_kv(B, S, device, seed=73)
    q_lat, q_pe, o_lat = _rand_q(B, q_len, 17, device, seed=73)
    with pytest.raises(ValueError, match="H must be"):
        ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)


@pytest.mark.gpu
def test_multiq_rejects_q_len_1_rank(device):
    """The q_len=1 tensor rank belongs to mla_decode_a16w8, not to the multi-query op."""
    B, S, H = 1, 2048, 16
    kv = _rand_kv(B, S, device, seed=79)
    q_lat, q_pe, o_lat = _rand_q(B, None, H, device, seed=79)
    with pytest.raises(ValueError, match=r"\[B,q_len,H,512\]"):
        ma.mla_decode_a16w8_multiq(q_lat, q_pe, kv, o_lat, SCALE, KV_SCALE)
