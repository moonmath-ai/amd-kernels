#!/usr/bin/env python
"""MLA a16w8 decode vs AITER's a16w8 MLA decode ASM kernel.

AITER's a16w8 kernel (bf16 Q against fp8 KV) is the only cell in its nhead=16 decode
surface with our dtypes, so it is the like-for-like bar. Both sides run in ONE process
on ONE shared paged fp8 KV pool, with the same Q, the same softmax scale, the same
end-aligned causal mask and the same page_size=1 shuffled slot permutation -- a kernel
that assumed contiguous slots would fail the validation step.

Everything is timed as a CUDA-GRAPH REPLAY. AITER's python op wrappers cost ~78 us of
CPU per call, so an eager loop measures its dispatch rather than its kernel at every
shape below S ~ 64K; replaying a captured graph removes the host cost from both sides.

AITER's `num_kv_splits` is SWEPT per shape and the best validated config is kept: a
single setting is wrong by up to 6x at low batch. Timing is round-robin -- every
candidate once per round, alternating direction, median over rounds -- because a
single long burst per candidate inherits whatever clock state the previous candidate
left behind, which is worth up to 17% on a power-capped part.

  python benchmark/bench_mla.py
  python benchmark/bench_mla.py --shapes 8:150000,16:150000
  AITER_PATH=/path/to/aiter python benchmark/bench_mla.py
"""
import argparse
import gc
import math
import os
import sys

import torch

import moonmath_amd as ma

LAT, ROPE = 512, 64
FUSED = LAT + ROPE  # 576
SCALE = FUSED**-0.5
KV_SCALE = 1.0
Q_LEN = 4  # AITER's a16w8 kernel rejects q_len > 4 on fp8 KV (it abort()s)
H = 16     # ... and its whole nhead=16 surface asserts nhead == 16

# (B, S). S=150000 is a long-context decode; 8192 is the short end.
SHAPES = [(1, 150000), (2, 150000), (8, 8192), (8, 32768),
          (8, 65536), (8, 150000), (16, 150000), (32, 8192)]

# aiter's persistent path takes a total split count. None = let aiter choose.
AITER_SPLITS = [None, 16, 32, 64, 128, 304]


def load_aiter():
    p = os.environ.get("AITER_PATH")
    if p:
        if not os.path.isfile(os.path.join(p, "hsa", "gfx942", "mla", "mla_asm.csv")):
            raise SystemExit(f"AITER_PATH={p} is not a complete aiter checkout "
                             "(missing hsa/gfx942/mla/mla_asm.csv)")
        sys.path.insert(0, os.path.abspath(p))
    try:
        import aiter
        import aiter.mla  # noqa: F401
        from aiter import dtypes
    except ImportError as e:
        raise SystemExit(f"aiter not importable ({e}). pip install -e '.[bench]', or set "
                         "AITER_PATH to a checkout with hsa/gfx942/mla/*.co built.")
    return aiter, dtypes


class Case:
    """One problem instance: Q, a paged fp8 KV pool on a shuffled slot permutation, and our op."""

    def __init__(self, B, S, seed=0):
        self.B, self.S = B, S
        dev = "cuda"
        g = torch.Generator(device=dev).manual_seed(seed)

        self.q_lat = (torch.randn(B, Q_LEN, H, LAT, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.q_pe = (torch.randn(B, Q_LEN, H, ROPE, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        self.o_lat = torch.zeros(B, Q_LEN, H, LAT, device=dev, dtype=torch.bfloat16)

        # kv_scale == 1.0, so the stored fp8 value IS the value the reference dequantizes.
        kv_fp8 = ((torch.randn(B, S, FUSED, generator=g, device=dev) * 0.5)
                  .to(torch.bfloat16).to(torch.float8_e4m3fnuz))

        # Scatter into a shared pool under a random permutation: page_size=1, so kv_indices is a
        # flat per-token slot list and nothing about it is contiguous.
        nslots = B * S
        self.pool = torch.empty(nslots, 1, FUSED, device=dev, dtype=torch.float8_e4m3fnuz)
        perm = torch.randperm(nslots, generator=g, device=dev).to(torch.int32)
        self.pool.view(nslots, FUSED).view(torch.uint8).index_copy_(
            0, perm.long(), kv_fp8.view(nslots, FUSED).view(torch.uint8))
        del kv_fp8
        self.kv_indices = perm
        self.kv_indptr = torch.arange(B + 1, device=dev, dtype=torch.int32) * S
        self.seq_lens = torch.full((B,), S, device=dev, dtype=torch.int32)
        torch.cuda.synchronize()

    def run(self):
        T = self.B * Q_LEN
        ma.mla_decode_a16w8(
            self.q_lat.view(T, H, LAT), self.q_pe.view(T, H, ROPE), self.pool,
            self.o_lat.view(T, H, LAT), self.seq_lens, self.kv_indices, self.kv_indptr,
            SCALE, KV_SCALE)

    def out(self):
        return self.o_lat

    def kv_bytes(self):
        return self.B * self.S * FUSED

    def reference(self, chunk=8192):
        """fp32 chunked streaming-softmax over the SAME dequantized pool.

        Draft position t attends KV [0, S - (q_len-1-t)) -- end-aligned causal, so the last
        position sees the whole sequence.
        """
        B, S = self.B, self.S
        qH = Q_LEN * H
        q = torch.cat([self.q_lat.float(), self.q_pe.float()], dim=-1).reshape(B, qH, FUSED)
        tpos = torch.arange(Q_LEN, device=q.device).repeat_interleave(H)
        lim = (S - (Q_LEN - 1 - tpos)).to(torch.int64)
        m = torch.full((B, qH), -float("inf"), device=q.device)
        l = torch.zeros(B, qH, device=q.device)
        acc = torch.zeros(B, qH, LAT, device=q.device)
        src = self.pool.view(B * S, FUSED)
        idx = self.kv_indices.view(B, S).long()
        for n0 in range(0, S, chunk):
            n1 = min(n0 + chunk, S)
            kv = src[idx[:, n0:n1].reshape(-1)].to(torch.bfloat16).float().view(B, n1 - n0, FUSED)
            s = torch.bmm(q, kv.transpose(1, 2)) * SCALE * KV_SCALE
            nidx = torch.arange(n0, n1, device=q.device)
            s = s.masked_fill_(nidx.view(1, 1, -1) >= lim.view(1, -1, 1), -float("inf"))
            nm = torch.maximum(m, s.amax(dim=-1))
            nm_f = torch.where(torch.isfinite(nm), nm, torch.zeros_like(nm))
            corr = torch.exp(torch.where(torch.isfinite(m), m, torch.full_like(m, -1e30)) - nm_f)
            pr = torch.nan_to_num(torch.exp(s - nm_f.unsqueeze(-1)), nan=0.0, posinf=0.0, neginf=0.0)
            acc = acc * corr.unsqueeze(-1) + torch.bmm(pr, kv[:, :, :LAT])
            l = l * corr + pr.sum(dim=-1)
            m = nm
            del kv, s, pr
        return (acc / l.clamp_min(1e-30).unsqueeze(-1)).view(B, Q_LEN, H, LAT)


class Aiter:
    """AITER a16w8, q_len 4, persistent, at one fixed num_kv_splits.

    Calls the two persistent-path ops directly with the scratch buffers hoisted out of the
    loop -- what aiter.mla.mla_decode_fwd itself does at nhead=16, minus the per-call
    torch.empty()s and python branching. That is strictly FAVOURABLE to aiter.
    """

    def __init__(self, aiter, dtypes, case, num_kv_splits):
        self.aiter = aiter
        B, S = case.B, case.S
        dev = "cuda"
        # q: [B*4, H, 576], token b*4+t <-> case.q_lat[b, t]
        self.q = torch.cat([case.q_lat, case.q_pe], dim=-1).reshape(B * Q_LEN, H, FUSED).contiguous()
        self.o = torch.empty(B * Q_LEN, H, LAT, dtype=torch.bfloat16, device=dev)
        self.kv = case.pool.view(B * S, 1, 1, FUSED)          # the SAME pool + permutation
        self.kv_indices, self.kv_indptr = case.kv_indices, case.kv_indptr
        self.qo_indptr = torch.arange(0, (B + 1) * Q_LEN, Q_LEN, dtype=torch.int32, device=dev)
        self.kv_last = torch.ones(B, dtype=torch.int32, device=dev)
        self.kv_scale = torch.full([1], KV_SCALE, dtype=torch.float32, device=dev)
        self.B = B

        nks = num_kv_splits if num_kv_splits is not None else 304
        info = aiter.get_mla_metadata_info_v1(
            B, Q_LEN, H, dtypes.bf16, dtypes.fp8, is_sparse=False, fast_mode=True,
            num_kv_splits=nks, intra_batch_mode=False)
        (self.wmd, self.windptr, self.wis,
         self.rindptr, self.rfm, self.rpm) = [torch.empty(sz, dtype=dt, device=dev)
                                              for (sz, dt) in info]
        aiter.get_mla_metadata_v1(
            self.qo_indptr, self.kv_indptr, self.kv_last, H, 1, False,
            self.wmd, self.wis, self.windptr, self.rindptr, self.rfm, self.rpm,
            page_size=1, kv_granularity=16, max_seqlen_qo=Q_LEN, uni_seqlen_qo=Q_LEN,
            fast_mode=True,
            max_split_per_batch=(num_kv_splits if num_kv_splits is not None else -1),
            intra_batch_mode=False, dtype_q_nope=dtypes.bf16, dtype_kv_nope=dtypes.fp8)
        torch.cuda.synchronize()

        npart = self.rpm.size(0)
        self.splits = nks
        self.logits = torch.empty((npart * Q_LEN, 1, H, LAT), dtype=torch.float32, device=dev)
        self.attn_lse = torch.empty((npart * Q_LEN, 1, H, 1), dtype=torch.float32, device=dev)

    def run(self):
        a = self.aiter
        a.mla_decode_stage1_asm_fwd(
            self.q, self.kv, self.qo_indptr, self.kv_indptr, self.kv_indices, self.kv_last,
            None, self.wmd, self.windptr, self.wis, Q_LEN, 1, 1, SCALE,
            self.logits, self.attn_lse, self.o, None, None, self.kv_scale, None, 1, 0, None, 0)
        a.mla_reduce_v1(self.logits, self.attn_lse, self.rindptr, self.rfm, self.rpm,
                        Q_LEN, self.splits, self.o, None)

    def out(self):
        return self.o.view(self.B, Q_LEN, H, LAT)


# ───────────────────────────── timing ─────────────────────────────
def bulk_mean_us(fn, iters, warmup):
    """Mean over `iters` back-to-back launches, ONE event pair around the whole loop."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def round_robin_median(fns, iters, warmup, rounds=7):
    """Time every candidate once per round, alternating direction, and take each one's median.

    One long burst per candidate is not a measurement on a power-capped part: a burst of a
    short kernel inherits the clock state the previous candidate left behind, which has been
    seen to move the same binary by 17%. Cycling inside each round gives every candidate the
    same thermal history; the median discards rounds where the clock was still moving.
    """
    names = list(fns)
    for n in names:
        for _ in range(warmup):
            fns[n]()
    torch.cuda.synchronize()
    samples = {n: [] for n in names}
    for r in range(rounds):
        for n in (names if r % 2 == 0 else names[::-1]):
            samples[n].append(bulk_mean_us(fns[n], iters, 0))
    return {n: sorted(v)[len(v) // 2] for n, v in samples.items()}


def make_graph(fn):
    """Capture one invocation into a CUDA graph, so replays carry no host cost."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def relerr(got, ref):
    got, ref = got.float(), ref.float()
    n = ref.abs().max().item()
    return (got - ref).abs().max().item() / n if n > 0 else float("nan")


def iters_for(B, S):
    work = B * S * H
    if work >= 8 * 65536 * 16:
        return 40, 15
    if work >= 8 * 8192 * 16:
        return 80, 20
    return 300, 50


def run_shape(aiter, dtypes, B, S, verbose):
    iters, warm = iters_for(B, S)
    case = Case(B, S)
    gc.collect()
    torch.cuda.empty_cache()

    # Sweep aiter's num_kv_splits; keep the fastest config that did not produce NaN.
    # `found`, not `best is None`: nks=None IS a valid config (aiter picks the split itself),
    # so None cannot double as the "nothing worked" sentinel.
    best, best_t, found, sweep = None, float("inf"), False, []
    for nks in AITER_SPLITS:
        try:
            cand = Aiter(aiter, dtypes, case, nks)
            cand.o.fill_(float("nan"))
            g = make_graph(cand.run)
            t = bulk_mean_us(g.replay, max(20, iters // 3), max(10, warm // 2))
            torch.cuda.synchronize()
            bad = torch.isnan(cand.o.float()).any().item()
            sweep.append((round(t, 2), f"nks={nks}", "NaN" if bad else "ok"))
            if not bad and t < best_t:
                best, best_t, found = nks, t, True
            del g, cand
        except Exception as e:
            sweep.append((float("nan"), f"nks={nks}", f"{type(e).__name__}: {str(e)[:50]}"))
        gc.collect()
        torch.cuda.empty_cache()
    if not found:
        raise SystemExit(f"every aiter config failed at B={B} S={S}: {sorted(sweep)}")
    if verbose:
        print(f"[sweep] B={B} S={S}: {sorted(sweep)}  -> nks={best}")

    ours, theirs = case, Aiter(aiter, dtypes, case, best)
    graphs = {"aiter": make_graph(theirs.run), "ours": make_graph(ours.run)}
    med = round_robin_median({n: g.replay for n, g in graphs.items()}, iters, warm)
    return med["ours"], med["aiter"], case.kv_bytes(), f"nks={best}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="", help="comma list of B:S (default: the table above)")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--validate-S", type=int, default=8192)
    ap.add_argument("-v", "--verbose", action="store_true", help="print the num_kv_splits sweep")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no ROCm/HIP device")
    torch.cuda.init()
    aiter, dtypes = load_aiter()
    shapes = ([tuple(int(x) for x in s.split(":")) for s in a.shapes.split(",")]
              if a.shapes else SHAPES)

    if not a.no_validate:
        S = a.validate_S
        case = Case(2, S)
        case.run()
        torch.cuda.synchronize()
        ref = case.reference()
        theirs = Aiter(aiter, dtypes, case, None)
        theirs.run()
        torch.cuda.synchronize()
        print(f"[validate] B=2 S={S} H={H} q_len={Q_LEN} vs an fp32 reference over the same "
              f"dequantized KV\n[validate]   ours={relerr(case.out(), ref):.3e}   "
              f"aiter={relerr(theirs.out(), ref):.3e}")
        del case, theirs, ref
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n=== MLA a16w8 decode, H={H}, q_len={Q_LEN} -- microseconds (CUDA-graph replay) ===")
    print(f"{'B':>4} {'S':>8} {'KV MB':>8} | {'ours':>9} {'aiter':>9} {'speedup':>8} "
          f"| {'ours TB/s':>10} {'aiter TB/s':>10}")
    print("-" * 78)
    speedups = []
    for (B, S) in shapes:
        o_us, a_us, kvb, label = run_shape(aiter, dtypes, B, S, a.verbose)
        sp = a_us / o_us
        speedups.append(sp)
        print(f"{B:>4} {S:>8} {kvb / 1e6:>8.0f} | {o_us:>9.2f} {a_us:>9.2f} {sp:>7.2f}x "
              f"| {kvb / o_us / 1e6:>10.2f} {kvb / a_us / 1e6:>10.2f}   ({label})")
        gc.collect()
        torch.cuda.empty_cache()
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\ngeomean speedup {geo:.2f}x over {len(speedups)} shapes "
              f"({min(speedups):.2f}x - {max(speedups):.2f}x)")
    print("\nBoth sides read ONE shared paged fp8 pool (page_size=1, shuffled slots) with the same\n"
          "Q, scale and end-aligned causal mask. aiter's num_kv_splits is swept per shape and the\n"
          "best validated config kept; -v prints the sweep. aiter's a16w8 kernel is nhead=16 only\n"
          "and rejects q_len > 4 on fp8 KV, so H and q_len are fixed here.")


if __name__ == "__main__":
    main()
