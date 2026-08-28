#!/usr/bin/env python3
"""MXFP4 MoE GEMMs vs AITER's Triton MXFP4 kernel on MI300X.

Both projections of a Kimi-K3 MoE layer, over the same routing metadata and the
same weights:

    gate/up   C[rows, 2*I] = A[T, H]      @ dequant(w13[e])   fan-out over top_k
    down      C[rows, H]   = A[rows, I]   @ dequant(w2[e])    one row per (token, expert)

The activation epilogue is off on both sides, so the two kernels compute the
same dot products and the ratio is kernel speed, not a fused elementwise op.

Every config in ``TILES`` is timed at every shape and the fastest is reported,
so the baseline is the best AITER can do here rather than whatever it would
pick. Metadata is aligned separately per tile, so each config pays its own
padding.

Candidates are interleaved within a pass and the pass median is reported:
this VF is power-capped, and drift across a run is larger than the differences
between neighbouring tiles.

    python benchmark/bench_moe.py --markdown
"""

import argparse
import statistics

import torch
import triton.language as tl

from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

import moonmath_attention as ma

#  Kimi-K3: 896 routed experts of width 3072, top-16, over a 3584-wide latent.
#  INTER is the tensor-parallel shard of the expert width, so it names the
#  layout: 3072/8 at TP8, 3072/4 at TP4, and the full 3072 under expert
#  parallelism, where the experts are not sharded at all.
NUM_EXPERTS, HIDDEN, TOP_K = 896, 3584, 16
INTER_BY_LAYOUT = {"tp8": 384, "tp6": 512, "tp4": 768, "ep8": 3072}

#  The shapes a K3 server actually runs: two decode batch sizes, and two
#  chunked-prefill chunk sizes.
TOKENS = [8, 32, 8192, 16384]


def tile(bm, bn, bk=128, group=1, warps=4, stages=2, eu=0, kpack=1):
    return dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=group,
                num_warps=warps, num_stages=stages, waves_per_eu=eu,
                matrix_instr_nonkdim=16, kpack=kpack)


#  AITER tile candidates. The first two are the entries in AITER's own
#  aiter/ops/triton/configs/moe/gfx942-MOE-MX_FP4.json; the rest are what a
#  coordinate descent over every knob the kernel takes found at these shapes.
#  It is worth the extra candidates: BLOCK_SIZE_N=256 and waves_per_eu appear
#  in no AITER table and are worth 13-18% on down.
TILES = [
    tile(16, 64), tile(64, 128),                            # AITER's own table
    tile(32, 64), tile(16, 128), tile(32, 128), tile(64, 64),
    tile(16, 64, stages=1, warps=2), tile(16, 64, kpack=2, eu=4),
    tile(16, 128, eu=4), tile(64, 256, group=2, eu=2),
    tile(64, 128, kpack=2, eu=2), tile(64, 256, stages=1, kpack=2, eu=2),
]


def tile_name(cfg):
    """Only the knobs that vary across TILES, so the winner is readable."""
    name = f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}"
    for knob, tag, default in (("GROUP_SIZE_M", "g", 1), ("num_warps", "w", 4),
                               ("num_stages", "s", 2), ("kpack", "k", 1),
                               ("waves_per_eu", "e", 0)):
        if cfg[knob] != default:
            name += f"_{tag}{cfg[knob]}"
    return name


# ---------------------------------------------------------------------------
#  Routing
# ---------------------------------------------------------------------------
def align(topk_ids, block_m):
    """moe_align_block_size: sort rows by expert, pad each expert to block_m.

    Padding slots carry the sentinel ``rows``, which both kernels read as
    'do not write'.
    """
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    rows = flat.numel()

    counts = torch.bincount(flat, minlength=NUM_EXPERTS)
    padded = ((counts + block_m - 1) // block_m) * block_m
    zero = torch.zeros(1, dtype=torch.int64, device=dev)
    dst_base = torch.cat([zero, padded.cumsum(0)[:-1]])
    src_base = torch.cat([zero, counts.cumsum(0)[:-1]])

    order = torch.argsort(flat, stable=True)
    expert_of = flat[order]
    dst = dst_base[expert_of] + (torch.arange(rows, device=dev) - src_base[expert_of])

    total = int(padded.sum())
    sorted_ids = torch.full((total,), rows, dtype=torch.int32, device=dev)
    sorted_ids[dst] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(NUM_EXPERTS, device=dev, dtype=torch.int32), padded // block_m)
    ntpp = torch.tensor([total], dtype=torch.int32, device=dev)
    return sorted_ids, expert_ids, ntpp, expert_ids.numel()


def route(num_tokens, gen, device):
    """``top_k`` DISTINCT expert ids per token, uniform over the pool.

    K3 routes with ``noaux_tc``, whose per-expert bias exists to equalize expert
    load, so a uniform marginal is the shape a balanced router aims for. Drawing
    without replacement matters because a token cannot pick one expert twice,
    and it is what sets how many experts a batch touches at all.
    """
    ids = torch.argsort(torch.rand(num_tokens, NUM_EXPERTS, device=device,
                                   generator=gen), dim=1)[:, :TOP_K].to(torch.int32)
    active = int((torch.bincount(ids.reshape(-1).long(),
                                 minlength=NUM_EXPERTS) > 0).sum())
    return ids, active


# ---------------------------------------------------------------------------
#  Timing
# ---------------------------------------------------------------------------
def time_interleaved(fns, warmup, iters, passes):
    """Median ms/call per candidate, candidates round-robined inside a pass."""
    keys = list(fns)
    for k in keys:
        for _ in range(warmup):
            fns[k]()
    torch.cuda.synchronize()

    samples = {k: [] for k in keys}
    events = {k: (torch.cuda.Event(True), torch.cuda.Event(True)) for k in keys}
    for p in range(passes):
        for k in keys[p % len(keys):] + keys[:p % len(keys)]:
            start, end = events[k]
            start.record()
            for _ in range(iters):
                fns[k]()
            end.record()
            torch.cuda.synchronize()
            samples[k].append(start.elapsed_time(end) / iters)
    return {k: statistics.median(v) for k, v in samples.items()}


def max_rel_err(got, ref):
    """Max relative error over the elements carrying the tensor's magnitude.

    Chunked: at 262144 rows an fp32 upcast of a down output is several GB.
    """
    chunk = max(1, 2 ** 25 // ref.shape[1])
    scale = max(float(ref[i:i + chunk].float().abs().max())
                for i in range(0, ref.shape[0], chunk))
    worst = 0.0
    for i in range(0, ref.shape[0], chunk):
        a, b = got[i:i + chunk].float(), ref[i:i + chunk].float()
        live = b.abs() >= 0.01 * scale
        if live.any():
            worst = max(worst, float(((a - b).abs()[live] / b.abs()[live]).max()))
    return worst


# ---------------------------------------------------------------------------
#  One shape
# ---------------------------------------------------------------------------
class Projection:
    """One of the two GEMMs, with its weights and both callers' launchers."""

    def __init__(self, name, N, K, top_k, weights, device):
        self.name, self.N, self.K, self.top_k = name, N, K, top_k
        self.B, self.Bs = weights
        self.Br = ma.repack_mxfp4(self.B)
        self.Bsr = ma.repack_mxfp4_scales(self.Bs)
        self.a_scale = torch.ones(1, dtype=torch.float32, device=device)
        self.b_scale = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device)

    def flops(self, rows):
        return 2.0 * rows * self.K * self.N

    def ours(self, A, C, meta, rows, active):
        """Launcher plus the tile it was planned for.

        The down kernel stages its whole A tile in LDS, so it has tiles only
        for the TP-sharded intermediate sizes and returns 0 elsewhere; those
        shapes run on the general-K gate/up kernel with no epilogue.
        """
        block_m = (ma.mxfp4_moe_down_block_m(rows, active, self.K)
                   if self.name == "down" else 0)
        if block_m == 0:
            block_m = ma.mxfp4_moe_gateup_block_m(rows, active)
            if not ma.mxfp4_moe_gateup_supports_k(self.K):
                raise ValueError(f"neither kernel serves K={self.K}")
            sorted_ids, expert_ids, ntpp, n_blocks = meta(block_m)
            def run():
                ma.mxfp4_moe_gateup(A, self.Br, self.Bsr, C, None, sorted_ids,
                                    expert_ids, ntpp, n_blocks, block_m, rows,
                                    self.top_k, epilogue=ma.EPI_NONE)
            return run, block_m, "gate/up"

        sorted_ids, expert_ids, ntpp, n_blocks = meta(block_m)
        _, nt, n_steps = ma.mxfp4_moe_down_plan(rows, active, self.N, self.K, n_blocks)
        def run():
            ma.mxfp4_moe_down(A, self.Br, self.Bsr, C, None, sorted_ids, expert_ids,
                              ntpp, n_blocks, block_m, n_steps, nt, rows, self.top_k)
        return run, block_m, "down"

    def aiter(self, A, C, ids, meta, rows, cfg):
        sorted_ids, expert_ids, ntpp, _ = meta(cfg["BLOCK_SIZE_M"])
        weights = torch.ones(rows, 1, dtype=torch.float32, device=A.device)
        def run():
            fused_moe_mxfp4(A, self.B, C, self.a_scale, self.b_scale, None, self.Bs,
                            weights, ids, sorted_ids, expert_ids, ntpp, False,
                            self.top_k, False, False, cfg, tl.bfloat16)
        return run


def bench_shape(num_tokens, projections, device, gen, args):
    rows = num_tokens * TOP_K
    ids, active = route(num_tokens, gen, device)
    meta_cache = {}

    def meta(block_m):
        if block_m not in meta_cache:
            meta_cache[block_m] = align(ids, block_m)
        return meta_cache[block_m]

    results = []
    for proj in projections:
        m = num_tokens if proj.name == "gate/up" else rows
        A = torch.randn(m, proj.K, dtype=torch.bfloat16, device=device,
                        generator=gen) * 0.1
        C_ours = torch.zeros(rows, proj.N, dtype=torch.bfloat16, device=device)
        C_aiter = torch.zeros(rows, 1, proj.N, dtype=torch.bfloat16, device=device)

        run_ours, block_m, kernel = proj.ours(A, C_ours, meta, rows, active)
        fns = {"ours": run_ours}
        for cfg in TILES:
            #  AITER's masked-K path reads out of bounds when BLOCK_SIZE_K does
            #  not divide K, which faults the context rather than raising.
            if proj.K % cfg["BLOCK_SIZE_K"]:
                continue
            fns[tile_name(cfg)] = proj.aiter(A, C_aiter, ids, meta, rows, cfg)

        #  Correctness before timing: a launcher that silently does not run
        #  leaves C at zeros and would otherwise time as a huge win.
        for fn in fns.values():
            fn()
        torch.cuda.synchronize()
        live = torch.zeros(rows, dtype=torch.bool, device=device)
        sorted_ids = meta(block_m)[0]
        live[sorted_ids[sorted_ids < rows].long()] = True
        ref_live = C_aiter.view(rows, proj.N)[live]
        err = max_rel_err(C_ours[live], ref_live)
        assert err < 5e-2, f"{proj.name} T={num_tokens}: rel err {err:.3g} vs AITER"

        ms = time_interleaved(fns, args.warmup_iters, args.benchmark_iters, args.passes)
        best = min((v, k) for k, v in ms.items() if k != "ours")

        results.append(dict(tokens=num_tokens, proj=proj.name, rows=rows,
                            active=active, kernel=kernel, block_m=block_m,
                            ours=ms["ours"], aiter=best[0], tile=best[1],
                            err=err, tflops=proj.flops(rows) / (ms["ours"] * 1e9),
                            all_tiles={k: v for k, v in ms.items() if k != "ours"}))
        del A, C_ours, C_aiter, ref_live
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", default="tp8", choices=sorted(INTER_BY_LAYOUT),
                   help="tensor/expert-parallel shard of the expert width")
    p.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    p.add_argument("--warmup-iters", type=int, default=5)
    p.add_argument("--benchmark-iters", type=int, default=20)
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--markdown", action="store_true", help="emit the README table")
    p.add_argument("--show-tiles", action="store_true",
                   help="print every AITER tile, not just the winner")
    args = p.parse_args()

    device = torch.device("cuda")
    inter = INTER_BY_LAYOUT[args.layout]
    gen = torch.Generator(device=device).manual_seed(0)

    def weights(N, K):
        B = torch.randint(0, 256, (NUM_EXPERTS, N, K // 2), dtype=torch.uint8,
                          device=device, generator=gen)
        #  A narrow exponent band keeps the products in bf16 range; the value
        #  distribution is irrelevant to timing and both sides see the same one.
        Bs = torch.randint(120, 134, (NUM_EXPERTS, N, K // 32), dtype=torch.uint8,
                           device=device, generator=gen)
        return B, Bs

    print(f"{torch.cuda.get_device_name(0)}  {args.layout.upper()}: "
          f"E={NUM_EXPERTS} H={HIDDEN} I={inter} top_k={TOP_K}  "
          f"median of {args.passes} passes x {args.benchmark_iters} iters")

    projections = [
        Projection("gate/up", 2 * inter, HIDDEN, TOP_K,
                   weights(2 * inter, HIDDEN), device),
        Projection("down", HIDDEN, inter, 1, weights(HIDDEN, inter), device),
    ]

    rows = []
    for num_tokens in args.tokens:
        for r in bench_shape(num_tokens, projections, device, gen, args):
            rows.append(r)
            print(f"  T={r['tokens']:<6} {r['proj']:<8} rows={r['rows']:<7}"
                  f" active={r['active']:<4} ours({r['kernel']}, BM={r['block_m']})"
                  f" {r['ours']:8.3f} ms  aiter({r['tile']}) {r['aiter']:8.3f} ms"
                  f"  {r['aiter']/r['ours']:5.2f}x  {r['tflops']:6.1f} TFLOP/s"
                  f"  relerr {r['err']:.1e}")
            if args.show_tiles:
                print("      " + "  ".join(f"{k} {v:.3f}"
                                           for k, v in sorted(r["all_tiles"].items())))

    if args.markdown:
        print(f"\n| Tokens | Rows | Projection | Ours (ms) | AITER best (ms) "
              f"| Best tile | Speedup | Ours TFLOP/s |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['tokens']} | {r['rows']} | {r['proj']} | **{r['ours']:.3f}** "
                  f"| {r['aiter']:.3f} | {r['tile']} | {r['aiter']/r['ours']:.2f}× "
                  f"| {r['tflops']:.0f} |")

    for name in ("gate/up", "down"):
        sel = [r for r in rows if r["proj"] == name]
        if sel:
            geo = statistics.geometric_mean([r["aiter"] / r["ours"] for r in sel])
            print(f"geomean speedup vs AITER, {name}: {geo:.2f}x")


if __name__ == "__main__":
    main()
