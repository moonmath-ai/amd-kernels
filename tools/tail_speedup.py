#!/usr/bin/env python3
"""Predict the dense tail KV-split speedup for a given attention shape.

The forward grid tiles into 384-row q-blocks (`QTilesPerWG·QTileRows`) and runs
in rounds of 304 CUs. A fractional last round (`wgs % 304`) strands CUs while its
CTAs still walk the FULL kv stream. The tail KV-split recovers it by splitting
each stranded q-block's kv work `G` ways across all CUs (FlashDecoding-style),
exporting fp32 (o,m,l) partials and merging them.

This reproduces, in pure host arithmetic, exactly what `plan_tail_split` in
`csrc/attention_kernel.hip` decides — the enable guards, the `G` search, and the
resulting wall-time model — so you can see whether a shape benefits *before*
running it. It is the structural ceiling: it models KV-block wall-time only and
ignores fixed per-CTA cost, the v_transpose precompute, and launch latency, so
measured gains land at or just under it.

    $ python tools/tail_speedup.py 1 32760 12          # B S H  (self-attn)
    $ python tools/tail_speedup.py 1 32760 12 --skv 4096   # cross-attn S_kv
    $ python tools/tail_speedup.py --scan                   # find the big winners
"""

import argparse
import math

# Kernel constants (keep in sync with csrc/attention_kernel.hip).
QBLOCK_ROWS = 384  # QTilesPerWG(24) * QTileRows(16)
KV_BLOCK = 64  # KvBlockRows
NUM_CUS = 304  # NumCUs
MAX_PARTS = 2 * NUM_CUS  # 608
MERGE_CONST = 3  # the "+3" merge proxy in the span model


def plan_tail(B, S, H, Skv=None):
    """Return a dict describing the tail plan + predicted speedup for (B, S, H).

    Mirrors plan_tail_split: bail (gain 1.0) when the grid is even, the tail is
    ≥95% of a full round, kv is too small to amortize the merge, or no split
    factor G beats running the full stranded round.
    """
    Skv = Skv or S
    nqb = math.ceil(S / QBLOCK_ROWS)  # q-blocks per (B,H) = CTAs/head
    nkb = math.ceil(Skv / KV_BLOCK)  # kv blocks
    wgs = nqb * B * H  # total CTAs
    rounds, tail = divmod(wgs, NUM_CUS)  # full CU-rounds, stranded CTAs
    info = dict(nqb=nqb, nkb=nkb, wgs=wgs, rounds=rounds, tail=tail, G=1, gain=1.0)

    if tail == 0 or tail > NUM_CUS * 19 // 20 or nkb < 48:
        info["reason"] = (
            "even grid"
            if tail == 0
            else "tail ≥95% of a full round" if tail > NUM_CUS * 19 // 20
            else "kv too small to amortize merge"
        )
        return info

    def span(g):
        return math.ceil(tail * g / NUM_CUS) * math.ceil(nkb / g) + MERGE_CONST

    best_g, best_span = 1, nkb + MERGE_CONST  # g=1 = the no-split baseline
    g_hi = min(MAX_PARTS // tail, nkb // 4)
    for g in range(2, g_hi + 1):
        s = span(g)
        if s < best_span:
            best_span, best_g = s, g

    if best_g == 1:
        info["reason"] = "no split factor beats the full stranded round"
        return info

    # Wall-time model: no-split = (rounds+1) full streams; split = rounds + span(G).
    gain = (rounds + 1) * nkb / (rounds * nkb + best_span)
    info.update(G=best_g, span=best_span, gain=gain, reason="split")
    return info


def _fmt(B, S, H, Skv=None):
    p = plan_tail(B, S, H, Skv)
    skv = Skv or S
    head = f"(B={B}, S={S}, H={H}" + (f", Skv={skv}" if skv != S else "") + ")"
    print(head)
    print(f"  nqb={p['nqb']}  nkb={p['nkb']}  wgs={p['wgs']}  "
          f"full_rounds={p['rounds']}  tail={p['tail']}")
    if p["gain"] > 1.0:
        print(f"  -> SPLIT  G={p['G']}  span={p['span']}  "
              f"predicted speedup = {p['gain']:.3f}x")
    else:
        print(f"  -> no split ({p['reason']})  speedup = 1.000x")


def _scan(top=20):
    """Sweep small/moderate shapes and print the biggest tail wins."""
    res = []
    for BH in (8, 12, 16, 24, 32, 48, 64):
        for S in range(2048, 131072 + 1, 1024):
            p = plan_tail(1, S, BH)
            if p["gain"] > 1.0:
                res.append((p["gain"], BH, S, p["rounds"], p["tail"], p["G"]))
    res.sort(reverse=True)
    print(f"{'gain':>7} {'B*H':>4} {'S':>7} {'rounds':>6} {'tail':>4} {'G':>3}")
    for gain, BH, S, r, tail, G in res[:top]:
        print(f"{gain:6.3f}x {BH:4d} {S:7d} {r:6d} {tail:4d} {G:3d}")
    print("\nRule of thumb: the tail wins big when full_rounds is 0–1 "
          "(small B*H and modest S — an under-occupied GPU).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("B", type=int, nargs="?", help="batch")
    ap.add_argument("S", type=int, nargs="?", help="query seq len")
    ap.add_argument("H", type=int, nargs="?", help="heads")
    ap.add_argument("--skv", type=int, default=None, help="K/V seq len (cross-attn)")
    ap.add_argument("--scan", action="store_true", help="list the highest-gain shapes")
    args = ap.parse_args()
    if args.scan:
        _scan()
    elif args.B and args.S and args.H:
        _fmt(args.B, args.S, args.H, args.skv)
    else:
        ap.error("give B S H (e.g. `1 32760 12`) or --scan")
