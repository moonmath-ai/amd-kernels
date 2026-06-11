#!/usr/bin/env python3
"""Parse AMDGPU .s listing from hipcc -save-temps and emit HTML schedule/wait analysis."""

from __future__ import annotations

import html
import re
import sys
from collections import Counter
from pathlib import Path

# Reference notes (order-of-magnitude; exact numbers vary by chip/voltage)
LATENCY_NOTES = """
<ul>
<li><b>VMEM</b> (buffer/global load): typically ~400–800+ cycles; compiler schedules
<code>s_waitcnt vmcnt(N)</code> so at most N loads are in flight before consuming results.</li>
<li><b>LDS</b> (<code>ds_*</code>): ~20–28 cycles from issue to operand ready; <code>lgkmcnt</code> tracks.</li>
<li><b>MFMA 32×32×8 bf16</b> on CDNA3: 4-cycle issue interval for back-to-back; operands must be ready.</li>
<li><b>VALU</b>: most ops ~4 cycles; <code>v_perm</code>, <code>v_exp</code> may be longer.</li>
</ul>
"""


def parse_lines(path: Path) -> list[tuple[int, str]]:
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if not s.startswith(";"):
            out.append((i, line))
    return out


WAIT_RE = re.compile(
    r"s_waitcnt\s+(vmcnt\s*\(\s*(\d+)\s*\))?(?:\s+)?(lgkmcnt\s*\(\s*(\d+)\s*\))?"
)


def parse_waitcnt(line: str) -> tuple[int | None, int | None]:
    """Return (vmcnt_max, lgkmcnt_max) or None if not matched."""
    if "s_waitcnt" not in line:
        return None, None
    m = re.search(r"vmcnt\s*\(\s*(\d+)\s*\)", line)
    lg = re.search(r"lgkmcnt\s*\(\s*(\d+)\s*\)", line)
    return (
        int(m.group(1)) if m else None,
        int(lg.group(1)) if lg else None,
    )


def classify(line: str) -> str:
    t = line.strip()
    if t.startswith("v_mfma"):
        return "mfma"
    if "buffer_load" in t or "global_load" in t:
        return "vmem_load"
    if t.startswith("ds_"):
        return "lds"
    if "ds_write" in t or "ds_read" in t:
        return "lds"
    if "s_waitcnt" in t:
        return "waitcnt"
    if "wave barrier" in t or "s_barrier" in t:
        return "barrier"
    return "other"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    temp_asm = root / "temp" / "attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.s"
    root_asm = root / "csrc" / "attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.s"
    if len(sys.argv) > 1:
        asm = Path(sys.argv[1])
    elif temp_asm.is_file():
        asm = temp_asm
    else:
        asm = root_asm
    if not asm.is_file():
        print(f"Missing ISA file: {asm}", file=sys.stderr)
        print(
            "Compile with: make isa-report  (writes temp/*.s) or hipcc ... -save-temps ...",
            file=sys.stderr,
        )
        return 1

    rows = parse_lines(asm)
    kernel_name = None
    kernel_start = None
    for ln, line in rows:
        stripped = line.strip()
        if re.match(r"^attn_fwd_[A-Za-z0-9_]+:$", stripped):
            kernel_start = ln
            kernel_name = stripped[:-1]
            break
    if kernel_name is None:
        kernel_name = asm.stem

    counts = Counter()
    waits: list[dict] = []
    instr_idx = 0
    for ln, line in rows:
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("."):
            continue
        if re.match(r"^attn_fwd_[A-Za-z0-9_]+:$", stripped):
            continue
        instr_idx += 1
        cat = classify(raw)
        counts[cat] += 1
        if cat == "waitcnt":
            vm, lg = parse_waitcnt(raw)
            waits.append(
                {
                    "line": ln,
                    "instr_idx": instr_idx,
                    "text": stripped,
                    "vmcnt": vm,
                    "lgkmcnt": lg,
                }
            )

    # Pair waits with preceding VMEM burst (heuristic: lines since last wait)
    wait_pairs = []
    last_wait_line = kernel_start or 1
    for w in waits:
        gap = w["line"] - last_wait_line
        wait_pairs.append({**w, "lines_since_prev_wait": gap})
        last_wait_line = w["line"]

    vm_hist = Counter()
    lg_hist = Counter()
    for w in waits:
        if w["vmcnt"] is not None:
            vm_hist[w["vmcnt"]] += 1
        if w["lgkmcnt"] is not None:
            lg_hist[w["lgkmcnt"]] += 1

    out_html = asm.parent / "isa_attn_report.html"
    parts: list[str] = []
    h = parts.append
    h(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>ISA: {html.escape(kernel_name)}</title>"
    )
    h(
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:1.2rem;max-width:1200px}"
    )
    h(
        "table{border-collapse:collapse;margin:0.5rem 0} th,td{border:1px solid #ccc;padding:4px 8px;font-size:13px}"
    )
    h(
        "th{background:#eee} code{background:#f4f4f4;padding:1px 4px} .note{color:#444;font-size:14px}"
    )
    h("</style></head><body>")
    h(f"<h1>gfx942 ISA report: <code>{html.escape(asm.name)}</code></h1>")
    h(
        f"<p class='note'>Kernel symbol <code>{html.escape(kernel_name)}</code> starts at line ~{kernel_start} in the listing.</p>"
    )
    h(
        "<h2>Instruction mix (approximate)</h2><table><tr><th>Category</th><th>Count</th></tr>"
    )
    for k, v in counts.most_common():
        h(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>")
    h("</table>")

    h("<h2>Latency reference (typical)</h2>")
    h(LATENCY_NOTES)

    h("<h2><code>s_waitcnt</code> histogram</h2>")
    h("<table><tr><th>vmcnt(k)</th><th>count</th></tr>")
    for k in sorted(vm_hist.keys()):
        h(f"<tr><td>{k}</td><td>{vm_hist[k]}</td></tr>")
    h("</table>")
    h("<table><tr><th>lgkmcnt(k)</th><th>count</th></tr>")
    for k in sorted(lg_hist.keys()):
        h(f"<tr><td>{k}</td><td>{lg_hist[k]}</td></tr>")
    h("</table>")

    h("<h2>All <code>s_waitcnt</code> (order)</h2>")
    h(
        "<p>Column <em>Δlines</em> = source lines since previous wait — rough proxy for schedule distance.</p>"
    )
    h("<table><tr><th>#</th><th>line</th><th>Δlines</th><th>instruction</th></tr>")
    for i, w in enumerate(wait_pairs, 1):
        h(
            "<tr><td>%d</td><td>%d</td><td>%d</td><td><code>%s</code></td></tr>"
            % (i, w["line"], w["lines_since_prev_wait"], html.escape(w["text"]))
        )
    h("</table>")

    h("<h2>Parallelism / schedule observations</h2><ul>")
    h(
        "<li><b>K / V tile to LDS</b>: long runs of <code>buffer_load_dword … lds</code> with interleaved "
        "<code>m0</code> updates — many in-flight VMEM ops.</li>"
    )
    h(
        "<li><b>Main loop</b>: watch the <code>s_waitcnt vmcnt(...)</code> points before LDS-backed "
        "consumption; those are the schedule boundaries that matter most for source mapping.</li>"
    )
    h(
        "<li><b>Q in VGPRs</b>: loaded once; overlaps initial K fills (Q global read hides under K VMEM).</li>"
    )
    h(
        "<li><b>V overlap</b>: after QK, V tiles issue to the same LDS lines as K; softmax "
        "(<code>v_exp_f32</code>, <code>v_fma_f32</code>, …) is scheduled in the same region as "
        "<code>buffer_load_dword</code> from V — look for VMEM loads a few dozen lines before "
        "<code>v_exp_f32</code> in the loop body (not a hard barrier between V fill and exp).</li>"
    )
    h(
        "<li><b>PV</b>: <code>s_waitcnt vmcnt(0)</code> before PV MFMAs drains V→LDS; PV then uses LDS reads + MFMA.</li>"
    )
    h(
        "<li><b>QK/PV MFMA</b>: staggered LDS reads + <code>s_waitcnt lgkmcnt</code> around MFMA issue groups.</li>"
    )
    h("</ul>")

    full_text = asm.read_text(encoding="utf-8", errors="replace")
    h("<h2>Full listing</h2>")
    h("<details><summary>Show entire <code>.s</code> file (hipcc output)</summary>")
    h("<pre style='overflow:auto;max-height:70vh;font-size:11px;line-height:1.2'>")
    h(html.escape(full_text))
    h("</pre></details>")

    h("</body></html>")
    out_html.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_html} ({out_html.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
