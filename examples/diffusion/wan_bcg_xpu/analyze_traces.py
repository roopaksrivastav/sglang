"""Compare two chrome traces (eager vs breakable-CUDA-graph) from the same request.

Reports, per trace:
  * wall span of the profiled window
  * device time split by category (kernel / memcpy / memset), which is the number
    that actually matters -- the host-side numbers in this runtime are misleading
    because eager submission never syncs.
  * top device kernels by total time
  * host-side event counts, to show the launch-overhead difference the graph is
    supposed to remove.

Usage: python analyze_traces.py A.trace.json.gz B.trace.json.gz
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def summarize(path: str) -> dict:
    events = load(path)
    by_cat_time: dict[str, float] = defaultdict(float)
    by_cat_count: dict[str, int] = defaultdict(int)
    kernels: dict[str, list] = defaultdict(lambda: [0.0, 0])
    span_lo, span_hi = None, None

    for e in events:
        cat = e.get("cat")
        if cat is None or e.get("ph") != "X":
            continue
        dur = e.get("dur", 0) or 0
        by_cat_time[cat] += dur
        by_cat_count[cat] += 1
        if cat in DEVICE_CATS:
            name = e.get("name", "?")
            kernels[name][0] += dur
            kernels[name][1] += 1
            ts = e.get("ts", 0)
            span_lo = ts if span_lo is None else min(span_lo, ts)
            span_hi = max(span_hi or 0, ts + dur)

    device_time = sum(by_cat_time[c] for c in DEVICE_CATS)
    return {
        "path": path,
        "device_us": device_time,
        "span_us": (span_hi - span_lo) if span_lo is not None else 0.0,
        "by_cat_time": dict(by_cat_time),
        "by_cat_count": dict(by_cat_count),
        "kernels": kernels,
        "n_events": len(events),
    }


def shorten(name: str, width: int = 58) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = (summarize(p) for p in sys.argv[1:3])

    print("=" * 96)
    print(
        f"{'':<34}{'A: ' + a['path'].split('/')[-1][:24]:>30}{'B: ' + b['path'].split('/')[-1][:24]:>30}"
    )
    print("=" * 96)

    def row(label: str, va, vb, fmt="{:.1f}"):
        sa = fmt.format(va) if isinstance(va, float) else str(va)
        sb = fmt.format(vb) if isinstance(vb, float) else str(vb)
        ratio = f"{vb / va:.2f}x" if isinstance(va, (int, float)) and va else ""
        print(f"{label:<34}{sa:>30}{sb:>30}   {ratio}")

    row("device busy time (ms)", a["device_us"] / 1e3, b["device_us"] / 1e3)
    row("device span, first..last (ms)", a["span_us"] / 1e3, b["span_us"] / 1e3)
    row(
        "device utilisation (%)",
        100 * a["device_us"] / a["span_us"] if a["span_us"] else 0.0,
        100 * b["device_us"] / b["span_us"] if b["span_us"] else 0.0,
    )
    print()
    for cat in (
        "kernel",
        "gpu_memcpy",
        "gpu_memset",
        "cpu_op",
        "xpu_runtime",
        "gpu_user_annotation",
        "python_function",
        "user_annotation",
    ):
        if cat in a["by_cat_count"] or cat in b["by_cat_count"]:
            row(
                f"  {cat}: count",
                a["by_cat_count"].get(cat, 0),
                b["by_cat_count"].get(cat, 0),
            )
            row(
                f"  {cat}: time (ms)",
                a["by_cat_time"].get(cat, 0.0) / 1e3,
                b["by_cat_time"].get(cat, 0.0) / 1e3,
            )
    row("total trace events", a["n_events"], b["n_events"])

    # Union of the top kernels from both sides, so a kernel that only appears in
    # one of the two is still visible rather than silently dropped.
    top = set()
    for s in (a, b):
        top |= {
            k for k, _ in sorted(s["kernels"].items(), key=lambda kv: -kv[1][0])[:14]
        }
    print()
    print("-" * 96)
    print(f"{'device kernel':<60}{'A ms (n)':>17}{'B ms (n)':>17}")
    print("-" * 96)
    for name in sorted(
        top, key=lambda n: -(a["kernels"].get(n, [0])[0] + b["kernels"].get(n, [0])[0])
    ):
        ta, na = a["kernels"].get(name, [0.0, 0])
        tb, nb = b["kernels"].get(name, [0.0, 0])
        print(
            f"{shorten(name):<60}{f'{ta / 1e3:.1f} ({na})':>17}{f'{tb / 1e3:.1f} ({nb})':>17}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
