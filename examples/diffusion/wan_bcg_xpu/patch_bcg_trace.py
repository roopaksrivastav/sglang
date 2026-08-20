"""Instrument the breakable CUDA graph (BCG) machinery with capture/replay tracing.

Answers two questions the previous attempt could not:

  1. *What* is captured, and where are the graph broken into segments —
     each break point is reported with the exact submodule path that caused it
     (e.g. ``blocks.0.attn1 (UlyssesAttention)``).
  2. *Where* replay dies — with ``BCG_TRACE_SYNC=1`` every segment replay and
     every eager break function is followed by a device synchronize, so the
     last line printed before a SIGSEGV names the exact segment or break point
     that faulted instead of leaving us with a bare exit code 139.

Both trace blocks are appended to the end of their module rather than spliced
into the middle, so this does not depend on any surrounding source matching a
specific sglang revision. Rebinding at end-of-module still works because the
importers of ``eager_on_graph`` (the diffusion attention layer) import this
module first and therefore pick up the wrapped version.

Tracing is off unless ``BCG_TRACE=1`` is in the environment, so a patched tree
behaves exactly like a stock one by default.

This is debug instrumentation, not a feature: it edits the installed sources in
place (keeping a ``.orig.bak`` next to each) and is meant to be applied to a
scratch checkout while bringing BCG up on a new model. Revert with::

    git checkout -- python/sglang/srt/model_executor/runner_backend_utils \\
                    python/sglang/multimodal_gen/runtime/breakable_cuda_graph

Usage:  python patch_bcg_trace.py [--sglang-root .../python/sglang]
"""

from __future__ import annotations

import argparse
import pathlib

# examples/diffusion/wan_bcg_xpu/ -> repo root -> python/sglang
DEFAULT_SGLANG_ROOT = pathlib.Path(__file__).resolve().parents[3] / "python" / "sglang"

MARK = "# ===================== BCG TRACE (experiment) ====================="

LOWLEVEL_REL = (
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/"
    "breakable_cuda_graph.py"
)
RUNNER_REL = "multimodal_gen/runtime/breakable_cuda_graph/runner.py"


LOWLEVEL_BLOCK = f'''

{MARK}
# Appended by examples/diffusion/wan_bcg_xpu/patch_bcg_trace.py.
# Active only when BCG_TRACE is set.
import os as _bcg_os

_BCG_TRACE = _bcg_os.environ.get("BCG_TRACE", "") not in ("", "0")
_BCG_TRACE_SYNC = _bcg_os.environ.get("BCG_TRACE_SYNC", "") not in ("", "0")

if _BCG_TRACE:
    import sys as _bcg_sys

    def _bcg_log(msg: str) -> None:
        # stderr + flush: a SIGSEGV takes the process down without flushing
        # buffered stdout, which is why the first attempt saw no diagnostics.
        print(f"[BCGTRACE] {{msg}}", file=_bcg_sys.stderr, flush=True)

    def _bcg_sync(what: str) -> None:
        # NB: only ever call this where a synchronize is *legal*. Syncing while a
        # segment is open raises "wait cannot be called for a queue which is
        # recording to a command graph", which masks the real failure.
        if not _BCG_TRACE_SYNC:
            return
        get_device_module().synchronize()

    def _bcg_describe(obj, depth: int = 0) -> str:
        if torch.is_tensor(obj):
            dt = str(obj.dtype).replace("torch.", "")
            return f"T{{tuple(obj.shape)}}:{{dt}}@{{obj.device}}"
        if isinstance(obj, (list, tuple)) and depth < 2:
            head = ",".join(_bcg_describe(o, depth + 1) for o in list(obj)[:3])
            more = ",..." if len(obj) > 3 else ""
            return f"{{type(obj).__name__}}[{{len(obj)}}]({{head}}{{more}})"
        if hasattr(obj, "num_values"):
            return f"Boxed[{{obj.num_values}}]"
        return type(obj).__name__

    def _bcg_site(args) -> str:
        """Identity of the module whose forward triggered this break point.

        ``_bcg_path`` is stamped onto every submodule by the runner-side trace
        just before capture; without it we can still name the class.
        """
        if args and isinstance(args[0], torch.nn.Module):
            mod = args[0]
            path = getattr(mod, "_bcg_path", None) or "<unnamed>"
            return f"{{path}} ({{type(mod).__name__}})"
        return "<free function>"

    # ---- break points: wrap eager_on_graph so every split is announced ----
    _bcg_orig_eager_on_graph = eager_on_graph

    def eager_on_graph(enable: bool):  # noqa: F811 - deliberate rebind
        _inner_decorator = _bcg_orig_eager_on_graph(enable)

        def decorator(inner):
            wrapped = _inner_decorator(inner)
            if not enable:
                return wrapped

            def traced(*args, **kwargs):
                capture = _current_capture_var.get()
                if capture is None:
                    # Not capturing: plain pass-through, and also the path taken
                    # when the module runs fully eager.
                    return wrapped(*args, **kwargs)

                graph = capture.cuda_graph
                seg_idx = len(graph._segments)
                site = _bcg_site(args)
                _bcg_log(
                    f"capture:   segment #{{seg_idx}} ENDS at break -> {{site}}"
                )
                out = wrapped(*args, **kwargs)
                sites = getattr(graph, "_bcg_sites", None)
                if sites is None:
                    sites = graph._bcg_sites = []
                sites.append(site)
                _bcg_log(
                    f"capture:   break #{{len(graph._break_fns) - 1}} runs EAGER "
                    f"-> {{site}} out={{_bcg_describe(out)}}; segment "
                    f"#{{len(graph._segments)}} now open"
                )
                return out

            return traced

        return decorator

    # ---- capture segment boundaries ----
    _bcg_orig_begin = BreakableCUDAGraphCapture._begin_new_segment
    _bcg_orig_end = BreakableCUDAGraphCapture._end_current_segment

    def _bcg_traced_begin(self) -> None:
        # No sync here: capture_begin leaves the queue recording, so a
        # synchronize would itself raise and hide the genuine capture error.
        _bcg_orig_begin(self)
        _bcg_log(
            f"capture:   capture_begin ok (segment "
            f"#{{len(self.cuda_graph._segments)}}, "
            f"{{type(self._current_graph).__name__}}, pool={{self._pool}})"
        )

    def _bcg_traced_end(self) -> None:
        idx = len(self.cuda_graph._segments)
        _bcg_orig_end(self)
        _bcg_sync("capture_end")
        _bcg_log(f"capture:   capture_end ok (segment #{{idx}} sealed)")

    BreakableCUDAGraphCapture._begin_new_segment = _bcg_traced_begin
    BreakableCUDAGraphCapture._end_current_segment = _bcg_traced_end

    # ---- replay: the segfault localizer ----
    _bcg_orig_replay = BreakableCUDAGraph.replay

    def _bcg_traced_replay(self) -> None:
        sites = getattr(self, "_bcg_sites", [])
        stream = get_device_module().current_stream()
        token = _current_stream_var.set(stream)
        _bcg_log(
            f"replay: BEGIN {{len(self._segments)}} segment(s), "
            f"{{len(self._break_fns)}} eager break(s), stream={{stream}}"
        )
        try:
            for i, seg in enumerate(self._segments):
                _bcg_log(f"replay:   segment #{{i}} -> replay()")
                seg.replay()
                _bcg_sync("replay_seg")
                _bcg_log(f"replay:   segment #{{i}} ok")
                if i < len(self._break_fns):
                    site = sites[i] if i < len(sites) else "<unknown site>"
                    _bcg_log(f"replay:   break #{{i}} -> eager {{site}}")
                    self._break_fns[i]()
                    _bcg_sync("replay_break")
                    _bcg_log(f"replay:   break #{{i}} ok")
        finally:
            _current_stream_var.reset(token)
        _bcg_log("replay: END ok")

    # ---- replay: the slowness localizer ----
    # BCG_TRACE_TIME is the low-overhead alternative to the per-line trace
    # above: one summary line per replay instead of ~120, so it can run in a
    # timing measurement. It splits a replay into graph-segment time vs
    # eager-break time to show which side the cost is on. Pair with
    # BCG_TRACE_SYNC=1 for true device-time attribution (the syncs cost
    # wall-clock but stop the async queue from smearing time across entries);
    # with SYNC=0 the numbers are host submit cost only.
    def _bcg_timed_replay(self) -> None:
        import time as _bcg_time

        sites = getattr(self, "_bcg_sites", [])
        stream = get_device_module().current_stream()
        token = _current_stream_var.set(stream)
        t_seg = 0.0
        t_brk = 0.0
        worst_seg = (0.0, -1)
        worst_brk = (0.0, -1)
        t0 = _bcg_time.perf_counter()
        try:
            for i, seg in enumerate(self._segments):
                a = _bcg_time.perf_counter()
                seg.replay()
                _bcg_sync("replay_seg")
                d = _bcg_time.perf_counter() - a
                t_seg += d
                if d > worst_seg[0]:
                    worst_seg = (d, i)
                if i < len(self._break_fns):
                    a = _bcg_time.perf_counter()
                    self._break_fns[i]()
                    _bcg_sync("replay_break")
                    d = _bcg_time.perf_counter() - a
                    t_brk += d
                    if d > worst_brk[0]:
                        worst_brk = (d, i)
        finally:
            _current_stream_var.reset(token)
        total = _bcg_time.perf_counter() - t0
        n_seg = len(self._segments)
        n_brk = len(self._break_fns)
        ws = sites[worst_seg[1]] if 0 <= worst_seg[1] < len(sites) else "?"
        wb = sites[worst_brk[1]] if 0 <= worst_brk[1] < len(sites) else "?"
        _bcg_log(
            f"replay: total={{total * 1e3:.2f}}ms | "
            f"{{n_seg}} graph segments={{t_seg * 1e3:.2f}}ms "
            f"(mean {{t_seg / max(n_seg, 1) * 1e3:.3f}}ms, "
            f"worst #{{worst_seg[1]}} {{worst_seg[0] * 1e3:.2f}}ms before {{ws}}) | "
            f"{{n_brk}} eager breaks={{t_brk * 1e3:.2f}}ms "
            f"(mean {{t_brk / max(n_brk, 1) * 1e3:.3f}}ms, "
            f"worst #{{worst_brk[1]}} {{worst_brk[0] * 1e3:.2f}}ms at {{wb}})"
        )

    if _bcg_os.environ.get("BCG_TRACE_TIME", "") not in ("", "0"):
        BreakableCUDAGraph.replay = _bcg_timed_replay
    else:
        BreakableCUDAGraph.replay = _bcg_traced_replay

    _bcg_log(
        f"instrumentation active (sync={{_BCG_TRACE_SYNC}}, is_xpu={{_is_xpu}}, "
        f"graph_cls={{'torch.xpu.XPUGraph' if _is_xpu else 'torch.cuda.CUDAGraph'}})"
    )
'''


RUNNER_BLOCK = f"""

{MARK}
# Appended by examples/diffusion/wan_bcg_xpu/patch_bcg_trace.py.
# Active only when BCG_TRACE is set.
import os as _bcg_os

if _bcg_os.environ.get("BCG_TRACE", "") not in ("", "0"):
    import sys as _bcg_sys

    def _bcg_log(msg: str) -> None:
        print(f"[BCGTRACE] {{msg}}", file=_bcg_sys.stderr, flush=True)

    def _bcg_shape(value) -> str:
        return str(_signature_summary_leaf(_signature_leaf(value)))

    _bcg_orig_capture = BaseBreakableCudaGraphRunner._capture

    def _bcg_traced_capture(self, kwargs, key):
        # Stamp a module path on every submodule so the low-level break-point
        # logger can name the exact attention module that split the graph.
        for name, module in self.transformer.named_modules():
            module._bcg_path = name or "<root>"

        _bcg_log("=" * 70)
        _bcg_log(
            f"capture: START module={{type(self.transformer).__name__}} "
            f"device={{self.device}} pool={{self._pool}}"
        )
        for name in sorted(kwargs):
            _bcg_log(f"capture:   input {{name}} = {{_bcg_shape(kwargs[name])}}")

        try:
            entry = _bcg_orig_capture(self, kwargs, key)
        except BaseException:
            # try_capture() catches this and logs only str(e), which loses the
            # frame that actually did the illegal graph operation. Print the
            # traceback here, then re-raise so the eager fallback still happens.
            import traceback as _bcg_tb

            _bcg_log("capture: FAILED -- traceback follows")
            _bcg_tb.print_exc(file=_bcg_sys.stderr)
            _bcg_sys.stderr.flush()
            raise

        sites = getattr(entry.graph, "_bcg_sites", [])
        _bcg_log(
            f"capture: DONE {{entry.num_segments}} graph segment(s) + "
            f"{{len(entry.graph._break_fns)}} eager break(s); "
            f"{{len(entry.static_leaves)}} static input tensor(s)"
        )
        _bcg_log(f"capture: output = {{_bcg_shape(entry.output)}}")
        for i, site in enumerate(sites):
            _bcg_log(f"capture:   [eager break #{{i}}] {{site}}")
        _bcg_log(
            "capture: CAPTURED = everything except the break points above "
            "(patch/proj/norm/modulation/rope/ffn/out); "
            "EAGER = the break points"
        )
        _bcg_log("=" * 70)
        return entry

    BaseBreakableCudaGraphRunner._capture = _bcg_traced_capture

    _bcg_orig_runner_replay = BaseBreakableCudaGraphRunner.replay

    def _bcg_traced_runner_replay(self, entry, kwargs):
        _bcg_log(
            f"runner: REPLAY entry with {{entry.num_segments}} segment(s); "
            f"copying {{len(entry.static_leaves)}} live tensor(s) into static bufs"
        )
        out = _bcg_orig_runner_replay(self, entry, kwargs)
        _bcg_log("runner: REPLAY returned ok")
        return out

    BaseBreakableCudaGraphRunner.replay = _bcg_traced_runner_replay

    _bcg_orig_call = BaseBreakableCudaGraphRunner.__call__

    @torch.no_grad()
    def _bcg_traced_call(self, **kwargs):
        if self._disabled_reason is not None:
            _bcg_log(f"runner: CALL -> eager (BCG disabled: {{self._disabled_reason}})")
            return self.transformer(**kwargs)
        key = self._signature(kwargs)
        if key in self.entries:
            verdict = "signature HIT -> replay"
        elif key in self._blocked:
            verdict = "signature BLOCKED (capture failed earlier) -> eager"
        elif self._should_capture_on_call(key):
            verdict = "signature MISS, in warmup -> capture then replay"
        else:
            verdict = "signature MISS, serving -> eager"
        _bcg_log(
            f"runner: CALL {{verdict}} (captured entries={{len(self.entries)}})"
        )
        return _bcg_orig_call(self, **kwargs)

    BaseBreakableCudaGraphRunner.__call__ = _bcg_traced_call

    _bcg_log("runner instrumentation active")
"""


def _append_once(path: str, block: str) -> None:
    src = open(path).read()
    if MARK in src:
        print(f"[bcg-trace] already instrumented: {path}")
        return
    try:
        open(path + ".orig.bak", "x").write(src)
    except FileExistsError:
        pass
    with open(path, "a") as f:
        f.write(block)
    print(f"[bcg-trace] instrumented: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-root", default=str(DEFAULT_SGLANG_ROOT))
    args = ap.parse_args()

    for rel, block in ((LOWLEVEL_REL, LOWLEVEL_BLOCK), (RUNNER_REL, RUNNER_BLOCK)):
        path = f"{args.sglang_root}/{rel}"
        open(path)  # fail loudly with the real path if the layout differs
        _append_once(path, block)

    # Byte-compile both so a syntax error surfaces now, not mid-run in a worker.
    import py_compile

    for rel in (LOWLEVEL_REL, RUNNER_REL):
        py_compile.compile(f"{args.sglang_root}/{rel}", doraise=True)
    print("[bcg-trace] both modules compile cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
