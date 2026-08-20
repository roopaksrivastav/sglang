"""Force torch.profiler on for serving requests via an env var.

``PipelineExecutor.profile_execution`` gates on ``batch.profile``, which is only
settable through the realtime/WebSocket request model -- the plain HTTP
``/v1/videos`` request has no ``profile`` field, and although
``VideoGenerationsRequest`` is ``extra="allow"`` the extra never reaches the
``Req``. Rather than plumb a new field end to end just to take a measurement,
override the gate:

    BCG_PROFILE=1              profile the first non-warmup request
    BCG_PROFILE_STEPS=2        denoising steps to capture (default 2)
    BCG_PROFILE_TAG=bcg        prefix for the trace filename, so the eager and
                               graph traces are told apart in one directory

Only the first eligible request is profiled; later ones run untouched, which
keeps the trace small and avoids paying profiler overhead on every request.
Warmup is still excluded -- profiling it would capture BCG *capture* rather than
replay, and the two are not comparable.
"""

from __future__ import annotations

import argparse
import pathlib
import py_compile

# examples/diffusion/wan_bcg_xpu/ -> repo root -> python/sglang
DEFAULT_SGLANG_ROOT = pathlib.Path(__file__).resolve().parents[3] / "python" / "sglang"

MARK = "# ===================== PROFILE FORCE (experiment) ====================="

REL = "multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py"

BLOCK = f"""

{MARK}
# Appended by examples/diffusion/wan_bcg_xpu/patch_profile_force.py.
import os as _pf_os

if _pf_os.environ.get("BCG_PROFILE", "") not in ("", "0"):
    import contextlib as _pf_contextlib
    import sys as _pf_sys

    _PF_STEPS = int(_pf_os.environ.get("BCG_PROFILE_STEPS", "2"))
    _PF_TAG = _pf_os.environ.get("BCG_PROFILE_TAG", "run")
    _pf_state = {{"done": False}}

    _pf_orig_profile_execution = PipelineExecutor.profile_execution

    @_pf_contextlib.contextmanager
    def _pf_profile_execution(self, batch, dump_rank: int = 0):
        if batch.is_warmup or _pf_state["done"]:
            with _pf_orig_profile_execution(self, batch, dump_rank=dump_rank):
                yield
            return

        _pf_state["done"] = True
        print(
            f"[PROFILE-FORCE] profiling request {{batch.request_id}} "
            f"({{_PF_STEPS}} denoising steps, tag={{_PF_TAG}})",
            file=_pf_sys.stderr,
            flush=True,
        )
        profiler = SGLDiffusionProfiler(
            # request_id becomes the trace filename prefix.
            request_id=f"{{_PF_TAG}}-{{batch.request_id}}",
            rank=get_world_rank(),
            full_profile=False,
            num_steps=_PF_STEPS,
            num_inference_steps=batch.num_inference_steps,
        )
        try:
            yield
        finally:
            profiler.stop(dump_rank=dump_rank)
            print(
                f"[PROFILE-FORCE] profiler stopped for tag={{_PF_TAG}}",
                file=_pf_sys.stderr,
                flush=True,
            )

    PipelineExecutor.profile_execution = _pf_profile_execution
    print(
        f"[PROFILE-FORCE] active (steps={{_PF_STEPS}}, tag={{_PF_TAG}})",
        file=_pf_sys.stderr,
        flush=True,
    )
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sglang-root", default=str(DEFAULT_SGLANG_ROOT))
    args = ap.parse_args()

    path = f"{args.sglang_root}/{REL}"
    src = open(path).read()
    if MARK in src:
        print(f"[profile-force] already instrumented: {path}")
    else:
        try:
            open(path + ".orig.bak", "x").write(src)
        except FileExistsError:
            pass
        with open(path, "a") as f:
            f.write(BLOCK)
        print(f"[profile-force] instrumented: {path}")
    py_compile.compile(path, doraise=True)
    print("[profile-force] compiles cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
