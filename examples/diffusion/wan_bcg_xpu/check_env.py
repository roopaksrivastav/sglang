"""Preflight check for the Wan + breakable-CUDA-graph (BCG) recipe.

Run this before serve.sh. Every failure printed here has cost someone an hour of
staring at a server log, so the checks are deliberately blunt:

    python check_env.py
    python check_env.py --model /path/to/Wan2.1-T2V-1.3B-Diffusers

Exit code 0 means "go", 1 means at least one hard requirement is missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

OK = "  ok   "
WARN = " warn  "
FAIL = " FAIL  "

_failed = False


def report(status: str, label: str, detail: str = "") -> None:
    global _failed
    if status is FAIL:
        _failed = True
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def check_torch() -> object | None:
    try:
        import torch
    except ImportError as e:
        report(FAIL, "import torch", str(e))
        return None
    report(OK, "torch", torch.__version__)
    return torch


def check_device(torch) -> str | None:
    """Return the accelerator family in use, or None if there is no device."""
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        n = torch.xpu.device_count()
        names = []
        for i in range(n):
            p = torch.xpu.get_device_properties(i)
            names.append(f"{i}:{p.name} {p.total_memory / 2**30:.1f} GiB")
        report(OK, f"XPU devices ({n})", "; ".join(names))
        if os.environ.get("ZE_AFFINITY_MASK"):
            report(OK, "ZE_AFFINITY_MASK", os.environ["ZE_AFFINITY_MASK"])
        return "xpu"
    if torch.cuda.is_available():
        report(
            OK,
            f"CUDA devices ({torch.cuda.device_count()})",
            torch.cuda.get_device_name(0),
        )
        return "cuda"
    report(FAIL, "accelerator", "neither torch.xpu nor torch.cuda is available")
    return None


def check_graph_api(torch, family: str) -> None:
    """BCG needs a graph class, a shared mempool, and a capture-state query.

    On XPU these landed in torch 2.13.0+xpu. torch 2.12 has no XPUGraph at all,
    and pre-2.13 builds also did not validate capture legality -- an illegal
    operation inside a segment produced silent corruption and a SIGSEGV on the
    first replay instead of an exception at capture time. Anything older than
    2.13 will waste your day.
    """
    mod = torch.xpu if family == "xpu" else torch.cuda
    graph_cls = "XPUGraph" if family == "xpu" else "CUDAGraph"
    for attr in (graph_cls, "graph_pool_handle", "is_current_stream_capturing"):
        if hasattr(mod, attr):
            report(OK, f"torch.{family}.{attr}")
        else:
            report(
                FAIL, f"torch.{family}.{attr}", "missing -- torch is too old for BCG"
            )


def check_sglang() -> None:
    if importlib.util.find_spec("sglang") is None:
        report(
            FAIL, "import sglang", "not installed (pip install -e 'python[diffusion]')"
        )
        return
    import sglang

    report(
        OK, "sglang", f"{sglang.__version__} from {os.path.dirname(sglang.__file__)}"
    )

    # The two source changes this recipe depends on. Checking them here means a
    # stale install (e.g. a non-editable copy of sglang shadowing the checkout)
    # is caught before a 90 second warmup instead of after it.
    from sglang.multimodal_gen.runtime.server_args.server_args import (
        BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS,
    )

    if "wan2.1-t2v-1.3b-diffusers" in BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS:
        report(OK, "BCG allowlist", "Wan2.1-T2V-1.3B present")
    else:
        report(
            FAIL, "BCG allowlist", "Wan missing -> BCG is silently disabled at startup"
        )
    if "WanT2V480PConfig" in BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS:
        report(OK, "BCG pipeline allowlist", "WanT2V480PConfig present")
    else:
        report(FAIL, "BCG pipeline allowlist", "WanT2V480PConfig missing")

    from sglang.kernels.ops.diffusion.triton import scale_shift

    # The removed fast path; if it is back, capture dies with "wait method
    # cannot be used for an event associated with a command graph". Comments are
    # stripped first because the fix leaves the offending expression quoted in a
    # NOTE explaining why it went away.
    if _contains_code(scale_shift.__file__, 'scale_blc.any().to("cpu"'):
        report(
            FAIL,
            "scale_shift host-sync fast path",
            "still present -> DiT capture will fail",
        )
    else:
        report(OK, "scale_shift host-sync fast path", "removed")


def _contains_code(path: str, needle: str) -> bool:
    """True if ``needle`` appears in ``path`` outside of comments.

    String literals are kept, since the needle contains one (``"cpu"``).
    """
    import io
    import tokenize

    with open(path, "rb") as f:
        tokens = tokenize.tokenize(io.BytesIO(f.read()).readline)
        code = "".join(t.string for t in tokens if t.type != tokenize.COMMENT)
    return needle in code


def check_model(model: str) -> None:
    if os.path.isdir(model):
        needed = ("model_index.json", "transformer")
        missing = [n for n in needed if not os.path.exists(os.path.join(model, n))]
        if missing:
            report(WARN, "model dir", f"{model} is missing {missing}")
        else:
            report(OK, "model dir", model)
    else:
        report(
            WARN, "model", f"{model} is not a local dir; it will be downloaded from HF"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default=os.environ.get("MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    )
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]} ({sys.executable})")
    torch = check_torch()
    if torch is not None:
        family = check_device(torch)
        if family is not None:
            check_graph_api(torch, family)
    check_sglang()
    check_model(args.model)

    print()
    print("NOT READY -- fix the FAIL lines above." if _failed else "ready.")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
