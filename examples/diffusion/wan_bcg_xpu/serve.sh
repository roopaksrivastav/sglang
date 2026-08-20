#!/usr/bin/env bash
# Launch sglang-diffusion serving Wan2.1-T2V-1.3B, with breakable CUDA graph
# (BCG) either on or off.
#
#   ./serve.sh bcg      # BCG enabled  (captures the DiT during warmup)
#   ./serve.sh eager    # control run, BCG disabled
#
# Overridable via environment:
#   MODEL       model path or HF id      (default Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
#   PORT        HTTP port                (default 30000)
#   CARD        value for ZE_AFFINITY_MASK; unset means "use every visible GPU"
#   SGLANG_BIN  sglang entry point       (default: `sglang` from PATH)
#   BCG_TRACE   1 to log every capture/replay step (needs patch_bcg_trace.py)
#
# Two flags below are load-bearing and easy to get wrong -- see the comments.
set -euo pipefail

MODE=${1:-bcg}
MODEL=${MODEL:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}
PORT=${PORT:-30000}
SGLANG_BIN=${SGLANG_BIN:-sglang}

# On multi-GPU Intel boxes --base-gpu-id does not really isolate a device: every
# rank still opens all /dev/dri/renderD* nodes and can wedge on a card another
# job is using. Pin at the driver level instead. The mask renumbers the chosen
# card to 0, which is why --base-gpu-id below stays 0.
if [ -n "${CARD:-}" ]; then
  export ZE_AFFINITY_MASK="$CARD"
fi

# Turn a SIGSEGV into a Python traceback instead of a bare exit code 139. Graph
# replay bugs on older torch builds surface exactly this way.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# Scratch space. The server's defaults for these are relative paths ("outputs/",
# "inputs/uploads"), which would litter whatever directory you launched from --
# i.e. this one, inside the repo.
WORKDIR=${WORKDIR:-$PWD/results}
mkdir -p "$WORKDIR"

# Where SGLDiffusionProfiler writes chrome traces (see patch_profile_force.py).
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR=${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-$WORKDIR/traces}

COMMON=(
  --model-path "$MODEL"
  --num-gpus 1 --base-gpu-id 0
  --ulysses-degree 1 --ring-degree 1
  --dit-precision bf16 --vae-precision bf16 --text-encoder-precisions bf16
  --output-path "$WORKDIR/outputs"
  --input-save-path "$WORKDIR/inputs"

  # The DiT must stay fully resident. Layerwise offload streams weights on a
  # separate copy stream that graph capture cannot legally cover, and FSDP
  # inserts collectives mid-forward. 1.3B bf16 is ~2.6 GB, so this is free here;
  # it is also the reason this recipe does not extend to Wan2.2-A14B as-is.
  --dit-cpu-offload false
  --dit-layerwise-offload false
  --use-fsdp-inference false

  # MUST be explicit, not just omitted. --performance-mode speed turns
  # torch.compile on by default, and DenoisingStage._maybe_offload_during_compile
  # then installs a LayerwiseOffloadManager on the DiT purely so the compile
  # autotune can run one layer at a time -- silently reinstating the per-layer
  # H2D weight streaming that makes capture illegal, even with
  # --dit-layerwise-offload false. An explicit false wins over the performance
  # mode default. Symptom if you forget: capture fails with "Event dependency
  # from handler::depends_on does not correspond to a node within the graph".
  --performance-mode speed
  --enable-torch-compile false

  # The umt5-xxl text encoder runs once per request, outside the captured
  # region, so parking it on the CPU between requests costs nothing.
  --text-encoder-cpu-offload true
  --vae-cpu-offload false

  # `serve` is the only entry point that drives synthetic warmup, and warmup is
  # the only thing that triggers BCG capture. (`sglang generate` has no warmup
  # path and, with BCG on, just blocks in broadcast_pyobj until the gloo
  # timeout.) A request is replayed only if its input signature matches what
  # warmup captured, so resolution AND frame count must agree -- 832x480x17f.
  --warmup-mode server
  --warmup-resolutions 832x480
  --warmup-steps 1

  --port "$PORT"
)

case "$MODE" in
  eager)
    echo "### MODE: eager control (BCG off), ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"
    exec "$SGLANG_BIN" serve "${COMMON[@]}" --enable-breakable-cuda-graph false
    ;;
  bcg)
    echo "### MODE: breakable CUDA graph, ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<all>}"
    # Only meaningful if patch_bcg_trace.py has been applied; harmless otherwise.
    # The full trace prints ~120 lines per replay, so keep it off when timing.
    export BCG_TRACE=${BCG_TRACE:-0}
    export BCG_TRACE_SYNC=${BCG_TRACE_SYNC:-0}
    exec "$SGLANG_BIN" serve "${COMMON[@]}" \
      --enable-breakable-cuda-graph true \
      --bcg-text-buckets 512
    ;;
  *)
    echo "usage: $0 [bcg|eager]" >&2
    exit 2
    ;;
esac
