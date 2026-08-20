# Wan2.1-T2V with breakable CUDA graph on Intel XPU

Reproduction kit for running the Wan video DiT under sglang-diffusion's
**breakable CUDA graph** (BCG) path on an Intel GPU, plus the one-line kernel fix
that makes capture possible.

**Status:** capture and replay both work on Wan2.1-T2V-1.3B, and the output video
is **byte-identical to the eager control** at the same seed. Performance is
neutral (~2% slower end to end) because this workload is already device-bound —
see [Results](#results). The value here is that the graph path is now *correct*
on a video DiT, and the failure modes are documented.

---

## Requirements

| | |
|---|---|
| GPU | Intel Arc / Arc Pro / Data Center GPU. Verified on **Arc Pro B60** (23.9 GiB). |
| torch | **≥ 2.13.0+xpu** — hard requirement, see below. |
| model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (~2.6 GB bf16 DiT; auto-downloaded). |
| memory | ~12 GiB device memory for 832x480x17f at bf16 eager, **~20 GiB with BCG** (the graph mempool holds every segment's allocations for the process lifetime). |

**Why torch 2.13 specifically.** `torch.xpu.XPUGraph` and
`torch.xpu.graph_pool_handle` do not exist in 2.12 at all. Just as important,
2.13 *validates* capture legality: an illegal operation inside a segment raises a
`RuntimeError` naming the problem. Earlier builds accepted it silently and
produced a corrupt graph that SIGSEGV'd on the first replay with no diagnostics —
which is what made this bug expensive to find. Do not try this on 2.12.

Install per the [XPU platform docs](../../../docs_new/docs/hardware-platforms/xpu.mdx),
substituting the 2.13 wheels:

```bash
pip3 install torch==2.13.0+xpu torchao==0.17.0+xpu torchvision==0.28.0+xpu \
    torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
pip install -e "python[diffusion]"
```

## Quickstart

```bash
cd examples/diffusion/wan_bcg_xpu

# 1. Preflight. Catches every environment problem we hit, including a stale
#    non-editable sglang install shadowing your checkout.
python check_env.py                       # or: --model /path/to/local/model

# 2. Run the eager control, then BCG, and compare.
CARD=0 ./run_experiment.sh both           # CARD sets ZE_AFFINITY_MASK; omit for all GPUs
```

`run_experiment.sh` starts a server per mode, sends three identical requests,
shuts it down, and prints a summary. The last line is the correctness verdict:

```
=== [eager] starting server, log: .../results/serve-eager.log
=== [eager] ready; sending 2 request(s)
    request 1: job fca68a7a-… completed inference_time_s=12.63 peak_memory_mb=11504.0
...
================ SUMMARY ================

--- [eager] ---
  inference_time_s per request:
    req1: 12.63s  peak=11504 MiB
    req2: 10.38s  peak=11504 MiB
  md5 fc6cdcc32e87680a0f7d0cf200776979  req1/fca68a7a-….mp4

--- [bcg] ---
  [Diffusion BCG] captured 61 segment(s)
  inference_time_s per request:
    req1: 12.83s  peak=20300 MiB
    req2: 10.57s  peak=20300 MiB
  md5 fc6cdcc32e87680a0f7d0cf200776979  req1/b44c2d4e-….mp4

PASS: eager and BCG outputs are byte-identical.
```

Note that `POST /v1/videos` is asynchronous — it returns `{"status":"queued"}`
immediately. `generate.sh` polls `GET /v1/videos/{id}` to completion; a bare curl
would report success while the job was still running (or had failed).

To drive it manually instead:

```bash
CARD=0 ./serve.sh bcg          # or: ./serve.sh eager
./generate.sh ./out            # in another shell, once /health responds
```

## The fix

BCG capture of the Wan DiT failed with:

```
RuntimeError: wait method cannot be used for an event associated with a command graph
```

The cause was a fast path in `fuse_scale_shift_kernel`
([`kernels/ops/diffusion/triton/scale_shift.py`](../../../python/sglang/kernels/ops/diffusion/triton/scale_shift.py)):
when `scale` and `shift` were both scalar zeros it skipped the kernel and did a
plain `copy_` — but it *decided* that by reading the values back to the host:

```python
if not (scale_blc.any().to("cpu", non_blocking=True)
        or shift_blc.any().to("cpu", non_blocking=True)):
```

`non_blocking=True` does not help: Python truthiness on those CPU tensors forces
a device-to-host wait, and a host sync is illegal inside graph capture. Wan
reaches this once per transformer block, via
`RMSNormScaleShift.forward_native` → `WanTransformerBlock.self_attn_residual_norm`.
Because `forward_native` is hard-decorated `@torch.compile` on XPU, Inductor
lowered the read into a `_d2h_event_buf1.synchronize()` inside the captured
region.

Removing the fast path costs nothing. With `scale = shift = 0` the kernel already
computes `x * (1.0 + 0) + 0 == x`, and the `copy_` it substituted moves exactly
as many bytes as the kernel writes — so the fast path was paying a full device
sync to save nothing at all. It was a pessimization even outside capture.

A `torch.xpu.is_current_stream_capturing()` guard would be the obvious
alternative and does work correctly on XPU, but it is the wrong tool here:
`forward_native` is `@torch.compile`d, so Dynamo traces the guard during the
first (non-capturing) warmup forward and can constant-fold it into the graph.

The other change in this PR just adds Wan2.1-T2V-1.3B to the BCG allowlists in
`server_args.py`. Without it `_adjust_breakable_cuda_graph_support` silently sets
`enable_breakable_cuda_graph = False` at startup and everything runs eager.

## What actually gets captured

Nothing is annotated or wrapped by hand — three separate framework mechanisms
decide it:

1. **The unit** is one whole DiT transformer forward. `DenoisingStage` routes
   `current_model(**call_kwargs)` through a `DiffusionBreakableCudaGraphRunner`
   instead of calling it directly, with one runner per module (keyed on
   `id(current_model)`, so `transformer` and `transformer_2` get separate graph
   state).
2. **The break points** are the DiT attention modules.
   `layers/attention/layer.py` monkey-patches `forward` on `UlyssesAttention`,
   `UlyssesAttention_VSA`, `LocalAttention` and `USPAttention` at import time, so
   break points are a property of those classes rather than of any model. Wan
   routes through `USPAttention`.
3. **The split is recorded, not planned.** `eager_on_graph` checks a ContextVar:
   during capture it ends the open segment, runs attention eagerly, stores the
   eager callable, and opens a new segment. Segment boundaries are simply
   wherever a wrapped `forward` happened to be called.

So for Wan2.1-1.3B's 30 blocks (self-attn + cross-attn each) you get
`30 * 2 + 1 = ` **61 graph segments and 60 eager break points**, and nobody
chose 61. Inside the graphs: patch embed, RoPE and modulation, norms, QKV and
output projections, FFN, final layer. Eager on every replay: the attention calls
themselves — which is deliberate, since sequence-parallel all-to-alls, varlen
packing and dynamic/sparse attention kernels either cannot or should not be
captured.

Capture happens **only during warmup** (`serve --warmup-mode server`), and a
serving request replays only on an exact input-signature match.

## Results

Wan2.1-T2V-1.3B, 832x480x17f, 8 steps, Arc Pro B60, torch 2.13.0+xpu.

| | eager | BCG |
|---|---|---|
| output md5 | `fc6cdcc3…` | `fc6cdcc3…` (identical) |
| `inference_time_s`, 2 requests | 12.63 / 10.38 s | 12.83 / 10.57 s (+1.6% / +1.9%) |
| peak device memory | 11504 MiB | **20300 MiB** (+8.8 GiB) |
| segments captured | — | 61/61 |
| replays per request | — | 16/16 |
| device utilisation | **99.4%** | not comparable (see below) |
| `cpu_op` events | 53107 | 6595 |
| `xpu_runtime` events (time) | 6937 (240 ms) | 1415 (36 ms) |

BCG delivers exactly the host-side saving it promises — an 8x cut in launch
overhead — and it does not matter, because **eager already keeps the device 99.4%
busy**. There is no host-side bubble to remove. Graphs pay off when the host
cannot feed the device (many small kernels, LLM decode); Wan's forward is ~3 ms
attentions and large GEMMs.

The memory cost is the more consequential result: peak device memory goes from
11.5 GiB to 19.8 GiB, a **1.8x increase**, because the graph mempool has to hold
every segment's intermediate allocations plus the static input buffers for the
whole captured forward, all live simultaneously for the process lifetime. On a
1.3B DiT that is affordable. It is another reason the larger Wan variants are not
reachable by this path without a residency story.

Per-forward replay is 621 ms of device time (427 ms in graph segments, 195 ms in
eager attention breaks). A bare `XPUGraph.replay()` costs 0.010 ms, so there is
no per-segment dispatch penalty — 61 segments is not the problem.

Two measurable overheads BCG adds:

- **480 extra D2D memcpys (45.9 ms/request)** — the bridge copies that feed each
  eager break's output into the next segment's static input buffers.
- **~32 ms per forward (~515 ms per request) of pure Python** recomputing the
  runner's cache key. `_signature_leaf` / `_flatten_tensors` are each called
  75,061 times *per forward* because the signature walks the `mask_strategy`
  kwarg, a 50x60x24 nested list. That is essentially the entire ~2% regression,
  and it is fixable by memoizing the signature for structural non-tensor kwargs.
  Not addressed here.

Two caveats when reading your own profiles:

- **Graph-internal kernels are invisible to kineto.** The BCG trace shows 27
  distinct kernel names / 1149 launches against eager's 49 / 8800; the GEMMs,
  rotary, fused-norm and scale-shift kernels are simply absent, and only
  `torch/xpu/graphs.py: replay` appears, as a Python call. Never compare
  device-busy totals across the two — BCG's apparent "31% utilisation" is an
  artifact of missing events, not idleness.
- **Eager step timings do not device-sync.** `[DenoisingStage] average time per
  step` measures host submit time in eager mode but real device time under graph
  replay, which makes eager look ~10x faster than it is. Compare end-to-end
  numbers, not per-step numbers.

## Troubleshooting

| Symptom in the server log | Cause | Fix |
|---|---|---|
| `[Diffusion BCG] disabled for WanT2V480PConfig` | model or pipeline config not in the allowlist | this PR adds both; check `python check_env.py` |
| no `[Diffusion BCG] captured …` line at all | warmup never ran | BCG only captures during warmup, and only `sglang serve` warms up. `sglang generate` has no warmup path and, with BCG on, blocks in `broadcast_pyobj` until the 1800 s gloo timeout |
| `wait method cannot be used for an event associated with a command graph` | a host sync inside the captured region | the `scale_shift` fix in this PR; if it is a different site, use `patch_bcg_trace.py` to get the traceback |
| `Event dependency from handler::depends_on does not correspond to a node within the graph` | weight streaming inside capture | pass `--enable-torch-compile false` **explicitly**. `--performance-mode speed` enables torch.compile, and `_maybe_offload_during_compile` then installs a `LayerwiseOffloadManager` even with `--dit-layerwise-offload false` |
| capture succeeds, output is fine, but no speedup and no replay logs | request signature ≠ warmup signature | `size` AND `num_frames` must match warmup (832x480x**17f**). 33 frames gives latents `(1,16,9,60,104)` vs the captured `(1,16,5,60,104)`. Outside warmup the runner never captures a new signature; it logs `signature MISS, serving -> eager` (visible only with `BCG_TRACE=1`) |
| bare exit code 139, no traceback | torch < 2.13 accepting an illegal capture | upgrade; also `PYTHONFAULTHANDLER=1`, which `serve.sh` sets |
| hang in `ur::level_zero::urEventWait` on a multi-GPU box | `--base-gpu-id` does not isolate devices; ranks open every `renderD*` node | set `CARD=<n>` so `serve.sh` exports `ZE_AFFINITY_MASK` |
| out of memory during warmup, but eager fits | BCG's mempool roughly doubles peak device memory (11.5 → 19.8 GiB here) | lower the resolution/frame count, or run eager — this is inherent to capturing a whole DiT forward |

## Optional instrumentation

Both scripts edit the installed sources in place (leaving a `.orig.bak`), are
idempotent, and are inert unless their env var is set. Revert with `git checkout`.

**Capture/replay tracing** — names every break point by module path, and prints
the capture traceback that `try_capture()` otherwise swallows. Indispensable when
bringing up a new model; this is how the `scale_shift` site was found.

```bash
python patch_bcg_trace.py
BCG_TRACE=1 BCG_TRACE_SYNC=1 ./serve.sh bcg
```

`BCG_TRACE_SYNC=1` syncs after every segment and every break, so the last line
before a crash names the exact culprit. `BCG_TRACE_TIME=1` swaps the ~120
lines/replay firehose for a single summary line splitting graph-segment time from
eager-break time — usable during a timing run.

**Profiling** — `/v1/videos` has no `profile` field (it exists only on the
realtime WebSocket request model), so the profiler is forced on by env var:

```bash
python patch_profile_force.py
BCG_PROFILE=1 BCG_PROFILE_TAG=bcg ./serve.sh bcg      # then send one request
python analyze_traces.py traces/eager-*.json.gz traces/bcg-*.json.gz
```

## Extending this

- **Other Wan sizes.** Add the model id to
  `BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS` and the pipeline config class name
  to `BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS` in `server_args.py`. Note
  the config check uses the exact class name, so `WanT2V720PConfig` needs its own
  entry even though it subclasses `WanT2V480PConfig`.
- **Wan2.2-A14B is blocked on memory, not on graphs.** One transformer is ~28 GB
  bf16 against 23.9 GiB per card; the layerwise offload that makes it fit is
  precisely what capture cannot cover; and BCG itself needs ~1.8x the peak memory
  of eager. It needs a residency solution before the graph path is even testable.
- **Bringing up an unlisted model.** Break points are chosen by attention-class
  membership, and nothing verifies that the enclosed code is capture-legal — that
  is exactly how the `scale_shift` host sync slipped through, since it lives in a
  normalization layer no break point covers. Expect to run
  `patch_bcg_trace.py`, read the capture traceback, and hunt for `.item()`,
  `.cpu()`, `bool(tensor)` and `.synchronize()` on the path.
