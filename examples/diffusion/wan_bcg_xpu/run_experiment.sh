#!/usr/bin/env bash
# One-command reproduction: start the server, send a few identical requests,
# shut it down, and report what BCG actually did.
#
#   ./run_experiment.sh both     # eager control, then BCG, then compare (default)
#   ./run_experiment.sh bcg
#   ./run_experiment.sh eager
#
# Environment:
#   MODEL          model path or HF id (default Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
#   CARD           ZE_AFFINITY_MASK value; unset means "all visible GPUs"
#   OUTDIR         results directory (default ./results)
#   REQUESTS       requests per mode (default 3)
#   READY_TIMEOUT  seconds to wait for /health (default 1200; model download +
#                  warmup on a cold cache is slow)
#   PORT           default 30000
#
# With `both`, the two modes run sequentially against the same seed, so the
# output videos must be byte-identical. That equality is the correctness test:
# graph replay is only trustworthy if it reproduces eager bit for bit.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-both}
OUTDIR=${OUTDIR:-$PWD/results}
REQUESTS=${REQUESTS:-3}
READY_TIMEOUT=${READY_TIMEOUT:-1200}
PORT=${PORT:-30000}
PY=${PY:-python3}
# serve.sh keeps its scratch dirs (outputs/, inputs/, traces/) under WORKDIR
# rather than the cwd, so a run inside the repo leaves no stray files.
export PORT PY
export WORKDIR=${WORKDIR:-$OUTDIR}

strip_ansi() { sed -e 's/\x1b\[[0-9;]*m//g'; }

# One line of "<inference time>  peak=<memory>" from a completed job.json.
job_line() {
  "$PY" - "$1" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("inference_time_s")
m = d.get("peak_memory_mb")
print(f"{t:.2f}s" if t else f"status={d.get('status')}, no timing",
      f" peak={m:.0f} MiB" if m else "")
EOF
}

wait_for_health() {
  local pid=$1 deadline=$((SECONDS + READY_TIMEOUT))
  while ((SECONDS < deadline)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "!! server exited before becoming ready" >&2
      return 1
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
  done
  echo "!! server not ready after ${READY_TIMEOUT}s" >&2
  return 1
}

stop_server() {
  local pid=$1
  # serve.sh forks a scheduler process per GPU plus an HTTP server, so signal
  # the whole process group (setsid gave it its own) rather than just the shell.
  kill -INT -- "-$pid" 2>/dev/null
  for _ in $(seq 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null
  # `wait` reaps quietly; without it bash prints its own "Killed" job notice.
  wait "$pid" 2>/dev/null
  # Give the driver a moment to release the device before the next mode starts.
  sleep 5
}

run_mode() {
  local mode=$1
  local log="$OUTDIR/serve-$mode.log"
  local dir="$OUTDIR/$mode"
  mkdir -p "$dir"

  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "!! something is already listening on port $PORT" >&2
    return 1
  fi

  echo "=== [$mode] starting server, log: $log"
  setsid "$HERE/serve.sh" "$mode" >"$log" 2>&1 &
  local pid=$!

  if ! wait_for_health "$pid"; then
    echo "--- last 40 lines of $log ---" >&2
    tail -40 "$log" | strip_ansi >&2
    stop_server "$pid"
    return 1
  fi
  echo "=== [$mode] ready; sending $REQUESTS request(s)"

  local i
  for i in $(seq "$REQUESTS"); do
    echo -n "    request $i: "
    "$HERE/generate.sh" "$dir/req$i" || echo "    request $i FAILED" >&2
  done

  stop_server "$pid"
  echo "=== [$mode] stopped"
}

summarize() {
  local mode=$1
  local log="$OUTDIR/serve-$mode.log"
  [ -f "$log" ] || return 0
  echo
  echo "--- [$mode] ---"

  # Capture happens once, during warmup. No line here means BCG never engaged:
  # look for "[Diffusion BCG] disabled for ..." (allowlist) or a capture
  # traceback (an illegal op inside a segment) in the log.
  local cap
  cap=$(grep -o "\[Diffusion BCG\] captured [0-9]* segment(s)" "$log" | head -1)
  if [ -n "$cap" ]; then
    echo "  $cap"
  elif [ "$mode" = bcg ]; then
    echo "  !! no capture line found -- BCG did not engage"
    grep -m3 "Diffusion BCG" "$log" | strip_ansi | sed 's/^/     /'
  fi

  # Server-measured inference time, straight from the completed job. This is the
  # number to compare between modes.
  echo "  inference_time_s per request:"
  local j
  while IFS= read -r j; do
    echo "    $(basename "$(dirname "$j")"): $(job_line "$j")"
  done < <(find "$OUTDIR/$mode" -name job.json -type f 2>/dev/null | sort)

  # Per-step denoising times, with the caveat that makes them dangerous: eager
  # submission never device-syncs, so this line measures host submit time in
  # eager mode but real device time under graph replay. It will claim eager is
  # ~10x faster. Believe inference_time_s, not this.
  echo "  [DenoisingStage] per-step (NOT comparable across modes, see README):"
  grep -o "\[DenoisingStage\] average time per step: .*" "$log" | strip_ansi | sed 's/^/    /'

  # md5 of the produced video(s). Same seed + same request => identical bytes.
  local f
  while IFS= read -r f; do
    echo "  md5 $(md5sum "$f" | cut -d' ' -f1)  $(basename "$(dirname "$f")")/$(basename "$f")"
  done < <(find "$OUTDIR/$mode" -name '*.mp4' -type f 2>/dev/null | sort)
}

mkdir -p "$OUTDIR"

case "$MODE" in
  both)
    run_mode eager || exit 1
    run_mode bcg || exit 1
    ;;
  eager|bcg)
    run_mode "$MODE" || exit 1
    ;;
  *)
    echo "usage: $0 [both|bcg|eager]" >&2
    exit 2
    ;;
esac

echo
echo "================ SUMMARY ================"
summarize eager
summarize bcg

if [ "$MODE" = both ]; then
  a=$(find "$OUTDIR/eager/req1" -name '*.mp4' -type f 2>/dev/null | head -1)
  b=$(find "$OUTDIR/bcg/req1" -name '*.mp4' -type f 2>/dev/null | head -1)
  echo
  if [ -n "$a" ] && [ -n "$b" ]; then
    if [ "$(md5sum "$a" | cut -d' ' -f1)" = "$(md5sum "$b" | cut -d' ' -f1)" ]; then
      echo "PASS: eager and BCG outputs are byte-identical."
    else
      echo "FAIL: outputs differ. Graph replay is not reproducing eager."
      echo "      (First check the bcg log for 'signature MISS' -- a mismatched"
      echo "      resolution or num_frames means the request never replayed.)"
    fi
  else
    echo "?? could not find both videos to compare ($a | $b)"
  fi
fi
