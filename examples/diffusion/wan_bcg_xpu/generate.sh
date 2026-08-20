#!/usr/bin/env bash
# Send one text-to-video request to a server started by serve.sh and wait for it
# to finish.
#
#   ./generate.sh [output_dir]
#
# POST /v1/videos is asynchronous: it returns 200 with {"status":"queued"}
# immediately, so a bare curl tells you nothing about whether generation worked.
# This polls GET /v1/videos/{id} until the job reaches completed or failed, then
# prints the server-measured inference time and writes the final job JSON to
# <output_dir>/job.json.
#
# size AND num_frames must match what warmup captured (832x480x17f), otherwise
# the captured graph is never replayed and the request silently runs eager.
# num_frames feeds the latent frame count, which is part of the BCG signature:
# 33 frames yields hidden_states (1,16,9,60,104) instead of the captured
# (1,16,5,60,104). Outside warmup the runner never captures a new signature, it
# just logs "signature MISS, serving -> eager" (only visible with BCG_TRACE=1).
set -euo pipefail

OUT=${1:-$PWD/out}
PORT=${PORT:-30000}
POLL_TIMEOUT=${POLL_TIMEOUT:-3600}
PY=${PY:-python3}

mkdir -p "$OUT"

# Read one top-level key out of a JSON object on stdin; empty if absent or null.
jget() {
  "$PY" -c 'import json,sys
try:
    v = json.load(sys.stdin).get(sys.argv[1])
except Exception:
    v = None
print("" if v is None else v)' "$1"
}

resp=$(curl -sS --fail --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/videos" \
  -F 'prompt=A red fox walking through a snowy forest at sunrise, cinematic' \
  -F 'size=832x480' \
  -F "num_frames=${NUM_FRAMES:-17}" \
  -F "num_inference_steps=${STEPS:-8}" \
  -F "seed=${SEED:-42}" \
  -F "output_path=$OUT")

job_id=$(printf '%s' "$resp" | jget id)
if [ -z "$job_id" ]; then
  echo "submit failed: $resp" >&2
  exit 1
fi

deadline=$((SECONDS + POLL_TIMEOUT))
while :; do
  job=$(curl -sS --fail --max-time 30 "http://127.0.0.1:$PORT/v1/videos/$job_id") || {
    echo "poll failed for job $job_id (did the server die?)" >&2
    exit 1
  }
  status=$(printf '%s' "$job" | jget status)
  case "$status" in
    completed) break ;;
    failed)
      printf '%s' "$job" >"$OUT/job.json"
      echo "job $job_id FAILED: $(printf '%s' "$job" | jget error)" >&2
      exit 1
      ;;
  esac
  if ((SECONDS >= deadline)); then
    echo "job $job_id still '$status' after ${POLL_TIMEOUT}s, giving up" >&2
    exit 1
  fi
  sleep 2
done

printf '%s' "$job" >"$OUT/job.json"
echo "job $job_id completed" \
     "inference_time_s=$(printf '%s' "$job" | jget inference_time_s)" \
     "peak_memory_mb=$(printf '%s' "$job" | jget peak_memory_mb)"
echo "  -> $(printf '%s' "$job" | jget file_path)"
