#!/bin/bash
# services/llm-bench/bench-model.sh
# Usage: bench-model.sh <name> <repo> <file> <params_total_b> <params_active_b> <quant> <device>
# device: a770 | igpu | cpu_offload
#
# Backend: llama.cpp + Vulkan (/opt/llama.cpp-vulkan/build/bin/llama-bench) — the winner
# of the Vulkan/SYCL/OpenArc shootout. See services/llm-bench/shootout-results.md for why:
# SYCL and OpenArc are both theoretically more Intel-optimized but crash on the actual 2026
# model architectures in this matrix (Gemma 4, Qwen 3.6); Vulkan is the one that works.
#
# Device targeting (confirmed live via `llama-bench --list-devices` on this box):
#   Vulkan0 = Intel Arc A770 (16GB dedicated VRAM)
#   Vulkan1 = Intel iGPU / Arrow Lake (UMA, shares system RAM)
set -uo pipefail  # not -e: a failed model must still produce a "failed" result row, not abort the matrix

NAME="$1"; REPO="$2"; FILE="$3"
PARAMS_TOTAL="$4"; PARAMS_ACTIVE="$5"; QUANT="$6"; DEVICE="$7"

MODEL_DIR="/opt/models"
MODEL_PATH="${MODEL_DIR}/${NAME}.gguf"
RESULTS="/opt/llm-bench/results.jsonl"
LLAMA_BENCH="/opt/llama.cpp-vulkan/build/bin/llama-bench"

mkdir -p "${MODEL_DIR}" "$(dirname "${RESULTS}")"

log_result() {
  local status="$1" pp="$2" tg="$3" vram="$4" error="$5"
  python3 - "$NAME" "$PARAMS_TOTAL" "$PARAMS_ACTIVE" "$QUANT" "$DEVICE" "$status" "$pp" "$tg" "$vram" "$error" "$RESULTS" <<'PY'
import json, sys, datetime
name, ptot, pact, quant, device, status, pp, tg, vram, error, results = sys.argv[1:]
row = {
    "name": name, "params_total_b": float(ptot), "params_active_b": float(pact),
    "quant": quant, "device": device, "status": status,
    "pp_tok_s": float(pp) if pp else None, "tg_tok_s": float(tg) if tg else None,
    "vram_mb": float(vram) if vram else None, "error": error or None,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(results, "a") as f:
    f.write(json.dumps(row) + "\n")
PY
}

echo "=== ${NAME} (${DEVICE}): downloading ==="
if ! curl -sS -L --fail -C - -o "${MODEL_PATH}" "https://huggingface.co/${REPO}/resolve/main/${FILE}"; then
  log_result "failed" "" "" "" "download failed for ${REPO}/${FILE}"
  rm -f "${MODEL_PATH}"
  exit 0
fi

# Verify the download actually completed — a truncated file loads "successfully" as
# garbage in some tools but llama-bench correctly refuses it; still worth an explicit
# size check so a truncation shows up as a download failure, not a confusing bench failure.
EXPECTED_SIZE=$(curl -sI -L "https://huggingface.co/${REPO}/resolve/main/${FILE}" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
ACTUAL_SIZE=$(stat -c%s "${MODEL_PATH}" 2>/dev/null || stat -f%z "${MODEL_PATH}")
if [[ -n "${EXPECTED_SIZE}" && "${EXPECTED_SIZE}" != "${ACTUAL_SIZE}" ]]; then
  log_result "failed" "" "" "" "size mismatch: expected ${EXPECTED_SIZE} bytes, got ${ACTUAL_SIZE}"
  rm -f "${MODEL_PATH}"
  exit 0
fi

echo "=== ${NAME} (${DEVICE}): benchmarking ==="
case "${DEVICE}" in
  a770)        DEVICE_FLAGS="--device Vulkan0 -ngl 999" ;;
  igpu)        DEVICE_FLAGS="--device Vulkan1 -ngl 999" ;;
  cpu_offload) DEVICE_FLAGS="--device Vulkan0 -ngl 999 -ncmoe 999" ;; # all non-MoE layers on A770, all MoE experts forced to system RAM
  *)
    log_result "failed" "" "" "" "unknown device '${DEVICE}'"
    rm -f "${MODEL_PATH}"
    exit 0
    ;;
esac

BENCH_OUT=$("${LLAMA_BENCH}" -m "${MODEL_PATH}" -p 512 -n 128 -r 3 -o json ${DEVICE_FLAGS} 2>&1)
BENCH_STATUS=$?

if [[ ${BENCH_STATUS} -ne 0 ]]; then
  log_result "failed" "" "" "" "llama-bench exited ${BENCH_STATUS}: $(echo "$BENCH_OUT" | tail -c 500)"
else
  # Schema confirmed live during the shootout: BENCH_OUT is a JSON array of two objects —
  # [0] = prompt-processing test (n_prompt=512, n_gen=0), [1] = generation test
  # (n_prompt=0, n_gen=128). Both report throughput in "avg_ts".
  JSON_ONLY=$(echo "$BENCH_OUT" | sed -n '/^\[/,/^\]/p')
  PP=$(echo "$JSON_ONLY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['avg_ts'])" 2>/dev/null)
  TG=$(echo "$JSON_ONLY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[1]['avg_ts'])" 2>/dev/null)
  if [[ -z "${PP}" || -z "${TG}" ]]; then
    log_result "failed" "" "" "" "llama-bench exited 0 but JSON parse failed: $(echo "$BENCH_OUT" | tail -c 500)"
  else
    log_result "ok" "${PP}" "${TG}" "" ""
  fi
fi

echo "=== ${NAME}: cleanup ==="
rm -f "${MODEL_PATH}"
