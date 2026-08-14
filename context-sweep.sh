#!/bin/bash
# services/llm-bench/context-sweep.sh
# Usage: context-sweep.sh <name> <repo> <file> <device>
# device: a770 | igpu | cpu_offload
#
# Finds the real max usable context window for a model on this hardware --
# motivated by the 2026-08-13 finding that a model's throughput-benchmark context
# (512 tokens) tells you nothing about whether it can serve a realistic context
# size. Sweeps ascending checkpoints for both standard (fp16) and quantized (q8_0)
# KV cache, stopping at the first checkpoint that fails or hangs. A "hang" (no
# response within the per-attempt timeout) is treated as a failure at that
# checkpoint, same as a clean OOM -- both mean "don't deploy at this size."
set -uo pipefail

NAME="$1"; REPO="$2"; FILE="$3"; DEVICE="$4"

MODEL_DIR="/opt/models"
MODEL_PATH="${MODEL_DIR}/${NAME}.gguf"
RESULTS="/opt/llm-bench/context-results.jsonl"
LLAMA_SERVER="/opt/llama.cpp-vulkan/build/bin/llama-server"
PORT=8092
CHECKPOINTS=(8192 16384 24576 32768 49152 65536 98304 131072)
# Capped at 128K (2026-08-14) -- past this point we're well beyond any of this
# session's practical deployment targets (Hermes needs 64K; most real agentic tool
# budgets fit in 100-128K), and models with lots of headroom were running the full
# ladder to 256K for no additional decision-relevant data.

mkdir -p "${MODEL_DIR}" "$(dirname "${RESULTS}")"

log_result() {
  local kv_mode="$1" max_ctx="$2" note="$3"
  python3 - "$NAME" "$DEVICE" "$kv_mode" "$max_ctx" "$note" "$RESULTS" <<'PY'
import json, sys, datetime
name, device, kv_mode, max_ctx, note, results = sys.argv[1:]
row = {
    "name": name, "device": device, "kv_mode": kv_mode,
    "max_context": int(max_ctx) if max_ctx else None,
    "note": note or None,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
with open(results, "a") as f:
    f.write(json.dumps(row) + "\n")
PY
}

case "${DEVICE}" in
  a770)        DEVICE_FLAGS="--device Vulkan0 -ngl 999" ;;
  igpu)        DEVICE_FLAGS="--device Vulkan1 -ngl 999" ;;
  cpu_offload) DEVICE_FLAGS="--device Vulkan0 -ngl 999 -ncmoe 999" ;;
  *) echo "unknown device '${DEVICE}'"; exit 1 ;;
esac

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "=== ${NAME}: downloading ==="
  if ! curl -sS -L --fail -C - -o "${MODEL_PATH}" "https://huggingface.co/${REPO}/resolve/main/${FILE}"; then
    log_result "standard" "" "download failed"
    rm -f "${MODEL_PATH}"
    exit 0
  fi
fi

try_context() {
  local ctx="$1" kv_flags="$2"
  pkill -9 -f llama-server 2>/dev/null
  sleep 2
  local logf="/tmp/ctxsweep-${NAME}-${ctx}.log"
  timeout 40 "${LLAMA_SERVER}" -m "${MODEL_PATH}" ${DEVICE_FLAGS} -c "${ctx}" ${kv_flags} \
    --port "${PORT}" --host 127.0.0.1 > "${logf}" 2>&1 &
  local pid=$!
  local waited=0
  while [[ ${waited} -lt 35 ]]; do
    if grep -q "listening on" "${logf}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null
      pkill -9 -f llama-server 2>/dev/null
      rm -f "${logf}"
      return 0
    fi
    if grep -qiE "out.*device.*memory|exiting due to|allocateMemory.*Error" "${logf}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null
      pkill -9 -f llama-server 2>/dev/null
      rm -f "${logf}"
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  # Timed out with no clear success/failure signal -- treat as a hang/failure.
  kill -9 "${pid}" 2>/dev/null
  pkill -9 -f llama-server 2>/dev/null
  rm -f "${logf}"
  return 1
}

sweep_kv_mode() {
  local kv_label="$1" kv_flags="$2"
  local max_working=0
  for ctx in "${CHECKPOINTS[@]}"; do
    echo "=== ${NAME} [${kv_label}]: trying ctx=${ctx} ==="
    if try_context "${ctx}" "${kv_flags}"; then
      max_working="${ctx}"
    else
      echo "  failed/hung at ${ctx}, stopping this KV mode"
      break
    fi
  done
  if [[ "${max_working}" -eq 0 ]]; then
    log_result "${kv_label}" "" "failed even at the smallest checkpoint (${CHECKPOINTS[0]})"
  else
    log_result "${kv_label}" "${max_working}" ""
  fi
}

sweep_kv_mode "standard" ""
sweep_kv_mode "q8_0" "-ctk q8_0 -ctv q8_0"

echo "=== ${NAME}: cleanup ==="
rm -f "${MODEL_PATH}"
