#!/bin/bash
# tune-moe-offload.sh
# Usage: tune-moe-offload.sh <name> <repo> <file>
#
# Sweeps -ncmoe (llama.cpp's MoE-CPU-offload flag) from 999 (full offload, the naive
# default) down through progressively smaller values, re-benchmarking at each step.
# Every -ncmoe layer forced onto the CPU is one fewer expert running at GPU speed --
# 999 throws away whatever VRAM headroom the card actually has. Stops at the first
# value that fails to load OR times out (a hang is a failure too, same as a clean OOM --
# both mean "don't deploy at this value").
set -uo pipefail

NAME="$1"; REPO="$2"; FILE="$3"

MODEL_DIR="/opt/models"
MODEL_PATH="${MODEL_DIR}/${NAME}.gguf"
RESULTS="/opt/llm-bench/moe-tune-results.jsonl"
LLAMA_BENCH="/opt/llama.cpp-vulkan/build/bin/llama-bench"
TIMEOUT_SECS=90

mkdir -p "${MODEL_DIR}" "$(dirname "${RESULTS}")"

log_result() {
  local ncmoe="$1" status="$2" pp="$3" tg="$4" error="$5"
  python3 - "$NAME" "$ncmoe" "$status" "$pp" "$tg" "$error" "$RESULTS" <<'PY'
import json, sys, datetime
name, ncmoe, status, pp, tg, error, results = sys.argv[1:]
row = {
    "model": name, "ncmoe": int(ncmoe), "status": status,
    "pp_tok_s": float(pp) if pp else None, "tg_tok_s": float(tg) if tg else None,
    "error": error or None,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(results, "a") as f:
    f.write(json.dumps(row) + "\n")
PY
}

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "=== ${NAME}: downloading ==="
  if ! curl -sS -L --fail -C - -o "${MODEL_PATH}" "https://huggingface.co/${REPO}/resolve/main/${FILE}"; then
    log_result "999" "failed" "" "" "download failed for ${REPO}/${FILE}"
    rm -f "${MODEL_PATH}"
    exit 0
  fi
fi

for N in 999 64 48 40 32 24 16 12 8 4 0; do
  echo "=== ${NAME} -ncmoe ${N} ==="
  OUT=$(timeout "${TIMEOUT_SECS}" "${LLAMA_BENCH}" -m "${MODEL_PATH}" -p 512 -n 128 -r 3 -o json \
    --device Vulkan0 -ngl 999 -ncmoe "${N}" 2>&1)
  STATUS=$?

  if [[ ${STATUS} -eq 124 ]]; then
    echo "  TIMED OUT (${TIMEOUT_SECS}s) -- stopping sweep, lower N would be worse"
    log_result "${N}" "timeout" "" "" "${TIMEOUT_SECS}s timeout, killed"
    break
  elif [[ ${STATUS} -ne 0 ]] || ! echo "${OUT}" | grep -q '"avg_ts"'; then
    echo "  FAILED: $(echo "${OUT}" | tail -c 300)"
    log_result "${N}" "failed" "" "" "$(echo "${OUT}" | tail -c 300 | tr '\n"' '  ')"
    break
  else
    JSON_ONLY=$(echo "${OUT}" | sed -n '/^\[/,/^\]/p')
    PP=$(echo "${JSON_ONLY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['avg_ts'])" 2>/dev/null)
    TG=$(echo "${JSON_ONLY}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[1]['avg_ts'])" 2>/dev/null)
    echo "  OK: pp=${PP} tg=${TG} tok/s"
    log_result "${N}" "ok" "${PP}" "${TG}" ""
  fi
done

echo "=== ${NAME}: cleanup ==="
rm -f "${MODEL_PATH}"
