#!/bin/bash
# services/llm-bench/run-matrix.sh
set -uo pipefail

MODELS_YAML="/opt/llm-bench/models.yaml"
RESULTS="/opt/llm-bench/results.jsonl"
PROGRESS="/opt/llm-bench/progress.json"

touch "${RESULTS}"

# Build the set of already-completed (name, device) pairs from results.jsonl so a
# re-invocation skips anything already benched — including failures, which count as
# "done" (re-run manually if you want to retry a failure).
python3 - "$MODELS_YAML" "$RESULTS" <<'PY' > /tmp/matrix_todo.tsv
import yaml, json, sys

models_yaml, results_path = sys.argv[1:]
done = set()
try:
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["name"], row["device"]))
except FileNotFoundError:
    pass

with open(models_yaml) as f:
    models = yaml.safe_load(f)["models"]

for m in models:
    for device in m["devices"]:
        if (m["name"], device) in done:
            continue
        print(f"{m['name']}\t{m['repo']}\t{m['file']}\t{m['params_total_b']}\t{m['params_active_b']}\t{m['quant']}\t{device}")
PY

TOTAL=$(wc -l < /tmp/matrix_todo.tsv)
echo "=== ${TOTAL} (model, device) pairs remaining ==="

I=0
while IFS=$'\t' read -r NAME REPO FILE PTOT PACT QUANT DEVICE; do
  I=$((I+1))
  echo "=== [$I/$TOTAL] ${NAME} on ${DEVICE} ==="
  /opt/llm-bench/bench-model.sh "$NAME" "$REPO" "$FILE" "$PTOT" "$PACT" "$QUANT" "$DEVICE"
  python3 -c "import json; json.dump({'completed': $I, 'total': $TOTAL, 'last': '$NAME/$DEVICE'}, open('$PROGRESS', 'w'))"
done < /tmp/matrix_todo.tsv

echo "=== Matrix complete: $(wc -l < "${RESULTS}") total result rows ==="
