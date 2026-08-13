#!/usr/bin/env python3
# services/llm-bench/aggregate-results.py
import json
import csv
import sys


def main(results_path="results.jsonl", csv_path="results.csv"):
    rows = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rows.sort(key=lambda r: (r["params_total_b"], r["device"]))

    fields = [
        "name", "params_total_b", "params_active_b", "quant", "device",
        "status", "pp_tok_s", "tg_tok_s", "vram_mb", "error", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} rows written to {csv_path}")
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"  {len(ok)} ok, {len(failed)} failed")
    for r in failed:
        print(f"  FAILED: {r['name']} ({r['device']}): {r['error']}")

    if ok:
        print("\n  name                     params  device        pp_tok/s   tg_tok/s")
        for r in ok:
            print(
                f"  {r['name']:<24} {r['params_total_b']:>6.1f}B {r['device']:<12} "
                f"{r['pp_tok_s'] or 0:>9.1f} {r['tg_tok_s'] or 0:>10.1f}"
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
