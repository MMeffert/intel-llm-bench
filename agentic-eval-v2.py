#!/usr/bin/env python3
"""
Agentic-capability eval v2 — harder, discriminating tests.

v1 (agentic-eval.py) was a smoke test: 5 easy cases that gemma-4-12b,
devstral-2-24b, and qwen3.6-27b all tied on (5/5). This version targets
exactly the gap v1 couldn't see: longer tool chains, tool selection under
distractor noise, error recovery, multi-call aggregation requiring real
arithmetic/comparison over tool results, and code-fix correctness (grading
actual output content, not just call format).
"""
import json
import os
import sys
import urllib.request

# Runs against a scratch port (8081), never the production local-llm.service on
# 8080 -- avoids any interruption to the already-deployed, potentially-in-use
# gemma-4-12b production endpoint while candidates are swapped in and out.
SERVER = f"http://localhost:{os.environ.get('LLM_BENCH_PORT', '8081')}/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a range of lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the codebase for a text pattern, returns matching file paths",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_metadata",
            "description": "Get metadata for a file: line count, size in bytes, last modified date. Use this instead of read_file when you only need counts/size/dates, not the file's actual content.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact snippet of a file's content with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_content": {"type": "string"},
                    "new_content": {"type": "string"},
                },
                "required": ["path", "old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_pattern",
            "description": "Search for a regex pattern within a single specific file",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite for a given path and return pass/fail counts",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def call_llm(messages, tools=TOOLS, temperature=0):
    payload = {"model": "local", "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        SERVER, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def first_call(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None, None
    fn = tcs[0]["function"]
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError:
        args = None
    return fn["name"], args


def tool_result_msg(tool_calls, content):
    return {"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": content}


# ---------------------------------------------------------------------------
# F: Long chain (3 steps) + arithmetic over a tool result (10% of line count,
# rounded) — tests whether the model actually uses a returned number instead
# of just chaining calls mechanically.
# ---------------------------------------------------------------------------
def test_F():
    messages = [{
        "role": "user",
        "content": "Find the file that defines the Order class, check how many lines "
                   "it has, then read just the first 10% of the file (rounded to the "
                   "nearest line).",
    }]
    r1 = call_llm(messages)
    m1 = r1["choices"][0]["message"]
    name1, args1 = first_call(m1)
    if name1 != "search_files":
        return {"pass": False, "reason": f"step 1: expected search_files, got {name1}"}
    messages.append(m1)
    messages.append(tool_result_msg(m1["tool_calls"], json.dumps({"matches": ["models/order.py"]})))

    r2 = call_llm(messages)
    m2 = r2["choices"][0]["message"]
    name2, args2 = first_call(m2)
    if name2 != "get_file_metadata" or not args2 or "order.py" not in str(args2.get("path", "")):
        return {"pass": False, "reason": f"step 2: expected get_file_metadata(models/order.py), got {name2} {args2}"}
    messages.append(m2)
    messages.append(tool_result_msg(m2["tool_calls"], json.dumps({"lines": 240, "size_bytes": 6800, "last_modified": "2026-07-01"})))

    r3 = call_llm(messages)
    m3 = r3["choices"][0]["message"]
    name3, args3 = first_call(m3)
    if name3 != "read_file" or not args3:
        return {"pass": False, "reason": f"step 3: expected read_file, got {name3}"}
    end_line = args3.get("end_line")
    start_line = args3.get("start_line", 1)
    path_ok = "order.py" in str(args3.get("path", ""))
    # 10% of 240 = 24; allow 18-30 for rounding/interpretation slack
    end_ok = isinstance(end_line, int) and 18 <= end_line <= 30
    start_ok = start_line in (0, 1)
    if path_ok and end_ok and start_ok:
        return {"pass": True, "reason": f"correctly chained 3 steps and computed ~10% of 240 lines (end_line={end_line})"}
    return {"pass": False, "reason": f"step 3: wrong args, got path={args3.get('path')} start={start_line} end={end_line} (expected end_line ~24)"}


# ---------------------------------------------------------------------------
# G: Tool selection precision — get_file_metadata is the efficient/correct
# choice among 8 tools including read_file (works but wasteful/wrong intent).
# ---------------------------------------------------------------------------
def test_G():
    messages = [{"role": "user", "content": "How many lines does config.py have?"}]
    r = call_llm(messages)
    m = r["choices"][0]["message"]
    name, args = first_call(m)
    if name == "get_file_metadata" and args and "config.py" in str(args.get("path", "")):
        return {"pass": True, "reason": "selected the precise tool (get_file_metadata) among 8 options"}
    return {"pass": False, "reason": f"expected get_file_metadata(config.py), got {name} {args}"}


# ---------------------------------------------------------------------------
# H: Error recovery — first read_file call 404s with a suggested correction;
# model must retry with the corrected filename, not give up or hallucinate.
# ---------------------------------------------------------------------------
def test_H():
    messages = [{"role": "user", "content": "What does auth.py do? Summarize it briefly."}]
    r1 = call_llm(messages)
    m1 = r1["choices"][0]["message"]
    name1, args1 = first_call(m1)
    if name1 != "read_file":
        return {"pass": False, "reason": f"step 1: expected read_file, got {name1}"}
    messages.append(m1)
    messages.append(tool_result_msg(
        m1["tool_calls"],
        json.dumps({"error": "File not found: auth.py. Did you mean: auth_handler.py?"}),
    ))
    r2 = call_llm(messages)
    m2 = r2["choices"][0]["message"]
    name2, args2 = first_call(m2)
    if name2 == "read_file" and args2 and "auth_handler.py" in str(args2.get("path", "")):
        return {"pass": True, "reason": "correctly retried with the corrected filename after a tool error"}
    return {"pass": False, "reason": f"step 2: expected retry with auth_handler.py, got {name2} {args2}", "content": m2.get("content")}


# ---------------------------------------------------------------------------
# I: Multi-call aggregation — must call get_file_metadata 3x (one per file)
# then correctly compare the returned sizes to answer which is largest.
# ---------------------------------------------------------------------------
def test_I():
    files = {"a.py": 120, "b.py": 340, "c.py": 95}
    messages = [{"role": "user", "content": "Which of these files is largest by line count: a.py, b.py, or c.py?"}]
    calls_made = []
    for _ in range(4):
        r = call_llm(messages)
        m = r["choices"][0]["message"]
        name, args = first_call(m)
        if name is None:
            content = (m.get("content") or "").lower()
            if "b.py" in content and "a.py" not in content.split("b.py")[0][-20:]:
                # crude but effective: final answer names b.py
                if "b.py" in content:
                    return {"pass": True, "reason": f"correctly identified b.py as largest after {len(calls_made)} metadata calls", "calls": calls_made}
            return {"pass": False, "reason": f"final answer did not clearly name b.py as largest: {content!r}", "calls": calls_made}
        if name != "get_file_metadata":
            return {"pass": False, "reason": f"expected get_file_metadata calls only, got {name}"}
        path = str(args.get("path", "")) if args else ""
        target = next((f for f in files if f in path), None)
        calls_made.append(target or path)
        messages.append(m)
        lines = files.get(target, 0)
        messages.append(tool_result_msg(m["tool_calls"], json.dumps({"lines": lines, "size_bytes": lines * 30})))
    return {"pass": False, "reason": "did not converge to a final answer within 4 rounds", "calls": calls_made}


# ---------------------------------------------------------------------------
# J: Code-fix correctness — grades actual generated code content, not just
# call format. Model must fix a real off-by-one bug via edit_file.
# ---------------------------------------------------------------------------
BUGGY_SNIPPET = "    total = 0\n    for i in range(len(items) - 1):\n        total += items[i].price\n    return total"

def test_J():
    messages = [{
        "role": "user",
        "content": "Read cart.py's calculate_total function, then fix the off-by-one "
                   "bug that skips the last item in the loop, using edit_file.",
    }]
    r1 = call_llm(messages)
    m1 = r1["choices"][0]["message"]
    name1, args1 = first_call(m1)
    if name1 != "read_file":
        return {"pass": False, "reason": f"step 1: expected read_file, got {name1}"}
    messages.append(m1)
    messages.append(tool_result_msg(m1["tool_calls"], json.dumps({"content": BUGGY_SNIPPET})))

    r2 = call_llm(messages)
    m2 = r2["choices"][0]["message"]
    name2, args2 = first_call(m2)
    if name2 != "edit_file" or not args2:
        return {"pass": False, "reason": f"step 2: expected edit_file, got {name2}", "content": m2.get("content")}
    new_content = str(args2.get("new_content", ""))
    old_content = str(args2.get("old_content", ""))
    fixed = "range(len(items) - 1)" not in new_content and "range(len(items))" in new_content
    targeted_right_snippet = "range(len(items) - 1)" in old_content or "len(items) - 1" in old_content
    if fixed and targeted_right_snippet:
        return {"pass": True, "reason": "correctly fixed the off-by-one (range(len(items) - 1) -> range(len(items)))"}
    return {"pass": False, "reason": f"fix incorrect or incomplete: old={old_content!r} new={new_content!r}"}


# ---------------------------------------------------------------------------
# K: No-call correctness under heavy distractor noise (8 tools offered).
# ---------------------------------------------------------------------------
def test_K():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    r = call_llm(messages)
    m = r["choices"][0]["message"]
    name, _ = first_call(m)
    if name is not None:
        return {"pass": False, "reason": f"expected no tool call, got {name}"}
    content = (m.get("content") or "")
    if "Paris" in content:
        return {"pass": True, "reason": "correctly answered directly with no tool call, 8 distractor tools offered"}
    return {"pass": False, "reason": f"no tool call (correct) but wrong/missing answer: {content!r}"}


TESTS = [
    ("F_long_chain_with_arithmetic", test_F),
    ("G_tool_selection_precision", test_G),
    ("H_error_recovery", test_H),
    ("I_multicall_aggregation", test_I),
    ("J_code_fix_correctness", test_J),
    ("K_no_call_under_distractor_noise", test_K),
]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    results = []
    for test_id, fn in TESTS:
        try:
            result = fn()
        except Exception as e:
            result = {"pass": False, "reason": f"exception: {e}"}
        result["test_id"] = test_id
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {test_id}: {result['reason']}")

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{model_name}: {passed}/{len(TESTS)} passed (v2, hard eval)")

    with open("/opt/llm-bench/agentic-results-v2.jsonl", "a") as f:
        f.write(json.dumps({"model": model_name, "passed": passed, "total": len(TESTS), "results": results}) + "\n")


if __name__ == "__main__":
    main()
