#!/usr/bin/env python3
"""
Agentic-capability eval — tool-calling correctness, not throughput.
Sends each test case to a running llama-server instance's OpenAI-compatible
/v1/chat/completions endpoint with real tool schemas, grades the response.
"""
import json
import sys
import urllib.request

SERVER = "http://localhost:8080/v1/chat/completions"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'Chicago'"},
                },
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
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to read"},
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
                "properties": {
                    "query": {"type": "string", "description": "Text or pattern to search for"},
                },
                "required": ["query"],
            },
        },
    },
]

# Fake tool results for the multi-step test — the harness plays "tool executor"
FAKE_SEARCH_RESULT = json.dumps({"matches": ["models/user.py"]})

TESTS = [
    {
        "id": "A_single_tool_call",
        "description": "Single, unambiguous tool call — tests basic tool-call format correctness",
        "messages": [{"role": "user", "content": "What's the weather in Chicago right now?"}],
        "tools": TOOLS,
        "expect": {"calls_tool": True, "tool_name": "get_weather", "arg_contains": {"location": "Chicago"}},
    },
    {
        "id": "B_multi_arg_extraction",
        "description": "Correct extraction of multiple structured arguments from natural language",
        "messages": [{"role": "user", "content": "Read lines 10 through 20 of config.py"}],
        "tools": TOOLS,
        "expect": {"calls_tool": True, "tool_name": "read_file",
                   "arg_contains": {"path": "config.py", "start_line": 10, "end_line": 20}},
    },
    {
        "id": "C_multistep_chaining",
        "description": "Multi-step: search first, then use the result to inform a second tool call",
        "messages": [{"role": "user", "content": "Find the file that defines the User class, then show me its first 15 lines."}],
        "tools": TOOLS,
        "expect": {"calls_tool": True, "tool_name": "search_files", "multistep": True},
    },
    {
        "id": "D_correct_non_call",
        "description": "Should NOT call a tool — no tool fits, model should just answer directly",
        "messages": [{"role": "user", "content": "What's 15% of 240?"}],
        "tools": TOOLS,
        "expect": {"calls_tool": False, "answer_contains": "36"},
    },
    {
        "id": "E_structured_json_no_tools",
        "description": "Structured JSON output without any tool schema — pure format-following",
        "messages": [{"role": "user", "content": "Extract the name and email from this text as JSON with keys 'name' and 'email', nothing else in your response: Contact Jane Doe at jane@example.com for details."}],
        "tools": None,
        "expect": {"calls_tool": False, "valid_json": True, "json_contains": {"email": "jane@example.com"}},
    },
]


def call_llm(messages, tools):
    payload = {"model": "local", "messages": messages, "temperature": 0}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        SERVER, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def grade(test, response):
    try:
        msg = response["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        return {"pass": False, "reason": f"malformed response: {e}", "raw": response}

    tool_calls = msg.get("tool_calls") or []
    called = len(tool_calls) > 0
    expect = test["expect"]

    if expect.get("calls_tool") and not called:
        return {"pass": False, "reason": "expected a tool call, got none", "content": msg.get("content")}
    if not expect.get("calls_tool") and called:
        return {"pass": False, "reason": f"expected NO tool call, got {tool_calls[0]['function']['name']}", "content": msg.get("content")}

    if called:
        fn = tool_calls[0]["function"]
        name_ok = fn["name"] == expect.get("tool_name")
        try:
            args = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            return {"pass": False, "reason": "tool call arguments not valid JSON", "raw_args": fn["arguments"]}
        arg_checks = expect.get("arg_contains", {})
        args_ok = all(str(args.get(k)) == str(v) for k, v in arg_checks.items())
        if not name_ok:
            return {"pass": False, "reason": f"wrong tool: called {fn['name']}, expected {expect.get('tool_name')}"}
        if not args_ok:
            return {"pass": False, "reason": f"wrong/missing args: got {args}, expected to contain {arg_checks}"}
        return {"pass": True, "reason": "correct tool + args", "tool_call": fn}

    content = (msg.get("content") or "").strip()
    if expect.get("valid_json"):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # try to find a JSON object inside the content (some models wrap in prose)
            import re
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                return {"pass": False, "reason": "no valid JSON found in response", "content": content}
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                return {"pass": False, "reason": "JSON-like text found but not parseable", "content": content}
        checks = expect.get("json_contains", {})
        ok = all(str(v) in str(parsed.get(k, "")) for k, v in checks.items())
        return {"pass": ok, "reason": "json field check" if ok else f"missing/wrong fields, got {parsed}", "content": content}

    if "answer_contains" in expect:
        ok = expect["answer_contains"] in content
        return {"pass": ok, "reason": "answer contains expected substring" if ok else f"expected '{expect['answer_contains']}' in response", "content": content}

    return {"pass": True, "reason": "no specific check, tool-call presence matched expectation", "content": content}


def run_multistep(test):
    """Test C needs a second round-trip: feed the tool result back and check the follow-up call."""
    messages = list(test["messages"])
    resp1 = call_llm(messages, test["tools"])
    msg1 = resp1["choices"][0]["message"]
    tool_calls = msg1.get("tool_calls") or []
    if not tool_calls or tool_calls[0]["function"]["name"] != "search_files":
        return {"pass": False, "reason": "step 1: did not call search_files as expected", "content": msg1.get("content")}

    messages.append(msg1)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_calls[0]["id"],
        "content": FAKE_SEARCH_RESULT,
    })
    resp2 = call_llm(messages, test["tools"])
    msg2 = resp2["choices"][0]["message"]
    tool_calls2 = msg2.get("tool_calls") or []
    if tool_calls2 and tool_calls2[0]["function"]["name"] == "read_file":
        try:
            args2 = json.loads(tool_calls2[0]["function"]["arguments"])
        except json.JSONDecodeError:
            return {"pass": False, "reason": "step 2: read_file args not valid JSON"}
        if "models/user.py" in str(args2.get("path", "")):
            return {"pass": True, "reason": "correctly chained search -> read_file with the search result's path"}
        return {"pass": False, "reason": f"step 2: read_file called but wrong path: {args2}"}
    return {"pass": False, "reason": "step 2: did not follow up with read_file using the search result", "content": msg2.get("content")}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    results = []
    for test in TESTS:
        try:
            if test["id"] == "C_multistep_chaining":
                result = run_multistep(test)
            else:
                resp = call_llm(test["messages"], test["tools"])
                result = grade(test, resp)
        except Exception as e:
            result = {"pass": False, "reason": f"exception: {e}"}
        result["test_id"] = test["id"]
        result["description"] = test["description"]
        results.append(result)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {test['id']}: {result['reason']}")

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{model_name}: {passed}/{len(TESTS)} passed")

    with open("/opt/llm-bench/agentic-results.jsonl", "a") as f:
        f.write(json.dumps({"model": model_name, "passed": passed, "total": len(TESTS), "results": results}) + "\n")


if __name__ == "__main__":
    main()
