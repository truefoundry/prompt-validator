"""
Run CVS guardrail test inputs against the deployed prompt and print results.

Usage:
    python run_cvs_tests.py [--inputs cvs_guardrail_tests/test_inputs_simple.json]
                            [--tests  cvs_guardrail_tests/tests_1.json]
                            [--fqn    chat_prompt:truefoundry/new-repo/cvs_guardrails_prompt:1]
"""
import argparse
import json
import sys
import time
import httpx

API_URL = "http://localhost:21120/chat"
DEFAULT_FQN = "chat_prompt:truefoundry/new-repo/cvs_guardrails_prompt:1"
DEFAULT_INPUTS = "cvs_guardrail_tests/test_inputs_simple.json"
DEFAULT_TESTS  = "cvs_guardrail_tests/tests_1.json"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def load_inputs(path: str) -> list[dict]:
    with open(path) as f:
        raw = json.load(f)
    result = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            result.append({"test_case_id": str(i), "test_case_name": item, "input": item, "expected_output": None})
        elif isinstance(item, dict):
            inp = item.get("data", {}).get("input") or item.get("input", "")
            expected = item.get("expected_output")
            result.append({
                "test_case_id": item.get("test_case_id", str(i)),
                "test_case_name": item.get("test_case_name", inp),
                "input": inp,
                "expected_output": expected,
            })
    return result


def call_backend(prompt_fqn: str, user_input: str, idx: int) -> str | None:
    payload = {
        "sessionId": f"cvs-test-{idx}-{int(time.time())}",
        "promptFQN": prompt_fqn,
        "userPromptTemplate": "{{input}}",
        "type": "validation",
        "recommendations": None,
    }
    # We need to actually get the model output for this input.
    # Use the llm_judge type to get raw model output via prompt FQN + test case.
    payload_direct = {
        "sessionId": f"cvs-direct-{idx}-{int(time.time())}",
        "promptFQN": prompt_fqn,
        "type": "llm_judge",
        "recommendations": None,
        "testCases": [
            {
                "test_case_id": str(idx),
                "test_case_name": f"test_{idx}",
                "data": {"input": user_input},
                "expected_output": "",
            }
        ],
    }
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(API_URL, json=payload_direct)
            resp.raise_for_status()
            data = resp.json()
        content = data.get("content", {})
        # llm_judge returns test_results with original_output
        test_results = content.get("test_results", [])
        if test_results:
            return test_results[0].get("original_output", "")
        # fallback: check eval_result
        return str(content)
    except Exception as e:
        return f"[ERROR] {e}"


def run_simple_prompt(prompt_fqn: str, user_input: str, idx: int) -> dict:
    """Call the prompt via verify_tests_exact to get actual LLM output + pass/fail."""
    payload = {
        "sessionId": f"cvs-run-{idx}-{int(time.time())}",
        "promptFQN": prompt_fqn,
        "type": "verify_tests_exact",
        "recommendations": None,
        "testCases": [
            {
                "test_case_id": str(idx),
                "test_case_name": f"test_{idx}",
                "data": {"input": user_input},
                "expected_output": "inScope: false",
            }
        ],
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency = round(time.time() - t0, 2)
        content = data.get("content", {})
        # verify_tests_exact returns all_tests + all_metrics
        all_tests = content.get("all_tests", [])
        if all_tests:
            t = all_tests[0]
            actual = t.get("actual_output", "")
            passed = t.get("pass_status", None)
            return {"actual": actual, "pass": passed, "score": None, "latency": latency}
        return {"actual": str(content)[:300], "score": None, "pass": None, "latency": latency}
    except Exception as e:
        return {"actual": f"[ERROR] {e}", "score": None, "pass": None, "latency": round(time.time() - t0, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=DEFAULT_INPUTS)
    parser.add_argument("--fqn", default=DEFAULT_FQN)
    args = parser.parse_args()

    test_cases = load_inputs(args.inputs)
    total = len(test_cases)

    print(f"\n{BOLD}CVS Guardrail Test Run{RESET}")
    print(f"Prompt FQN : {args.fqn}")
    print(f"Input file : {args.inputs}")
    print(f"Test cases : {total}\n")
    print("─" * 80)

    passed = 0
    failed = 0
    errors = 0
    latencies: list[float] = []
    total_start = time.time()

    for tc in test_cases:
        idx = tc["test_case_id"]
        name = tc["test_case_name"]
        user_input = tc["input"]

        print(f"\n[{idx}] {name}")
        print(f"  Input    : {user_input}")

        result = run_simple_prompt(args.fqn, user_input, int(idx) if str(idx).isdigit() else 0)
        actual = result.get("actual", "")
        score = result.get("score")
        is_pass = result.get("pass")
        latency = result.get("latency", 0.0)
        latencies.append(latency)

        print(f"  Output   : {actual[:200]}")
        print(f"  Latency  : {latency}s")

        if actual.startswith("[ERROR]"):
            print(f"  {RED}STATUS   : ERROR{RESET}")
            errors += 1
        elif is_pass is True:
            print(f"  {GREEN}STATUS   : PASS  (score={score}){RESET}")
            passed += 1
        elif is_pass is False:
            print(f"  {RED}STATUS   : FAIL  (score={score}){RESET}")
            failed += 1
        else:
            in_scope = "inscope: false" in actual.lower() or "inScope: false" in actual
            if in_scope:
                print(f"  {GREEN}STATUS   : PASS  (inScope: false detected){RESET}")
                passed += 1
            else:
                print(f"  {YELLOW}STATUS   : UNKNOWN  (no pass/fail info){RESET}")
                errors += 1

    total_elapsed = round(time.time() - total_start, 2)
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    min_latency = round(min(latencies), 2) if latencies else 0.0
    max_latency = round(max(latencies), 2) if latencies else 0.0

    print("\n" + "─" * 80)
    print(f"\n{BOLD}Results: {GREEN}{passed} passed{RESET} / {RED}{failed} failed{RESET} / {YELLOW}{errors} errors{RESET} (total: {total}){RESET}")
    print(f"{BOLD}Latency: avg={avg_latency}s  min={min_latency}s  max={max_latency}s  total={total_elapsed}s{RESET}")


if __name__ == "__main__":
    main()
