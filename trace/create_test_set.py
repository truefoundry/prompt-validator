



8"""Generate a test set JSON from traces.json.

Each test case contains:
  - system_prompt  : content of the system message
  - user_input     : fully rendered user messages (prompt_variables substituted, pure placeholders dropped)
  - expected_output: LLM response from tfy.output
  - metadata       : model_fqn, model_name, prompt_fqn, span_id
"""

import json
import re
import os

TRACES_PATH = os.path.join(os.path.dirname(__file__), "traces.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

_PURE_PLACEHOLDER = re.compile(r"^\s*\{\{[^}]+\}\}\s*$")


def _render(text: str, variables: dict) -> str:
    """Substitute {{key}} placeholders using variables dict."""
    def replacer(m):
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)
    return re.sub(r"\{\{(\w+)\}\}", replacer, text)


def _is_pure_placeholder(text: str) -> bool:
    """Return True if the message is only an unresolved {{placeholder}}."""
    return bool(_PURE_PLACEHOLDER.match(text))


def _extract_output(tfy_output_str: str) -> str:
    try:
        data = json.loads(tfy_output_str)
        return data["choices"][0]["message"]["content"] or ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def build_test_cases(spans: list[dict]) -> list[dict]:
    test_cases = []

    for span in spans:
        attrs = span.get("span_attributes") or {}

        resolved_input_str = attrs.get("tfy.resolved_input", "")
        if not resolved_input_str:
            continue

        try:
            resolved = json.loads(resolved_input_str)
        except (json.JSONDecodeError, TypeError):
            continue

        # Parse prompt_variables for substitution
        variables = {}
        pv_str = attrs.get("tfy.prompt_variables", "")
        if pv_str:
            try:
                variables = json.loads(pv_str)
            except (json.JSONDecodeError, TypeError):
                pass

        messages = resolved.get("messages", [])
        if not messages:
            continue

        system_prompt = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = _render(str(msg.get("content", "")).strip(), variables)

            if not content:
                continue
            if role == "system":
                system_prompt = content
            elif role == "user":
                # Drop messages that are still pure unresolved placeholders
                if not _is_pure_placeholder(content):
                    user_messages.append(content)
            # Skip assistant messages (they are template placeholders like {{current_explanation}})

        if not user_messages:
            continue

        expected_output = _extract_output(attrs.get("tfy.output", ""))
        if not expected_output:
            continue

        user_input = "\n\n".join(user_messages)

        test_cases.append({
            "test_case_id": str(len(test_cases)),
            "test_case_name": f"{attrs.get('tfy.prompt_version_fqn', 'unknown')} — {span.get('span_id', '')[:8]}",
            "system_prompt": system_prompt,
            "data": {"input": user_input},
            "expected_output": expected_output,
            "metadata": {
                "span_id": span.get("span_id", ""),
                "prompt_fqn": attrs.get("tfy.prompt_version_fqn", ""),
                "model_fqn": attrs.get("tfy.model.fqn", ""),
                "model_name": attrs.get("tfy.model.name", ""),
            },
        })

    return test_cases


if __name__ == "__main__":
    with open(TRACES_PATH) as f:
        spans = json.load(f)

    print(f"Loaded {len(spans)} spans from traces.json")

    test_cases = build_test_cases(spans)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Created {len(test_cases)} test cases → {OUTPUT_PATH}")

    from collections import Counter
    fqns = Counter(tc["metadata"]["prompt_fqn"] for tc in test_cases)
    print("\nBreakdown by prompt FQN:")
    for fqn, count in fqns.most_common():
        print(f"  {count:3d}x  {fqn}")
