"""Pipeline: fetch spans for 5 target prompt FQNs and build a test set.

For each target FQN, chunks over the last DAYS days in CHUNK_DAYS increments
to find spans with valid input/output pairs.  Results are written to
trace/testset_5prompts.json.

Usage:
    python trace/build_testset_pipeline.py
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────

# 5 target FQNs (base FQN without version so we match any version).
# Selected from prompts-floqast.json, confirmed to have active traces.
TARGET_FQNS = [
    "chat_prompt:floqast/fq4/variance-analysis-v2",
    "chat_prompt:floqast/automation/variance-analysis",
    "chat_prompt:floqast/fq4/variance-analysis",
    "chat_prompt:floqast/automation/variance-analysis-v2",
    "chat_prompt:floqast/automation/variance-analysis-v2-balance-sheet",
]

DAYS = 30          # look this many days back (wider window for unique inputs)
CHUNK_DAYS = 7     # fetch in weekly chunks
LIMIT_PER_CHUNK = 500  # max spans per API call
MAX_PER_FQN = 20   # stop collecting once we have this many unique test cases for an FQN

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "testset_5prompts.json")

# ── Helpers copied / adapted from trace_parser.py ────────────────────────────

import re

_PURE_PLACEHOLDER = re.compile(r"^\s*\{\{[^}]+\}\}\s*$")


def _render(text: str, variables: dict) -> str:
    def replacer(m: re.Match) -> str:
        key = m.group(1)
        return str(variables[key]) if key in variables else m.group(0)
    return re.sub(r"\{\{(\w+)\}\}", replacer, text)


def _is_pure_placeholder(text: str) -> bool:
    return bool(_PURE_PLACEHOLDER.match(text))


def _extract_output(tfy_output_str: str) -> str:
    try:
        data = json.loads(tfy_output_str)
        return data["choices"][0]["message"]["content"] or ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def _extract_input_output(span: dict) -> tuple[str, str, str] | None:
    """Return (system_prompt, user_input, expected_output) or None."""
    attrs = span.get("span_attributes") or {}

    resolved_input_str = attrs.get("tfy.resolved_input") or attrs.get("tfy.input") or ""
    if not resolved_input_str:
        return None

    try:
        resolved = json.loads(resolved_input_str)
    except (json.JSONDecodeError, TypeError):
        return None

    variables: dict = {}
    pv_str = attrs.get("tfy.prompt_variables", "")
    if pv_str:
        try:
            variables = json.loads(pv_str)
        except (json.JSONDecodeError, TypeError):
            pass

    messages = resolved.get("messages", [])
    if not messages:
        return None

    system_prompt = ""
    user_messages: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = _render(str(msg.get("content", "")).strip(), variables)
        if not content:
            continue
        if role == "system":
            system_prompt = content
        elif role == "user":
            if not _is_pure_placeholder(content):
                # Also skip messages whose resolved text is still fully unresolved
                stripped = re.sub(r"\{\{[^}]+\}\}", "", content).strip()
                if stripped:
                    user_messages.append(content)

    if not user_messages:
        return None

    expected_output = _extract_output(attrs.get("tfy.output", ""))
    if not expected_output:
        return None

    return system_prompt, "\n\n".join(user_messages), expected_output


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _load_tfy_client():
    """Create a TFY client from trace/.env credentials."""
    from dotenv import dotenv_values
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_vals = dotenv_values(env_path)

    tfy_host = env_vals.get("TFY_HOST") or os.environ.get("TFY_HOST", "")
    tfy_api_key = env_vals.get("TFY_API_KEY") or os.environ.get("TFY_API_KEY", "")

    if not tfy_host:
        raise ValueError("TFY_HOST not found in trace/.env")
    if not tfy_api_key:
        raise ValueError("TFY_API_KEY not found in trace/.env")

    from truefoundry_sdk import TrueFoundry
    return TrueFoundry(base_url=tfy_host, api_key=tfy_api_key)


def fetch_spans_in_window(
    tfy_client,
    start_time: datetime,
    end_time: datetime,
    limit: int = 500,
) -> list[dict]:
    """Fetch ChatCompletion spans in a time window."""
    from truefoundry_sdk import SpanAttributeFilter
    from truefoundry_sdk.types.span_attribute_filter_operator import SpanAttributeFilterOperator

    try:
        spans = tfy_client.traces.query_spans(
            data_routing_destination="default",
            start_time=start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            end_time=end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            filters=[
                SpanAttributeFilter(
                    span_attribute_key="tfy.span_type",
                    operator=SpanAttributeFilterOperator.EQUAL,
                    value="ChatCompletion",
                ),
            ],
            sort_direction="desc",
            limit=limit,
        )
        return [s.model_dump() for s in spans]
    except Exception as exc:
        print(f"  [WARN] API error: {exc}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> list[dict]:
    print("=" * 60)
    print("Build test set pipeline")
    print(f"Target FQNs: {len(TARGET_FQNS)}")
    print(f"Lookback: {DAYS} days in {CHUNK_DAYS}-day chunks")
    print("=" * 60)

    tfy_client = _load_tfy_client()

    # Use a set for O(1) base-FQN lookup
    target_fqns_set = set(TARGET_FQNS)

    # bucket -> list of test cases, keyed by base FQN
    collected: dict[str, list[dict]] = defaultdict(list)
    seen_span_ids: set[str] = set()
    # Deduplicate by (base_fqn, user_input) to avoid identical inputs within the same FQN
    seen_inputs: set[tuple[str, str]] = set()

    now = datetime.now(timezone.utc)

    for offset in range(0, DAYS, CHUNK_DAYS):
        chunk_end = now - timedelta(days=offset)
        chunk_start = now - timedelta(days=min(offset + CHUNK_DAYS, DAYS))

        # Check if we still need more spans for any FQN
        remaining = [
            fqn for fqn in TARGET_FQNS
            if len(collected[fqn]) < MAX_PER_FQN
        ]
        if not remaining:
            print("\nAll FQNs have enough test cases. Stopping early.")
            break

        print(
            f"\nChunk {chunk_start.date()} → {chunk_end.date()} "
            f"(need more for {len(remaining)} FQN(s)) ...",
            end=" ",
            flush=True,
        )

        spans = fetch_spans_in_window(tfy_client, chunk_start, chunk_end, limit=LIMIT_PER_CHUNK)
        print(f"{len(spans)} spans fetched", end="")

        chunk_hits = 0
        for span in spans:
            span_id = span.get("span_id", "")
            if span_id in seen_span_ids:
                continue

            attrs = span.get("span_attributes") or {}
            fqn_versioned: str = attrs.get("tfy.prompt_version_fqn", "") or ""

            # Strip version suffix to get base FQN:
            #   "chat_prompt:floqast/repo/name:version" -> "chat_prompt:floqast/repo/name"
            # FQNs have the pattern type:workspace/repo/name[:version]; the last
            # colon (when there are 2+) separates the version.
            if fqn_versioned.count(":") >= 2:
                fqn_base = fqn_versioned.rsplit(":", 1)[0]
            else:
                fqn_base = fqn_versioned

            matched_base = fqn_base if fqn_base in target_fqns_set else None

            if not matched_base:
                continue

            if len(collected[matched_base]) >= MAX_PER_FQN:
                continue

            result = _extract_input_output(span)
            if result is None:
                continue

            seen_span_ids.add(span_id)
            system_prompt, user_input, expected_output = result

            # Skip if we've already seen this exact input for this base FQN
            input_key = (matched_base, user_input)
            if input_key in seen_inputs:
                continue
            seen_inputs.add(input_key)

            tc_id = sum(len(v) for v in collected.values())
            tc = {
                "test_case_id": str(tc_id),
                "test_case_name": f"{fqn_versioned} — {span_id[:8]}",
                "system_prompt": system_prompt,
                "data": {"input": user_input},
                "expected_output": expected_output,
                "metadata": {
                    "span_id": span_id,
                    "prompt_fqn": fqn_versioned,
                    "base_fqn": matched_base,
                    "model_fqn": attrs.get("tfy.model.fqn", ""),
                    "model_name": attrs.get("tfy.model.name", ""),
                    "timestamp": span.get("timestamp", ""),
                },
            }
            collected[matched_base].append(tc)
            chunk_hits += 1

        if chunk_hits:
            print(f", {chunk_hits} matched")
        else:
            print()

    # Flatten, deduplicate globally by user_input, and re-number
    all_cases: list[dict] = []
    seen_global_inputs: set[str] = set()
    for base_fqn in TARGET_FQNS:
        for tc in collected[base_fqn]:
            user_input = tc["data"]["input"]
            if user_input in seen_global_inputs:
                continue
            seen_global_inputs.add(user_input)
            tc["test_case_id"] = str(len(all_cases))
            all_cases.append(tc)

    return all_cases


def build_grouped_output(test_cases: list[dict]) -> dict:
    """Build a grouped structure: { prompt_fqn -> {system_prompt, test_cases[]} }."""
    from collections import OrderedDict

    groups: dict[str, dict] = OrderedDict()
    for base_fqn in TARGET_FQNS:
        groups[base_fqn] = {"base_fqn": base_fqn, "system_prompt": "", "test_cases": []}

    local_ids: dict[str, int] = {fqn: 0 for fqn in TARGET_FQNS}

    for tc in test_cases:
        base_fqn = tc["metadata"]["base_fqn"]
        if base_fqn not in groups:
            continue
        if not groups[base_fqn]["system_prompt"]:
            groups[base_fqn]["system_prompt"] = tc["system_prompt"]

        local_id = local_ids[base_fqn]
        local_ids[base_fqn] += 1

        groups[base_fqn]["test_cases"].append({
            "test_case_id": str(local_id),
            "test_case_name": tc["test_case_name"],
            "data": tc["data"],
            "expected_output": tc["expected_output"],
            "metadata": tc["metadata"],
        })

    # Drop empty groups
    return {
        "total_test_cases": len(test_cases),
        "groups": [g for g in groups.values() if g["test_cases"]],
    }


def print_summary(test_cases: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"Test set summary: {len(test_cases)} total test cases")
    print("=" * 60)
    counts = Counter(tc["metadata"]["base_fqn"] for tc in test_cases)
    for base_fqn in TARGET_FQNS:
        cnt = counts.get(base_fqn, 0)
        status = "✓" if cnt > 0 else "✗ (no spans found)"
        print(f"  {cnt:3d}x  {base_fqn}  {status}")


if __name__ == "__main__":
    test_cases = run_pipeline()

    grouped = build_grouped_output(test_cases)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(grouped, f, indent=2)

    print_summary(test_cases)
    print(f"\nSaved to: {OUTPUT_PATH}")
