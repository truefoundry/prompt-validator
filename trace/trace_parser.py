"""Parse TFY span data into structured TraceInput objects."""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceInput:
    span_id: str
    trace_id: str
    timestamp: str
    system_prompt: str
    user_message: str
    trace_output: str
    model_name: str
    application: str
    latency_ms: float
    cost_usd: float
    prompt_fqn: str = ""


def load_spans_from_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_live_spans(
    hours: int = 24,
    prompt_fqn_filter: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 200,
    tfy_host: str | None = None,
    tfy_api_key: str | None = None,
    data_routing_destination: str = "default",
    email_filter: str | None = None,
) -> list[dict]:
    """Fetch spans from TFY.

    Args:
        hours: Hours to look back when start_time/end_time are not provided.
        prompt_fqn_filter: Base prompt FQN to filter client-side (substring match).
        start_time: Explicit start datetime (UTC). If None, derived from hours.
        end_time: Explicit end datetime (UTC). If None, defaults to now.
        limit: Max number of spans to fetch. Caps SDK pagination so the call
               returns quickly instead of exhausting all pages.
        tfy_host: TrueFoundry tenant base URL. If None, falls back to trace/.env / env vars.
        tfy_api_key: TrueFoundry API key. If None, falls back to trace/.env / env vars.
        data_routing_destination: TFY data routing destination (default: "default").
        email_filter: Filter spans by creator email via createdBySubjectSlug.

    Note: FQN filtering is done client-side after fetching because the TFY API
    does not reliably support string operators on tfy.prompt_version_fqn.
    """
    # Explicit args take priority; fall back to trace/.env then environment variables.
    if not tfy_host or not tfy_api_key:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        try:
            from dotenv import dotenv_values
            env_vals = dotenv_values(env_path)
        except ImportError:
            env_vals = {}
        tfy_host = tfy_host or env_vals.get("TFY_HOST") or os.environ.get("TFY_HOST", "")
        tfy_api_key = tfy_api_key or env_vals.get("TFY_API_KEY") or os.environ.get("TFY_API_KEY", "")

    if not tfy_host:
        raise ValueError("TFY_HOST not provided. Enter it in the UI or add it to trace/.env")
    if not tfy_api_key:
        raise ValueError("TFY_API_KEY not provided. Enter it in the UI or add it to trace/.env")

    from truefoundry_sdk import TrueFoundry, SpanAttributeFilter
    from truefoundry_sdk.types.span_attribute_filter_operator import SpanAttributeFilterOperator

    # Create a fresh client scoped to the FloQast tenant credentials
    tfy_client = TrueFoundry(base_url=tfy_host, api_key=tfy_api_key)

    now = datetime.now(timezone.utc)
    if end_time is None:
        end_time = now
    if start_time is None:
        start_time = end_time - timedelta(hours=hours)

    logger.info(
        f"[TRACE_FETCH] start={start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"end={end_time.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"fqn_filter={prompt_fqn_filter or 'none'} "
        f"email_filter={email_filter or 'none'} "
        f"destination={data_routing_destination}"
    )

    filters = [
        SpanAttributeFilter(
            span_attribute_key="tfy.span_type",
            operator=SpanAttributeFilterOperator.EQUAL,
            value="ChatCompletion",
        ),
    ]
    if prompt_fqn_filter:
        filters.append(
            SpanAttributeFilter(
                span_attribute_key="tfy.prompt_version_fqn",
                operator=SpanAttributeFilterOperator.IN,
                value=[prompt_fqn_filter],
            )
        )

    if email_filter:
        filters.append({
            "spanFieldName": "createdBySubjectSlug",
            "operator": "IN",
            "value": [email_filter],
        })

    raw_spans = tfy_client.traces.query_spans(
        data_routing_destination=data_routing_destination,
        start_time=start_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        end_time=end_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        filters=filters,
        sort_direction="desc",
        limit=limit,
    )
    spans = [span.model_dump() for span in raw_spans]
    logger.info(f"[TRACE_FETCH] Raw spans returned: {len(spans)}")

    return spans


def _extract_user_message(messages: list[dict]) -> str | None:
    """Return the combined content of all user-role messages, skipping unresolved template placeholders.

    Some prompts have multiple user turns (data summary, transactions, instructions, etc.).
    The last user turn is often a short template like "Translate to {{language}} language."
    which may still contain unresolved {{variable}} placeholders — those are skipped.
    All resolved user messages are joined so the full input context is preserved.
    """
    import re
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None

    resolved = []
    for m in user_msgs:
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        # Skip messages that are purely unresolved template placeholders
        # (entire content is one or more {{variable}} tokens with no other text)
        stripped = re.sub(r"\{\{[^}]+\}\}", "", content).strip()
        if not stripped:
            continue
        resolved.append(content)

    if resolved:
        return "\n\n".join(resolved)

    # All messages were unresolved templates — fall back to last user message
    return str(user_msgs[-1].get("content", ""))


def _extract_system_prompt(messages: list[dict]) -> str:
    """Return concatenated content of all system-role messages."""
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if not sys_msgs:
        return ""
    return "\n\n".join(str(m.get("content", "")) for m in sys_msgs)


def _extract_llm_output(tfy_output_str) -> str:
    """Extract the assistant message content from tfy.output (string or dict)."""
    try:
        output = json.loads(tfy_output_str) if isinstance(tfy_output_str, str) else tfy_output_str
    except (json.JSONDecodeError, TypeError):
        return ""
    try:
        content = output["choices"][0]["message"]["content"]
        if content is None:
            return ""
        # content may be a dict (structured output) — serialize it back to a string
        return content if isinstance(content, str) else json.dumps(content)
    except (KeyError, IndexError, TypeError):
        return ""


def parse_spans_to_inputs(spans: list[dict]) -> list[TraceInput]:
    """Filter Model spans and parse into TraceInput objects.

    Returns parsed inputs. Also populates parse_spans_to_inputs.skip_reasons
    with a dict of {reason: count} for diagnostic purposes.
    Also populates parse_spans_to_inputs.sample_span_names with up to 5 unique
    span names seen, for diagnostics.
    """
    results: list[TraceInput] = []
    seen_span_ids: set[str] = set()
    skip_reasons: dict[str, int] = {}
    _seen_span_names: list[str] = []

    def _skip(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for span in spans:
        attrs: dict[str, Any] = span.get("span_attributes") or {}

        span_type = attrs.get("tfy.span_type")
        if span_type not in ("Model", "ChatCompletion"):
            _skip(f"wrong_span_type:{span_type or 'None'}")
            continue

        span_id = span.get("span_id", "")
        if span_id in seen_span_ids:
            _skip("duplicate_span_id")
            continue
        seen_span_ids.add(span_id)

        # FQN resolution order:
        # 1. tfy.prompt_version_fqn top-level attribute
        # 2. prompt_version_fqn inside tfy.input JSON (TFY gateway spans)
        # 3. span_name after stripping provider prefix (must contain ":" to be a real FQN)
        _MODEL_PROVIDER_PREFIXES = (
            "openai/", "anthropic/", "google/", "mistral/", "cohere/",
            "fq11-bedrock/", "automation-bedrock/", "automation-eu-bedrock/",
        )
        # 1) Top-level span attribute (standard path)
        raw_fqn = attrs.get("tfy.prompt_version_fqn", "")
        if raw_fqn and ":" in str(raw_fqn) and "/" in str(raw_fqn):
            prompt_fqn = str(raw_fqn)
        else:
            # 2) FQN embedded inside tfy.input JSON (TFY gateway spans store it there)
            _tfy_input_raw = attrs.get("tfy.input", "")
            _input_fqn = ""
            if _tfy_input_raw:
                try:
                    _parsed = json.loads(_tfy_input_raw) if isinstance(_tfy_input_raw, str) else _tfy_input_raw
                    _input_fqn = _parsed.get("prompt_version_fqn", "") or ""
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            if _input_fqn and ":" in str(_input_fqn) and "/" in str(_input_fqn):
                prompt_fqn = str(_input_fqn)
            else:
                # 3) Fall back to span name
                span_name = span.get("span_name", "") or ""
                for prefix in ("ChatCompletion: ", "Model: "):
                    if span_name.startswith(prefix):
                        span_name = span_name[len(prefix):]
                        break
                span_name = span_name.strip()
                # Collect sample span names for diagnostics (up to 5 unique values)
                if span_name and span_name not in _seen_span_names and len(_seen_span_names) < 5:
                    _seen_span_names.append(span_name)

                if (
                    not span_name
                    or ":" not in span_name
                    or any(span_name.startswith(p) for p in _MODEL_PROVIDER_PREFIXES)
                ):
                    _skip("no_prompt_fqn_bare_model")
                    continue
                prompt_fqn = span_name

        # ChatCompletion spans store rendered messages in tfy.resolved_input;
        # Model spans store them directly in tfy.input.
        # Fall back to tfy.input for ChatCompletion spans that lack resolved_input.
        input_key = "tfy.resolved_input" if span_type == "ChatCompletion" else "tfy.input"
        tfy_input_str = attrs.get(input_key, "") or attrs.get("tfy.input", "")
        if not tfy_input_str:
            _skip(f"missing_{input_key.replace('.','_')}")
            warnings.warn(f"Span {span_id}: missing {input_key} — skipped")
            continue

        try:
            tfy_input = json.loads(tfy_input_str) if isinstance(tfy_input_str, str) else tfy_input_str
        except (json.JSONDecodeError, TypeError):
            _skip("invalid_json_input")
            warnings.warn(f"Span {span_id}: {input_key} is not valid JSON — skipped")
            continue

        messages = tfy_input.get("messages", [])
        user_message = _extract_user_message(messages)
        if user_message is None:
            _skip("no_user_message")
            warnings.warn(f"Span {span_id}: no user role message found — skipped")
            continue

        system_prompt = _extract_system_prompt(messages)
        tfy_output_str = attrs.get("tfy.output", "")
        trace_output = _extract_llm_output(tfy_output_str) if tfy_output_str else ""

        metadata = attrs.get("tfy.request.metadata") or {}
        application = metadata.get("application", "") if isinstance(metadata, dict) else ""

        results.append(
            TraceInput(
                span_id=span_id,
                trace_id=span.get("trace_id", ""),
                timestamp=span.get("timestamp", ""),
                system_prompt=system_prompt,
                user_message=user_message,
                trace_output=trace_output,
                model_name=attrs.get("tfy.model.name", ""),
                application=application,
                latency_ms=float(attrs.get("tfy.model.metric.latency_in_ms") or 0),
                cost_usd=float(attrs.get("tfy.model.metric.cost_in_usd") or 0),
                prompt_fqn=prompt_fqn,
            )
        )

    # Attach diagnostics as function attributes
    parse_spans_to_inputs.skip_reasons = skip_reasons  # type: ignore[attr-defined]
    parse_spans_to_inputs.sample_span_names = _seen_span_names  # type: ignore[attr-defined]
    return results


if __name__ == "__main__":
    default_path = os.path.join(os.path.dirname(__file__), "traces.json")
    spans = load_spans_from_file(default_path)
    inputs = parse_spans_to_inputs(spans)
    print(f"Parsed {len(inputs)} TraceInput objects from {len(spans)} spans\n")
    for ti in inputs:
        print(
            f"  span_id={ti.span_id}  app={ti.application}  model={ti.model_name}\n"
            f"    user_message: {ti.user_message[:80]!r}\n"
            f"    trace_output: {ti.trace_output[:80]!r}\n"
        )
