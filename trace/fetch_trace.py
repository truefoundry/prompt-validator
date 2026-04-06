import os
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from truefoundry import client
from truefoundry_sdk import SpanAttributeFilter
from truefoundry_sdk.types.span_attribute_filter_operator import SpanAttributeFilterOperator

DAYS = 90
CHUNK_DAYS = 7

now = datetime.now(timezone.utc)
results = []
seen_ids = set()

for offset in range(0, DAYS, CHUNK_DAYS):
    chunk_end = now - timedelta(days=offset)
    chunk_start = now - timedelta(days=min(offset + CHUNK_DAYS, DAYS))

    print(f"Fetching {chunk_start.date()} → {chunk_end.date()} ...", end=" ", flush=True)
    try:
        spans = client.traces.query_spans(
            data_routing_destination="default",
            start_time=chunk_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            end_time=chunk_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            filters=[
                SpanAttributeFilter(
                    span_attribute_key="tfy.span_type",
                    operator=SpanAttributeFilterOperator.EQUAL,
                    value="ChatCompletion",
                ),
            ],
            sort_direction="desc",
        )

        chunk_count = 0
        for span in spans:
            if span.span_id in seen_ids:
                continue
            attrs = span.span_attributes or {}
            fqn = attrs.get("tfy.prompt_version_fqn", "")
            if fqn and ":" in str(fqn) and "/" in str(fqn):
                seen_ids.add(span.span_id)
                results.append(span.model_dump())
                chunk_count += 1

        print(f"{chunk_count} valid spans")
    except Exception as e:
        print(f"ERROR: {e}")

output_path = os.path.join(os.path.dirname(__file__), "traces.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {len(results)} total valid ChatCompletion spans saved to {output_path}")
