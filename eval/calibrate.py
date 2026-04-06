"""Calibration script: run sample prompts against the backend and print actual scores.

Usage:
    python -m eval.calibrate
    python -m eval.calibrate --base-url http://localhost:21120 --model tfy-ai-vertex/gemini-3-flash-preview
    python -m eval.calibrate --write

This script:
  1. Picks one representative prompt per (domain, complexity) tier
  2. Calls POST /chat with type=validation (no recommendations → scores the prompt)
  3. Prints actual scores vs current expected ranges
  4. Re-run with --write to patch prompts.json with suggested ranges (±MARGIN)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.config import BASE_URL, GOLDEN_SET_PATH, MODEL_NAME
from eval.pipeline.runner import load_golden_set

_SAMPLE_IDS = [
    # simple
    "cs_simple_01", "code_simple_01",
    # medium
    "cs_medium_01", "code_medium_01", "data_medium_01", "class_medium_01",
    # complex
    "cs_complex_01", "reason_complex_01",
]

_MARGIN = 15
_CONCURRENCY = 5


async def score_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str | None,
    prompt_entry: dict,
) -> dict | None:
    payload = {
        "sessionId": f"calib_{prompt_entry['id']}",
        "type": "validation",
        "systemPrompt": prompt_entry["system_prompt"],
        "recommendations": None,
    }
    if model_name:
        payload["modelName"] = model_name

    try:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        er = data.get("content", {}).get("evalResult") or {}
        if not er:
            print(f"  [WARN] {prompt_entry['id']}: evalResult is null — status={data.get('statusCode')}")
            return None
        cs = er.get("criteriaScores") or er.get("criteria_scores") or {}
        return {
            "id": prompt_entry["id"],
            "domain": prompt_entry["domain"],
            "complexity": prompt_entry["complexity"],
            "actual": {
                "total": er.get("totalScore") or er.get("total_score"),
                "clarity_and_specificity": cs.get("clarityAndSpecificity") or cs.get("clarity_and_specificity"),
                "structure_and_organization": cs.get("structureAndOrganization") or cs.get("structure_and_organization"),
                "output_specification": cs.get("outputSpecification") or cs.get("output_specification"),
                "contextual_guidance": cs.get("contextualGuidance") or cs.get("contextual_guidance"),
                "error_handling": cs.get("errorHandling") or cs.get("error_handling"),
            },
            "current_ranges": prompt_entry.get("expected_score_ranges", {}),
        }
    except Exception as exc:
        print(f"  [ERROR] {prompt_entry['id']}: {exc}")
        return None


def _suggest_ranges(actual: dict, margin: int) -> dict:
    ranges = {}
    criteria = [
        "clarity_and_specificity",
        "structure_and_organization",
        "output_specification",
        "contextual_guidance",
        "error_handling",
    ]
    for c in criteria:
        val = actual.get(c)
        if val is not None:
            ranges[c] = {"lo": max(0, val - margin), "hi": min(100, val + margin)}
    total = actual.get("total")
    if total is not None:
        ranges["total"] = {"lo": max(0, total - margin), "hi": min(100, total + margin)}
    return ranges


async def _main(args: argparse.Namespace) -> None:
    golden_set = load_golden_set(args.golden_set)

    if args.all:
        sample_entries = golden_set
    else:
        sample_entries = [p for p in golden_set if p["id"] in _SAMPLE_IDS]

    print(f"\nCalibrating {len(sample_entries)} prompts against {args.base_url}")
    print(f"Model: {args.model or 'server default'}\n")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def _score_with_sem(client, entry):
        async with semaphore:
            return await score_prompt(client, args.base_url, args.model, entry)

    async with httpx.AsyncClient() as client:
        tasks = [_score_with_sem(client, entry) for entry in sample_entries]
        raw = await asyncio.gather(*tasks)
    results = [r for r in raw if r is not None]

    if not results:
        print("No results returned. Check that the backend is reachable.")
        return

    print(f"{'ID':<25} {'COMPLEXITY':<10} {'TOTAL':>6} {'CLARITY':>8} {'STRUCT':>7} {'OUTPUT':>7} {'CONTEXT':>8} {'ERROR':>6}")
    print("-" * 82)

    suggested_updates: dict[str, dict] = {}
    for r in results:
        a = r["actual"]
        cur = r["current_ranges"]
        total_ok = (
            cur.get("total", {}).get("lo", 0) <= (a["total"] or 0) <= cur.get("total", {}).get("hi", 100)
            if cur else True
        )
        flag = "" if total_ok else " ← OUT"
        def _fmt(v):
            return str(v) if v is not None else "?"

        print(
            f"{r['id']:<25} {r['complexity']:<10} "
            f"{_fmt(a['total']):>6} {_fmt(a['clarity_and_specificity']):>8} "
            f"{_fmt(a['structure_and_organization']):>7} {_fmt(a['output_specification']):>7} "
            f"{_fmt(a['contextual_guidance']):>8} {_fmt(a['error_handling']):>6}{flag}"
        )
        suggested_updates[r["id"]] = _suggest_ranges(a, args.margin)

    print(f"\nSuggested range updates (±{args.margin} around actual score):")
    print(json.dumps(suggested_updates, indent=2))

    if args.write:
        patched = 0
        for p in golden_set:
            if p["id"] in suggested_updates:
                p["expected_score_ranges"] = suggested_updates[p["id"]]
                patched += 1
        with open(args.golden_set, "w", encoding="utf-8") as f:
            json.dump(golden_set, f, indent=2)
        print(f"\nPatched {patched} prompts in {args.golden_set}")
    else:
        print("\nRe-run with --write to apply these ranges to prompts.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate golden set score ranges against the live backend.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--golden-set", default=GOLDEN_SET_PATH)
    parser.add_argument("--margin", type=int, default=_MARGIN,
                        help="Points above/below actual score for lo/hi bounds (default 15)")
    parser.add_argument("--concurrency", type=int, default=_CONCURRENCY,
                        help="Max parallel requests (default 5)")
    parser.add_argument("--all", action="store_true",
                        help="Calibrate all 50 prompts instead of the default 8 sample IDs")
    parser.add_argument("--write", action="store_true",
                        help="Write suggested ranges back to prompts.json")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
