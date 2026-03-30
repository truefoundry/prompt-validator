"""CLI entry point for the eval framework.

Usage:
    CONFIGBASEPATH=./config python -m eval.run_eval
    python -m eval.run_eval --no-analysis --concurrency 3
    python -m eval.run_eval --base-url http://localhost:21120 --model openai-main/gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path so src/ imports work in analyser.py
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.config import (
    BASE_URL,
    CONCURRENCY_LIMIT,
    GOLDEN_SET_PATH,
    MODEL_NAME,
    REPORTS_DIR,
)
from eval.pipeline.analyser import analyse_report
from eval.pipeline.report import build_report
from eval.pipeline.runner import load_golden_set, run_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the prompt-validator eval pipeline against the golden set."
    )
    parser.add_argument(
        "--golden-set", default=GOLDEN_SET_PATH, metavar="PATH",
        help="Path to prompts.json (default: eval/golden_set/prompts.json)",
    )
    parser.add_argument(
        "--base-url", default=BASE_URL,
        help="Backend base URL (default: $EVAL_BASE_URL or http://localhost:21120)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="Model name override (default: server default)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY_LIMIT,
        help="Max parallel prompts (default: $EVAL_CONCURRENCY or 5)",
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="Skip the LLM meta-analysis step",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Override the auto-generated run ID",
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    run_id = args.run_id or uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    intermediate_path = os.path.join(REPORTS_DIR, f"intermediate_{run_id}.ndjson")

    logger.info(
        "Starting eval run=%s | base_url=%s | model=%s | concurrency=%d",
        run_id, args.base_url, args.model or "server-default", args.concurrency,
    )

    golden_set = load_golden_set(args.golden_set)
    logger.info("Loaded %d prompts from %s", len(golden_set), args.golden_set)

    results = await run_all(
        golden_set=golden_set,
        base_url=args.base_url,
        model_name=args.model,
        concurrency=args.concurrency,
        intermediate_path=intermediate_path,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    run_metadata = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "model_name": args.model or "server-default",
        "base_url": args.base_url,
    }

    report = build_report(results, run_metadata)

    report_path = os.path.join(REPORTS_DIR, f"report_{run_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written → %s", report_path)

    s = report["summary"]
    logger.info(
        "SUMMARY | score_bound_pass_rate=%.1f%% | keyword_coverage_avg=%.2f | "
        "rec_quality_avg=%s | application_quality_avg=%s | "
        "avg_improvement_delta=%s | regressions=%d | failed_prompts=%d",
        (s["score_bound_pass_rate"] or 0) * 100,
        s["keyword_coverage_avg"] or 0,
        f"{s['rec_quality_avg']:.2f}" if s.get("rec_quality_avg") is not None else "n/a",
        f"{s['application_quality_avg']:.2f}" if s.get("application_quality_avg") is not None else "n/a",
        s["avg_improvement_delta"],
        s["regression_count"],
        report["run_metadata"]["failed_prompts"],
    )

    if not args.no_analysis:
        logger.info("Running LLM meta-analysis…")
        analysis = await analyse_report(report, model_name=args.model)
        analysis_path = os.path.join(REPORTS_DIR, f"report_{run_id}.analysis.md")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(analysis)
        logger.info("Analysis written → %s", analysis_path)

    # Remove intermediate file only on full success
    try:
        os.remove(intermediate_path)
    except OSError:
        pass


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
