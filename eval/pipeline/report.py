"""Aggregate per-prompt results into a final evaluation report."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_report(results: list[dict], run_metadata: dict) -> dict:
    """Build the final evaluation report from per-prompt results."""
    total = len(results)
    failed = sum(1 for r in results if r["get_recommendation"]["status"] == "error")

    # ── Top-level aggregates ──────────────────────────────────────────────────
    score_bound_passes = [
        r["validation"]["score_bounds_pass"]
        for r in results
        if r.get("validation")
    ]
    keyword_coverages = [
        r["validation"]["keyword_coverage"]
        for r in results
        if r.get("validation")
    ]
    rec_quality_scores = [
        r["validation"]["rec_quality_score"]
        for r in results
        if r.get("validation") and r["validation"].get("rec_quality_score") is not None
    ]
    app_quality_scores = [
        r["validation"]["application_quality_score"]
        for r in results
        if r.get("validation") and r["validation"].get("application_quality_score") is not None
    ]
    improvement_deltas = [
        r["validation"]["improvement_delta"]
        for r in results
        if r.get("validation") and r["validation"].get("improvement_delta") is not None
    ]
    regression_count = sum(1 for r in results if r.get("validation", {}).get("regression"))

    score_bound_pass_rate = _safe_mean([1.0 if p else 0.0 for p in score_bound_passes])
    keyword_coverage_avg = _safe_mean(keyword_coverages)
    rec_quality_avg = _safe_mean(rec_quality_scores)
    application_quality_avg = _safe_mean(app_quality_scores)
    avg_improvement_delta = _safe_mean(improvement_deltas)

    # ── Domain breakdown ──────────────────────────────────────────────────────
    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        domain_groups[r["domain"]].append(r)
    domain_breakdown = {
        domain: _compute_group_stats(group)
        for domain, group in domain_groups.items()
    }

    # ── Weakness type breakdown ───────────────────────────────────────────────
    weakness_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for w in r.get("intentional_weaknesses", []):
            weakness_groups[w].append(r)
    weakness_type_breakdown = {
        w: _compute_group_stats(group)
        for w, group in weakness_groups.items()
    }

    # ── Complexity breakdown ──────────────────────────────────────────────────
    complexity_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        complexity_groups[r["complexity"]].append(r)
    complexity_breakdown = {
        level: _compute_group_stats(group)
        for level, group in complexity_groups.items()
    }

    return {
        "run_metadata": {**run_metadata, "total_prompts": total, "failed_prompts": failed},
        "summary": {
            "score_bound_pass_rate": score_bound_pass_rate,
            "keyword_coverage_avg": keyword_coverage_avg,
            "rec_quality_avg": rec_quality_avg,
            "application_quality_avg": application_quality_avg,
            "avg_improvement_delta": avg_improvement_delta,
            "regression_count": regression_count,
            "domain_breakdown": domain_breakdown,
            "weakness_type_breakdown": weakness_type_breakdown,
            "complexity_breakdown": complexity_breakdown,
        },
        "per_prompt_results": results,
        "calibration_analysis": _compute_calibration(results),
    }


def _safe_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _compute_group_stats(group: list[dict]) -> dict[str, Any]:
    if not group:
        return {}
    v = [r["validation"] for r in group if r.get("validation")]
    passes = [x["score_bounds_pass"] for x in v]
    coverages = [x["keyword_coverage"] for x in v]
    rec_scores = [x["rec_quality_score"] for x in v if x.get("rec_quality_score") is not None]
    app_scores = [x["application_quality_score"] for x in v if x.get("application_quality_score") is not None]
    deltas = [x["improvement_delta"] for x in v if x.get("improvement_delta") is not None]
    return {
        "count": len(group),
        "score_bound_pass_rate": _safe_mean([1.0 if p else 0.0 for p in passes]),
        "keyword_coverage_avg": _safe_mean(coverages),
        "rec_quality_avg": _safe_mean(rec_scores),
        "application_quality_avg": _safe_mean(app_scores),
        "avg_improvement_delta": _safe_mean(deltas),
    }


def _compute_calibration(results: list[dict]) -> dict:
    """Compute calibration statistics across all prompts."""
    criteria = [
        "clarity_and_specificity",
        "structure_and_organization",
        "output_specification",
        "contextual_guidance",
        "error_handling",
    ]

    # Average actual scores per criterion
    criteria_scores: dict[str, list[int]] = {c: [] for c in criteria}
    for r in results:
        cs = (r.get("get_recommendation", {}).get("eval_result") or {}).get("criteria_scores") or {}
        for c in criteria:
            if c in cs:
                criteria_scores[c].append(cs[c])

    criteria_avg_actual = {
        c: round(sum(v) / len(v), 1) if v else None
        for c, v in criteria_scores.items()
    }

    # Delta between actual score and expected midpoint per criterion
    delta_by_criterion: dict[str, list[float]] = {c: [] for c in criteria}
    for r in results:
        expected = r.get("expected_score_ranges") or {}
        cs = (r.get("get_recommendation", {}).get("eval_result") or {}).get("criteria_scores") or {}
        for c in criteria:
            if c in expected and c in cs:
                lo, hi = expected[c]["lo"], expected[c]["hi"]
                mid = (lo + hi) / 2
                delta_by_criterion[c].append(cs[c] - mid)

    criteria_delta_vs_mid = {
        c: round(sum(v) / len(v), 1) if v else None
        for c, v in delta_by_criterion.items()
    }

    # Out-of-bounds counts per criterion
    oob_counts: dict[str, int] = {c: 0 for c in criteria}
    for r in results:
        per = r.get("validation", {}).get("score_bounds", {}).get("per_criterion", {})
        for c, detail in per.items():
            if not detail.get("pass"):
                oob_counts[c] = oob_counts.get(c, 0) + 1

    most_oob = max(oob_counts, key=lambda k: oob_counts[k]) if any(oob_counts.values()) else None

    # Most missed recommendation keywords
    missed_all: list[str] = []
    for r in results:
        missed_all.extend(r.get("validation", {}).get("keyword_details", {}).get("missed", []))

    most_missed = [kw for kw, _ in Counter(missed_all).most_common(10)]

    return {
        "criteria_avg_actual_scores": criteria_avg_actual,
        "criteria_score_delta_vs_expected_midpoint": criteria_delta_vs_mid,
        "most_often_out_of_bounds_criterion": most_oob,
        "out_of_bounds_counts": oob_counts,
        "most_missed_keywords": most_missed,
    }
