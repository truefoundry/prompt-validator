"""Validation functions for eval pipeline results."""

from __future__ import annotations


def validate_score_bounds(eval_result: dict, expected_ranges: dict) -> dict:
    """Check that each criterion score falls within expected [lo, hi] bounds.

    Args:
        eval_result: The evalResult dict from the API response, containing
                     total_score and criteria_scores.
        expected_ranges: Dict with criteria names and "total" as keys,
                         each mapping to {"lo": int, "hi": int}.

    Returns:
        {
          "per_criterion": {name: {"actual": int, "lo": int, "hi": int, "pass": bool}},
          "total": {"actual": int, "lo": int, "hi": int, "pass": bool},
          "all_pass": bool
        }
    """
    criteria = eval_result.get("criteria_scores") or {}
    total_actual = eval_result.get("total_score", 0)

    per_criterion: dict[str, dict] = {}
    all_pass = True

    for name, bounds in expected_ranges.items():
        if name == "total":
            continue
        lo, hi = bounds["lo"], bounds["hi"]
        actual = criteria.get(name, 0)
        passed = lo <= actual <= hi
        if not passed:
            all_pass = False
        per_criterion[name] = {"actual": actual, "lo": lo, "hi": hi, "pass": passed}

    total_bounds = expected_ranges.get("total", {})
    total_lo = total_bounds.get("lo", 0)
    total_hi = total_bounds.get("hi", 100)
    total_passed = total_lo <= total_actual <= total_hi
    if not total_passed:
        all_pass = False

    return {
        "per_criterion": per_criterion,
        "total": {"actual": total_actual, "lo": total_lo, "hi": total_hi, "pass": total_passed},
        "all_pass": all_pass,
    }


def validate_keyword_coverage(
    recommendations: list[str],
    expected_keywords: list[str],
    min_coverage: float = 0.6,
) -> dict:
    """Check that expected keywords appear (case-insensitive substring) in the recommendations.

    Args:
        recommendations: The list of recommendation strings from the API.
        expected_keywords: Keywords that should appear somewhere in the recommendations.
        min_coverage: Minimum fraction of keywords required (default 0.6).

    Returns:
        {"coverage": float, "matched": list[str], "missed": list[str], "pass": bool}
    """
    if not expected_keywords:
        return {"coverage": 1.0, "matched": [], "missed": [], "pass": True}

    joined = " ".join(recommendations).lower()
    matched = [kw for kw in expected_keywords if kw.lower() in joined]
    missed = [kw for kw in expected_keywords if kw.lower() not in joined]
    coverage = len(matched) / len(expected_keywords)

    return {
        "coverage": round(coverage, 3),
        "matched": matched,
        "missed": missed,
        "pass": coverage >= min_coverage,
    }


def validate_rec_quality(rec_quality_result: dict, min_score: float = 0.6) -> dict:
    """Summarise the rec_quality_judge result into a pass/fail signal.

    Args:
        rec_quality_result: Output of judges.evaluate_rec_quality.
        min_score: Minimum overall_score required to pass (default 0.6).

    Returns:
        {"score": float | None, "pass": bool, "verdict": str | None,
         "missed_weaknesses": list[str], "issues": list[str]}
    """
    score = rec_quality_result.get("overall_score")
    return {
        "score": score,
        "pass": score is not None and score >= min_score,
        "verdict": rec_quality_result.get("verdict"),
        "missed_weaknesses": rec_quality_result.get("missed_weaknesses", []),
        "issues": rec_quality_result.get("issues", []),
    }


def validate_application_quality(app_quality_result: dict, min_score: float = 0.6) -> dict:
    """Summarise the application_quality_judge result into a pass/fail signal.

    Args:
        app_quality_result: Output of judges.evaluate_application_quality.
        min_score: Minimum overall_score required to pass (default 0.6).

    Returns:
        {"score": float | None, "pass": bool, "verdict": str | None,
         "regressions": list[str], "new_issues": list[str],
         "unapplied_recs": list[str]}
    """
    score = app_quality_result.get("overall_score")
    unapplied = [
        r["recommendation"]
        for r in app_quality_result.get("per_rec", [])
        if not r.get("applied", True)
    ]
    return {
        "score": score,
        "pass": score is not None and score >= min_score,
        "verdict": app_quality_result.get("verdict"),
        "regressions": app_quality_result.get("regressions", []),
        "new_issues": app_quality_result.get("new_issues", []),
        "unapplied_recs": unapplied,
    }


def validate_improvement(judge_result: dict) -> dict:
    """Extract improvement signal from the LLM judge report.

    Args:
        judge_result: The testEvaluationResult dict from the API response
                      (output of LLMJudgeEvaluator.get_final_report).

    Returns:
        {
          "improvement_delta": float | None,
          "improved_count": int,
          "total_cases": int,
          "regression": bool,
          "neutral": bool,
        }
    """
    _REGRESSION_THRESHOLD = -0.05
    _NEUTRAL_THRESHOLD = 0.05

    summary = judge_result.get("summary", {})
    avg_delta = summary.get("avg_delta", {})
    overall_delta = avg_delta.get("overall")
    improved_count = summary.get("improved_count", 0)
    total_cases = summary.get("total_cases", 0)

    regression = overall_delta is not None and overall_delta < _REGRESSION_THRESHOLD
    neutral = overall_delta is not None and abs(overall_delta) <= _NEUTRAL_THRESHOLD

    return {
        "improvement_delta": overall_delta,
        "improved_count": improved_count,
        "total_cases": total_cases,
        "regression": regression,
        "neutral": neutral,
    }
