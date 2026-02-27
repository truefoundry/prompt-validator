from typing import Any


def extract_original_prompt(response_data: dict[str, Any]) -> str:
    direct_original = response_data.get("original_prompt")
    if isinstance(direct_original, str):
        return direct_original

    content = response_data.get("content", {})
    if isinstance(content, dict):
        content_original = content.get("original_prompt")
        if isinstance(content_original, str):
            return content_original

        eval_result = content.get("eval_result") or content.get("evalResult") or {}
        if isinstance(eval_result, dict):
            eval_original = eval_result.get("original_prompt") or eval_result.get("originalPrompt")
            if isinstance(eval_original, str):
                return eval_original

    return ""


def extract_recommendations(response_data: dict[str, Any]) -> list[str]:
    direct_recommendations = response_data.get("recommendations")
    if isinstance(direct_recommendations, list):
        return [str(item).strip() for item in direct_recommendations if str(item).strip()]

    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return []

    eval_result = content.get("eval_result") or content.get("evalResult") or {}
    if not isinstance(eval_result, dict):
        return []

    nested_recommendations = eval_result.get("recommendations")
    if not isinstance(nested_recommendations, list):
        return []
    return [str(item).strip() for item in nested_recommendations if str(item).strip()]


def extract_enhanced_prompt(response_data: dict[str, Any]) -> str:
    direct = response_data.get("enhanced_prompt")
    if isinstance(direct, str):
        return direct

    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return ""

    final_prompt_result = content.get("final_prompt_result") or content.get("finalPromptResult") or {}
    if isinstance(final_prompt_result, dict):
        prompt_messages = (
            final_prompt_result.get("prompt_messages_list")
            or final_prompt_result.get("promptMessagesList")
            or []
        )
        if isinstance(prompt_messages, list):
            lines: list[str] = []
            for item in prompt_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip()
                message = str(item.get("content", "")).strip()
                if message:
                    lines.append(f"{role}: {message}" if role else message)
            return "\n\n".join(lines)
    return ""


def extract_total_score(response_data: dict[str, Any]) -> int | None:
    direct = response_data.get("total_score")
    if isinstance(direct, int):
        return direct

    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return None

    eval_result = content.get("eval_result") or content.get("evalResult") or {}
    if not isinstance(eval_result, dict):
        return None

    nested = eval_result.get("total_score") or eval_result.get("totalScore")
    return nested if isinstance(nested, int) else None


def extract_criteria_scores(response_data: dict[str, Any]) -> dict[str, Any]:
    direct = response_data.get("criteria_scores")
    if isinstance(direct, dict):
        return direct

    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return {}

    eval_result = content.get("eval_result") or content.get("evalResult") or {}
    if not isinstance(eval_result, dict):
        return {}

    nested = eval_result.get("criteria_scores") or eval_result.get("criteriaScores")
    return nested if isinstance(nested, dict) else {}


def extract_test_evaluation_result(response_data: dict[str, Any]) -> dict | None:
    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return None
    result = content.get("test_evaluation_result") or content.get("testEvaluationResult")
    return result if isinstance(result, dict) else None


def extract_explanations(response_data: dict[str, Any]) -> dict[str, str]:
    direct = response_data.get("explanations")
    if isinstance(direct, dict):
        return {str(k): str(v) for k, v in direct.items()}

    content = response_data.get("content", {})
    if not isinstance(content, dict):
        return {}

    eval_result = content.get("eval_result") or content.get("evalResult") or {}
    if not isinstance(eval_result, dict):
        return {}

    nested = eval_result.get("explanations")
    if isinstance(nested, dict):
        return {str(k): str(v) for k, v in nested.items()}
    return {}
