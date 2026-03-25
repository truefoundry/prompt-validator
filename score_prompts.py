import json
import sys
import time
import httpx

API_URL = "http://localhost:21120/chat"
INPUT_FILE = "floqast-prompt-evaluation-template.json"
OUTPUT_FILE = "floqast-prompt-evaluation-template.json"


def parse_prompt_parts(prompt_text: str) -> tuple[str, str | None]:
    """Split a combined prompt string into system and user parts."""
    if "\n\nuser:" in prompt_text:
        idx = prompt_text.index("\n\nuser:")
        system_part = prompt_text[:idx]
        user_part = prompt_text[idx + len("\n\nuser:"):].strip()
    elif "\nuser:" in prompt_text:
        idx = prompt_text.index("\nuser:")
        system_part = prompt_text[:idx]
        user_part = prompt_text[idx + len("\nuser:"):].strip()
    else:
        system_part = prompt_text
        user_part = None

    if system_part.startswith("system:"):
        system_part = system_part[len("system:"):].strip()
    elif system_part.startswith("system: "):
        system_part = system_part[len("system: "):].strip()

    return system_part, user_part


def score_prompt(system_prompt: str, user_prompt: str | None, session_suffix: str) -> int | None:
    """Call the API to score a prompt, return total_score or None on failure."""
    payload = {
        "sessionId": f"scoring-{session_suffix}-{int(time.time())}",
        "systemPrompt": system_prompt,
        "type": "validation",
        "recommendations": None,
    }
    if user_prompt:
        payload["userPromptTemplate"] = user_prompt

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        eval_result = data.get("content", {}).get("eval_result") or data.get("content", {}).get("evalResult")
        if eval_result and "total_score" in eval_result:
            return eval_result["total_score"]

        if eval_result and "totalScore" in eval_result:
            return eval_result["totalScore"]

        print(f"  [WARN] Could not extract total_score from response: {json.dumps(data, indent=2)[:500]}")
        return None
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}")
        return None


def main():
    with open(INPUT_FILE, "r") as f:
        template = json.load(f)

    evaluations = template["evaluations"]
    total = len(evaluations)

    for i, ev in enumerate(evaluations):
        name = ev["name"]
        print(f"\n[{i+1}/{total}] Processing: {name}")

        original = ev.get("original_prompt", "")
        enhanced = ev.get("enhanced_prompt", "")

        if not original:
            print("  Skipping - no original_prompt")
            continue
        if not enhanced:
            print("  Skipping - no enhanced_prompt")
            continue

        sys_orig, user_orig = parse_prompt_parts(original)
        print(f"  Scoring original_prompt...")
        orig_score = score_prompt(sys_orig, user_orig, f"{name}-orig")
        print(f"  Original score: {orig_score}")

        sys_enh, user_enh = parse_prompt_parts(enhanced)
        print(f"  Scoring enhanced_prompt...")
        enh_score = score_prompt(sys_enh, user_enh, f"{name}-enh")
        print(f"  Enhanced score: {enh_score}")

        if orig_score is not None and enh_score is not None:
            direction = "increased" if enh_score > orig_score else "decreased" if enh_score < orig_score else "unchanged at"
            if direction == "unchanged at":
                ev["notes"] = f"{orig_score} unchanged at {enh_score}"
            else:
                ev["notes"] = f"{orig_score} {direction} to {enh_score}"
            print(f"  Notes: {ev['notes']}")
        else:
            ev["notes"] = f"original: {orig_score}, enhanced: {enh_score}"
            print(f"  Notes (partial): {ev['notes']}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(template, f, indent=2)

    print(f"\nDone! Updated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
