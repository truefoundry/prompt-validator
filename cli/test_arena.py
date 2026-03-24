"""CLI test for Arena Evaluation (ArenaGEval)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ORIGINAL_PROMPT = """You are a customer support agent for a pharmacy.
Answer customer questions about prescriptions and medications."""

ENHANCED_PROMPT = """You are a knowledgeable and empathetic customer support agent for a pharmacy.
When answering questions:
- Always acknowledge the customer's concern first
- Provide clear, actionable information about prescriptions and medications
- If you cannot help directly, guide them to the right resource (pharmacist, doctor, or store)
- Keep responses concise but complete"""

TEST_CASES = [
    {
        "data": {"input": "How do I check the status of my prescription refill?"},
        "expected_output": "",
    },
    {
        "data": {"input": "Can I take ibuprofen with my blood pressure medication?"},
        "expected_output": "",
    },
    {
        "data": {"input": "I need my prescription ready urgently, what can I do?"},
        "expected_output": "",
    },
]

CRITERIA = "Choose the response that is more empathetic, complete, and actionable for the customer."

if __name__ == "__main__":
    from src.chat.graph.arena_evaluator import run_arena_comparison

    print(f"Running Arena comparison on {len(TEST_CASES)} test cases...")
    print(f"Criteria: {CRITERIA}\n")

    result = run_arena_comparison(
        original_prompt=ORIGINAL_PROMPT,
        enhanced_prompt=ENHANCED_PROMPT,
        test_cases=TEST_CASES,
        criteria=CRITERIA,
    )

    wins = result["wins"]
    win_rate = result["win_rate"]
    per_test = result["per_test"]

    print(f"=== Results ===")
    print(f"Original Wins : {wins.get('Original', 0)}")
    print(f"Enhanced Wins : {wins.get('Enhanced', 0)}")
    print(f"Enhanced Win Rate: {win_rate:.0%}\n")

    for i, t in enumerate(per_test):
        print(f"--- Test {i+1} ---")
        print(f"Input   : {t['input'][:80]}")
        print(f"Winner  : {t['winner']}")
        print(f"Reason  : {t['reason'][:200]}")
        print(f"Original: {t['original_output'][:100]}")
        print(f"Enhanced: {t['enhanced_output'][:100]}")
        print()
