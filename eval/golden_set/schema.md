# Golden Set Schema Reference

Each entry in `prompts.json` has the following fields.

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique identifier, e.g. `cs_simple_01` |
| `domain` | string | One of 10 domains (see below) |
| `complexity` | string | `simple`, `medium`, or `complex` |
| `intentional_weaknesses` | list[string] | Weakness tags this prompt is designed to exhibit |
| `system_prompt` | string | The actual prompt text being evaluated |
| `expected_score_ranges` | object | Per-criterion and total `{lo, hi}` bounds (0–100 each) |
| `expected_recommendation_keywords` | list[string] | PE concepts that should appear in recommendations |
| `test_inputs` | list[{input: str}] | 2 representative inputs for the LLM judge step |
| `tags` | list[string] | Domain, complexity, and weakness tags for filtering |
| `notes` | string | Calibration notes explaining the score range choices |

## Domains (10)

`customer_service`, `coding_assistant`, `data_analysis`, `content_writing`,
`classification`, `information_extraction`, `rag_qa`, `reasoning`,
`summarization`, `multi_turn_conversation`

## Complexity Distribution

Each domain has 5 prompts: 2 simple, 2 medium, 1 complex.

## Weakness Catalogue

| Tag | Scoring Impact |
|---|---|
| `no_role_defined` | `clarity_and_specificity` ≤ 60 |
| `no_output_format` | `output_specification` ≤ 60 |
| `no_error_handling` | `error_handling` ≤ 50 |
| `no_context` | `contextual_guidance` ≤ 60 (when combined with no_examples_when_needed) |
| `no_examples_when_needed` | `contextual_guidance` ≤ 60 (when combined with no_context) |
| `no_structure` | `structure_and_organization` ≤ 70 |
| `vague_instructions` | `clarity_and_specificity` naturally low |
| `ambiguous_constraints` | mixed impact on clarity and error_handling |

**Short prompt rule**: fewer than 30 meaningful words → `clarity_and_specificity` ≤ 55

## Expected Recommendation Keywords (PE-Grounded)

Keywords should reflect prompt engineering best practices:

- **Structural**: `role`, `persona`, `output format`, `schema`, `JSON`, `XML tags`, `headers`
- **PE techniques**: `chain-of-thought`, `few-shot`, `example`, `negative constraints`
- **Error/edge cases**: `error handling`, `fallback`, `edge case`, `ambiguous`, `unsolvable`, `conflicting`
- **Domain-specific**: `citation`, `grounded`, `confidence`, `label set`, `word count`, `audience`

## Score Range Design Rules

Upper bounds must never exceed the applicable ceiling:
- `output_specification` hi ≤ 60 when `no_output_format` is present
- `error_handling` hi ≤ 50 when `no_error_handling` is present
- `clarity_and_specificity` hi ≤ 60 when `no_role_defined` is present
- `clarity_and_specificity` hi ≤ 55 when prompt has < 30 words
- `structure_and_organization` hi ≤ 70 when `no_structure` is present
- `contextual_guidance` hi ≤ 60 when both `no_context` and `no_examples_when_needed` are present
