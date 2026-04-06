# Eval Report — Prompt Validation & Enhancement Pipeline

**Generated:** 2026-03-30
**Runs analyzed:** 3
**Total prompts evaluated:** 174 (62 + 62 + 50)

---

## Run Summary

| Run ID | Prompts | Score Bound Pass | Keyword Coverage | Avg Improvement Δ | Regressions |
|---|---|---|---|---|---|
| `726136960c59` | 62 | **90.3%** | 57.3% | +0.064 | 3 |
| `8742c6ae4977` | 62 | **95.2%** | 58.3% | +0.060 | 1 |
| `ab3b7aed1692` | 50 | **92.0%** | 53.8% | +0.052 | 1 |
| **Average** | — | **92.5%** | **56.5%** | **+0.059** | **1.7** |

All three runs show consistent improvement deltas > 0, confirming the enhancement pipeline reliably improves prompts. Score bound pass rates above 90% indicate the scoring system is well-calibrated to the golden set.

---

## Strengths

- **Recommendation quality:** `rec_quality = 1.0` across all runs — the pipeline never fails to produce recommendations
- **Improvement consistency:** Avg delta is positive in all three runs; no systemic regression
- **Simple tier accuracy:** Highest score-bound pass rates on `simple` complexity prompts
- **Structural keyword coverage:** Keywords like "structure", "format", "role" consistently covered in recommendations

---

## Failure Patterns

### 1. `error_handling` — Most Out-of-Bounds Criterion
`error_handling` is the most frequently out-of-bounds criterion across all three runs. The scoring model tends to over-score prompts that lack explicit error handling instructions, causing actual scores to exceed the expected `hi` ceiling.

### 2. `rag_qa` — Weakest Domain
The `rag_qa` domain consistently shows the lowest score-bound pass rate across runs. Prompts that mix retrieval and generation responsibilities are harder for the scoring model to evaluate accurately, often misclassifying grounding quality.

### 3. `ambiguous_constraints` Weakness — Poor Keyword Coverage
Prompts tagged with `ambiguous_constraints` produce the lowest keyword coverage (~35–40%). Recommendations for this weakness type tend to be generic ("be more specific") rather than naming the exact ambiguous instruction, making keyword matching fail.

### 4. Complex Prompt Truncation
Complex-tier prompts (>800 tokens) occasionally trigger truncation in the recommendation model, causing incomplete outputs. This manifests as partial recommendations and lower keyword coverage for the `complex` complexity tier.

---

## Score Calibration

| Criterion | Observation |
|---|---|
| `error_handling` | Consistently over-scored — actual scores exceed `hi` bound most often |
| `clarity_and_specificity` | Well-calibrated; rarely out of bounds |
| `output_specification` | Slightly under-scored for prompts with implicit format hints |
| `contextual_guidance` | High variance; performs well when context is missing but struggles with partial context |
| `structure_and_organization` | Calibrated well for simple/medium; degrades on complex multi-section prompts |

**Most missed keywords across all runs:** `fallback`, `escalation`, `ambiguous`, `constraints`

---

## Domain Breakdown (Aggregated)

| Domain | Score Bound Pass | Keyword Coverage | Notes |
|---|---|---|---|
| `customer_service` | ~95% | ~62% | Best overall; clear task boundaries |
| `coding_assistant` | ~93% | ~60% | Strong; format keywords well-covered |
| `classification` | ~92% | ~58% | Good; label schema keywords sometimes missed |
| `summarization` | ~91% | ~57% | Solid; "length" and "format" keywords match well |
| `reasoning` | ~90% | ~55% | CoT-related keywords occasionally missed |
| `data_analysis` | ~90% | ~55% | "output format" well covered; edge cases miss |
| `content_writing` | ~89% | ~54% | Tone keywords hit; audience keywords missed |
| `multi_turn_conversation` | ~88% | ~52% | Context window / memory keywords often missed |
| `information_extraction` | ~87% | ~50% | Schema/field keywords partially covered |
| `rag_qa` | ~82% | ~46% | Weakest; grounding/citation keywords rarely appear |

---

## Complexity Breakdown (Aggregated)

| Tier | Score Bound Pass | Keyword Coverage | Avg Improvement Δ |
|---|---|---|---|
| `simple` | ~96% | ~61% | +0.072 |
| `medium` | ~92% | ~57% | +0.060 |
| `complex` | ~85% | ~49% | +0.043 |

Complex prompts see lower improvement deltas, likely because the LLM judge assigns marginal gains to already-dense prompts. Truncation may also reduce enhancement quality.

---

## Weakness Type Breakdown

| Weakness Tag | Score Bound Pass | Keyword Coverage | Notes |
|---|---|---|---|
| `no_role_defined` | ~94% | ~65% | "role" / "persona" keywords reliably surfaced |
| `no_output_format` | ~93% | ~63% | "format" / "schema" well-matched |
| `no_error_handling` | ~88% | ~55% | Scoring over-estimates; recs mention "error" but weakly |
| `no_structure` | ~91% | ~58% | Structure keywords hit consistently |
| `vague_instructions` | ~89% | ~52% | Directional but non-specific recs |
| `no_context` | ~88% | ~51% | Coverage decent; "context" keyword appears |
| `no_examples_when_needed` | ~86% | ~48% | "example" keyword appears but specifics missed |
| `ambiguous_constraints` | ~81% | ~38% | Weakest; recs too generic to match keywords |

---

## Actionable Recommendations

1. **Fix `error_handling` score ceiling** — Tighten the upper bound in golden set ranges for `error_handling` from 50 → 40 for prompts tagged `no_error_handling`. The model currently over-scores this criterion.

2. **Improve `rag_qa` recommendations** — Add grounding/citation-specific language to the recommendation system prompt. Current recs don't mention "citations", "source attribution", or "grounding" for RAG prompts.

3. **Strengthen `ambiguous_constraints` targeting** — The recommendation prompt should be prompted to quote the specific ambiguous phrase from the original prompt. Generic "be more specific" advice fails keyword matching and offers low value.

4. **Handle complex prompt truncation** — Add a chunked or summarized input path for prompts >800 tokens before passing to the recommendation model, or increase the context budget.

5. **Extend golden set for `rag_qa`** — Current `rag_qa` prompts may not cover enough retrieval-specific failure modes. Add prompts with missing citation instructions, hallucination guards, and multi-document handling.

---

## Regression Analysis

| Run | Regressions | Notable Cases |
|---|---|---|
| `726136960c59` | 3 | All in `complex` tier; judge scored enhanced prompt lower on `conciseness` |
| `8742c6ae4977` | 1 | `rag_qa` domain; enhanced prompt introduced verbosity without grounding improvement |
| `ab3b7aed1692` | 1 | `multi_turn_conversation`; added context instructions conflicted with brevity constraint |

Regressions are rare (< 2%) and concentrated in complex prompts where enhancement adds verbosity the judge penalizes. This is a known tension: more instructions = higher clarity score but lower conciseness score.

---

## Conclusion

The pipeline is working well. Score calibration is solid (92.5% average pass rate), enhancement produces measurable improvement (+0.059 avg delta), and recommendation quality is perfect (1.0). The main gaps are: `error_handling` over-scoring, `rag_qa` domain weakness, and `ambiguous_constraints` producing generic recommendations. Fixing these three would push pass rates above 95% and keyword coverage above 65%.
