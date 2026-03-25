# Codebase Issues

Tracked findings from the full code review. Check off items as they are resolved.

---

## Bugs (Fix First)

- `**chat_service.py:200-202**` — `_get_last_ai_message(events)` called N times inside a loop, always returns same result. Loop is pointless and should be removed.
- `**create_test_set.py:1**` — Stray `8` character before the docstring. Syntax error, file fails to parse.

---

## High Priority

### `src/chat/graph/primary_graph.py` — God Function

- `validator()` is 143 lines with 8 `elif` branches, each handling a completely different request type. Split into separate handler functions dispatched by request type.
- JSON fence-stripping (`````) copy-pasted inline at lines 159–164 and 319–324 despite `_sanitize_json_like_output` already existing in `chat_service.py`. Consolidated into `_strip_json_fences()` used by all handlers.
- Hardcoded relative file path `src/chat/data/test_results/test_results.json` written inside the graph node. Moved into `_call_prompt_service()`.
- Two large prompt strings (`_BEHAVIORAL_REC_PROMPT`, `_GENERATE_SUGGESTIONS_SYSTEM_PROMPT`) embedded inline in code. Moved to `prompts/behavioral_recommendations_system.txt` and `prompts/generate_suggestions_system.txt`, loaded at startup via `Path.read_text()`.
- Deferred imports inside `_handle_arena_comparison` and `_handle_deepeval_prompt_metrics`. Move to top of file.

### `ui/tabs/enhance.py` — 689-line file

- `_apply_suggestions_to_trace_prompt` and `_apply_suggestions_to_enhance_prompt` merged into one `_apply_suggestions()` with keyword params; thin wrappers kept for call-site readability.
- `_METRIC_GROUPS`, `_ALL_METRIC_KEYS`, `_DEFAULT_METRICS` — now imports `DEFAULT_METRICS` and `ALL_METRIC_KEYS` directly from `llm_judge_evaluator.py`.
- `hallucination_avoidance` duplicate removed from `_METRIC_GROUPS` "Conversational" group (kept in "Guardrails").
- `import requests`, `import pandas as pd` moved to top of file; `import json as _json` removed (uses `json` already imported at top).

### `ui/actions/evaluation.py` — Duplicated action functions

- `generate_overall_suggestions()` and `generate_enhance_suggestions()` merged into `_generate_suggestions()` with keyword params; thin public wrappers kept.

---

## Medium Priority

### `src/chat/service/chat_service.py`

- `_get_last_ai_message` declared `async def` but contains no `await`. Removed `async` and corresponding `await` at call site.
- Copy-pasted comment block describing healthcare chatbot workflow. Removed both stale comment blocks.
- JSON parsing inline at lines 259–276 duplicates the `_parse_with_sanitizer` already defined in the same file (lines 37–50). *(Careful — needs separate `_parse_json_with_fallback` helper, not direct reuse)*
- `traceback.print_exc()` and `error(...)` both called on exception. Removed `traceback.print_exc()`, kept `error()`.

### `src/chat/graph/deepeval_prompt_metrics.py`

- `run_deepeval_prompt_metrics` is sync wrapping an async loop, relying on `nest_asyncio` patching (lines 151–159). Should be `async def` awaited directly.
- Test cases processed sequentially inside `_run_all()` despite async context. Use `asyncio.gather` to parallelize across cases.
- `_safe_measure_sync` defined but never called — removed.

### `src/common/config/app_config.py`

- File `open()` at module level (lines 12–23) — any import failure causes silent `OSError` swallowing and leaves `yaml_settings` undefined, which would cause a `NameError` downstream.
- `yaml_settings.update(env_settings)` merges the entire OS environment into config. Any env var (e.g. `HOME`, `PATH`) becomes accessible via `CONFIG.get()`.

### `src/common/util/method_stats_util.py`

- Sync decorator applied to `async def _build_state_graph` (in `primary_graph.py`) — breaks awaiting, returns a coroutine object instead of the awaited result.
- Lines 16–48 large commented-out async implementation — deleted.

### `ui/tabs/trace_eval.py`

- `_render_prompt_input` duplicated from `ui/tabs/_shared.py`. Use the shared version. *(Careful — needs `fqn_widget_key` param added to _shared.py first)*
- `_JUDGE_METRICS` hardcoded locally — now `DEFAULT_METRICS + ["overall"]` imported from `llm_judge_evaluator.py`.
- `import pandas as pd` deferred inside render functions — moved to top of file.

### `ui/trace_actions.py`

- `run_trace_pipeline` Step 3 (lines 207–237) and `run_llm_judge_on_traces` (lines 285–346) duplicate the same payload construction and test case list building logic.
- `load_trace_inputs` and `fetch_live_trace_inputs` share ~20 lines of identical post-processing logic (lines 82–91 and 126–135). Extract to a shared helper.

### `trace/trace_parser.py`

- `parse_spans_to_inputs.skip_reasons = skip_reasons` (line 278) — mutates a function attribute as a return channel. 4 call sites depend on this side effect (`ui/trace_actions.py` lines 70, 85, 117, 129). Change to return a tuple `(results, skip_reasons)`.

### `trace/fetch_trace.py`

- No `if __name__ == "__main__"` guard — all API calls and file writes execute on import.
- FQN validation logic duplicated from `trace_parser.py` line 209.

### `ui/state.py`

- Default `session_id = "123"` (line 8) — all users who don't change the sidebar default share the same LangGraph checkpointer thread.

### `src/chat/graph/llm_judge_evaluator.py`

- Two independent `asyncio.Semaphore(5)` instances (lines 141, 236) — no shared limit across the two phases. Document or unify.
- `asyncio.get_event_loop().time()` deprecated in Python 3.10+. Replaced with `asyncio.get_running_loop().time()`.

---

## Low Priority / Dead Code

### `src/chat/utils/chat_utils.py`

- `create_entry_node()` and `pop_dialog_state()` are orphaned from a removed multi-assistant architecture. Docstrings still mention "prescriptions and pharmacies". Remove.
- `get_prompt_details()` is a one-line wrapper with no value. Remove and use `PromptService.get_prompt_details` directly.

### `src/chat/models/state_model.py`

- `dialog_state`, `interrupt_state`, `interrupt_question`, `update_stack()`, `_get_interrupt_states()`, `_get_assistant_names()` all unused — remnants of removed multi-assistant architecture. Remove.

### `src/chat/utils/llm_models.py`

- `AzureOpenAI` class (lines 269–319) defined but never imported or used anywhere. Remove or document as intentional.
- `HumanMessage` imported at top of file and again inside `generate()` / `a_generate()`. Remove redundant local imports.

### `src/common/service/logging/logger.py`

- `_method_stats`, `_functional_tags`, `_functional_metrics`, `_llm_stats` accumulate in memory but are never read or exported to any observability backend.
- `info()` raises `ValueError` if message is not a string — unnecessary guard that would crash on any non-string log message.

### `config/application.yaml`

- Dead `redis` (lines 29–39) and `postgres` (lines 41–45) blocks — not used anywhere.
- `assistants.definitions` is read in `state_model.py` but does not exist in the YAML.

### `ui/state.py`

- `trace_applications` key initialized but never set or read anywhere.

---

## Cross-Cutting Patterns

- **JSON fence-stripping duplicated in 4 places** — `primary_graph.py` instances consolidated into `_strip_json_fences()`. Remaining instance in `llm_judge_evaluator.py:207-209` still to be updated.
- `**reasoning if reasoning != "none" else None*`* repeated in 8+ action functions across `evaluation.py`, `recommendations.py`, `trace_actions.py`, `enhance.py`. Move into `_build_base_payload()` in `helpers.py`.
- `**model_name.strip()` guard** repeated in 6+ action functions. Encapsulate.
- **No shared `TestCase` schema** — `trace_actions.py` uses `data.input`, `arena_evaluator.py` handles both `data.input` and bare `input`, `deepeval_prompt_metrics.py` expects `input`/`original_output`/`enhanced_output`. Define a `TypedDict` or dataclass and use it consistently.
- `**_build_base_payload()` exists in `helpers.py`** but is not used in `evaluation.py` — every action function manually rebuilds the same session state fields inline.

