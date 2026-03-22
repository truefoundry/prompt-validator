#!/usr/bin/env python3
"""
Trace Evaluation CLI
====================
Fetch ChatCompletion traces → run the full enhancement pipeline:
  1. get_recommendation  — score the original system prompt
  2. apply_recommendation — produce an enhanced prompt
  3. llm_judge           — compare original vs enhanced output quality

Usage:
  python cli/main.py                        # live traces, last 24h
  python cli/main.py --source file          # load from trace/traces.json
  python cli/main.py --hours 48             # live, last 48h
  python cli/main.py --fqn-filter "my-prompt"
  python cli/main.py --index 3              # skip selection, run trace #3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from trace.trace_parser import (  # noqa: E402
    TraceInput,
    fetch_live_spans,
    load_spans_from_file,
    parse_spans_to_inputs,
)
from cli.pipeline import (  # noqa: E402
    step_get_recommendations,
    step_apply_recommendations,
    step_llm_judge,
)

# ── app setup ─────────────────────────────────────────────────────────────────
app = typer.Typer(
    name="trace-eval",
    help="Trace-based prompt evaluation pipeline.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

DEFAULT_BACKEND_URL = os.getenv("PROMPT_TUNER_BASE_URL", "http://localhost:21120")
TRACES_JSON_PATH = ROOT / "trace" / "traces.json"


# ── display helpers ───────────────────────────────────────────────────────────

def _snip(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text


def _show_traces_table(inputs: list[TraceInput], max_rows: int = 30) -> None:
    shown = inputs[:max_rows]

    table = Table(
        title=f"ChatCompletion Traces  ({len(shown)} shown of {len(inputs)} total)",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("#", style="bold white", width=4, no_wrap=True)
    table.add_column("System Prompt", style="green", max_width=55)
    table.add_column("User Input", style="yellow", max_width=55)
    table.add_column("Assistant Response", style="blue", max_width=55)

    for i, ti in enumerate(shown):
        table.add_row(
            str(i + 1),
            _snip(ti.system_prompt, 120),
            _snip(ti.user_message, 120),
            _snip(ti.trace_output or "(none)", 120),
        )
    console.print(table)


def _show_trace_detail(ti: TraceInput) -> None:
    console.print(Panel(
        f"[bold]Model:[/bold]  {ti.model_name or '(unknown)'}\n"
        f"[bold]Span ID:[/bold] {ti.span_id}\n"
        f"[bold]FQN:[/bold]    {ti.prompt_fqn or '(none)'}\n\n"
        f"[bold green]System Prompt:[/bold green]\n{ti.system_prompt or '(none)'}\n\n"
        f"[bold yellow]User Input:[/bold yellow]\n{ti.user_message}\n\n"
        f"[bold blue]Assistant Response:[/bold blue]\n{ti.trace_output or '(none)'}",
        title="Selected Trace",
        border_style="cyan",
        expand=False,
    ))


def _show_recommendations(recommendations: list[str], total_score: int | None, criteria: dict) -> None:
    score_line = f"[bold]Total Score:[/bold] [cyan]{total_score}/100[/cyan]\n\n" if total_score is not None else ""

    criteria_lines = ""
    if criteria:
        criteria_lines = "[bold]Criteria Scores:[/bold]\n"
        for k, v in criteria.items():
            bar = "█" * (v // 10) + "░" * (10 - v // 10)
            criteria_lines += f"  {k:<32} {bar} {v}\n"
        criteria_lines += "\n"

    recs_text = "\n".join(f"[yellow]{i+1}.[/yellow] {r}" for i, r in enumerate(recommendations))

    console.print(Panel(
        score_line + criteria_lines + recs_text,
        title=f"Step 1 — Recommendations ({len(recommendations)})",
        border_style="yellow",
    ))


def _show_enhanced_prompt(enhanced: str) -> None:
    preview = enhanced[:800] + ("\n[dim]… (truncated)[/dim]" if len(enhanced) > 800 else "")
    console.print(Panel(
        preview,
        title="Step 2 — Enhanced Prompt",
        border_style="green",
    ))


def _show_judge_results(result: dict) -> None:
    summary = result.get("summary", {})
    test_results = result.get("test_results", [])
    total = summary.get("total_cases", 0)
    improved = summary.get("improved_count", 0)
    rate = f"{improved/total*100:.0f}%" if total else "N/A"

    console.print(Panel(
        f"[bold]Cases:[/bold] {total}  |  "
        f"[bold]Improved:[/bold] [green]{improved}[/green]  |  "
        f"[bold]Improvement Rate:[/bold] [cyan]{rate}[/cyan]",
        title="Step 3 — Judge Summary",
        border_style="magenta",
    ))

    # Metrics delta table
    avg_orig = summary.get("avg_original", {})
    avg_enh = summary.get("avg_enhanced", {})
    avg_delta = summary.get("avg_delta", {})

    if avg_orig:
        mt = Table(box=box.SIMPLE, header_style="bold white", expand=False)
        mt.add_column("Metric", style="cyan", min_width=20)
        mt.add_column("Original", justify="right", min_width=10)
        mt.add_column("Enhanced", justify="right", min_width=10)
        mt.add_column("Delta", justify="right", min_width=10)
        for metric in avg_orig:
            o = avg_orig.get(metric, 0)
            e = avg_enh.get(metric, 0)
            d = avg_delta.get(metric, 0)
            color = "green" if d > 0 else ("red" if d < 0 else "white")
            mt.add_row(metric, f"{o:.2f}", f"{e:.2f}", f"[{color}]{d:+.2f}[/{color}]")
        console.print(mt)

    # Per-test breakdown
    for tr in test_results:
        scores = tr.get("scores", {})
        improved_flag = scores.get("improved", False)
        border = "green" if improved_flag else "red"
        icon = "✓ Improved" if improved_flag else "✗ No improvement"
        overall = scores.get("overall", "N/A")

        body = (
            f"[bold]Input:[/bold]\n{tr.get('input','')[:300]}\n\n"
            f"[bold yellow]Original Output:[/bold yellow]\n{tr.get('original_output','')[:400]}\n\n"
            f"[bold green]Enhanced Output:[/bold green]\n{tr.get('enhanced_output','')[:400]}\n\n"
            f"[bold]Result:[/bold] {icon}  [bold]Overall:[/bold] {overall}\n"
            f"[bold]Summary:[/bold] {scores.get('improvement_summary', 'N/A')}"
        )
        diffs = scores.get("key_differences", [])
        if diffs:
            body += "\n[bold]Key Differences:[/bold]\n" + "\n".join(f"  • {d}" for d in diffs)

        console.print(Panel(
            body,
            title=f"Test: {tr.get('test_case_name', tr.get('test_case_id', ''))}",
            border_style=border,
        ))


# ── load helpers ──────────────────────────────────────────────────────────────

def _load_live(days: int, limit: int, fqn_filter: str | None) -> list[TraceInput]:
    with console.status(f"[cyan]Fetching live ChatCompletion spans (last {days}d, max {limit})…[/cyan]"):
        spans = fetch_live_spans(hours=days * 24, limit=limit, prompt_fqn_filter=fqn_filter or None)

    if not spans:
        console.print("[yellow]No spans returned for the given time range.[/yellow]")
        return []

    with console.status("[cyan]Parsing spans…[/cyan]"):
        inputs = parse_spans_to_inputs(spans)

    skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
    skipped = len(spans) - len(inputs)
    skip_note = f"  (skipped {skipped}: {skip_reasons})" if skipped else ""
    console.print(f"[green]✓[/green] Fetched [bold]{len(inputs)}[/bold] traces from [bold]{len(spans)}[/bold] spans{skip_note}")
    return inputs


def _load_file(file_path: Path) -> list[TraceInput]:
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    with console.status(f"[cyan]Loading {file_path.name}…[/cyan]"):
        spans = load_spans_from_file(str(file_path))
        inputs = parse_spans_to_inputs(spans)
    skip_reasons = getattr(parse_spans_to_inputs, "skip_reasons", {})
    skipped = len(spans) - len(inputs)
    skip_note = f"  (skipped {skipped}: {skip_reasons})" if skipped else ""
    console.print(f"[green]✓[/green] Loaded [bold]{len(inputs)}[/bold] traces from [bold]{len(spans)}[/bold] spans{skip_note}")
    return inputs


def _pick_trace(inputs: list[TraceInput], index: int | None) -> TraceInput:
    """Return the chosen TraceInput — interactive prompt if index is None."""
    if index is not None:
        if index < 1 or index > len(inputs):
            console.print(f"[red]Index {index} out of range (1–{len(inputs)}).[/red]")
            raise typer.Exit(1)
        return inputs[index - 1]

    while True:
        try:
            raw = typer.prompt(f"\nSelect trace [1–{len(inputs)}]")
            idx = int(raw)
            if 1 <= idx <= len(inputs):
                return inputs[idx - 1]
            console.print(f"[red]Enter a number between 1 and {len(inputs)}.[/red]")
        except (ValueError, KeyboardInterrupt):
            console.print("\n[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)


# ── main command ──────────────────────────────────────────────────────────────

@app.command()
def run(
    source: str = typer.Option(
        "live",
        "--source", "-s",
        help="Trace source: [bold]live[/bold] (TrueFoundry) or [bold]file[/bold] (trace/traces.json).",
    ),
    days: int = typer.Option(
        1,
        "--days", "-d",
        help="Days back to fetch when source=live.",
    ),
    limit: int = typer.Option(
        200,
        "--limit", "-l",
        help="Max spans to fetch (caps SDK pagination). Lower = faster.",
    ),
    fqn_filter: Optional[str] = typer.Option(
        None,
        "--fqn-filter", "-f",
        help="Optional prompt FQN substring filter (client-side).",
    ),
    file_path: Optional[str] = typer.Option(
        None,
        "--file",
        help="Path to traces JSON (defaults to trace/traces.json when source=file).",
    ),
    index: Optional[int] = typer.Option(
        None,
        "--index", "-i",
        help="Trace number to run (1-based). Skips interactive selection.",
    ),
    backend_url: str = typer.Option(
        DEFAULT_BACKEND_URL,
        "--backend-url",
        help="Backend API URL.",
        envvar="PROMPT_TUNER_BASE_URL",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Override model name (default: uses model from the trace).",
    ),
    max_tokens: int = typer.Option(15000, "--max-tokens", help="Max tokens for LLM calls."),
    temperature: float = typer.Option(0.1, "--temperature", help="Temperature for LLM calls."),
    session_id: str = typer.Option("cli-session", "--session-id", help="Session ID for API calls."),
    show_all: bool = typer.Option(False, "--show-all", help="Show all traces, not just first 30."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
    save_output: Optional[str] = typer.Option(
        None,
        "--save-output",
        help="Save full pipeline results to this JSON file.",
    ),
):
    """
    Run the full trace evaluation pipeline:

    \b
    1. Fetch ChatCompletion traces from TrueFoundry (or a local JSON file)
    2. Display a table: model | system prompt | user input | assistant response
    3. Select a trace
    4. Step 1/3 — Analyze system prompt → get recommendations + score
    5. Step 2/3 — Apply recommendations → produce enhanced prompt
    6. Step 3/3 — Run LLM-as-judge using the trace's model to compare quality
    """
    console.rule("[bold cyan]Trace Evaluation CLI[/bold cyan]")
    console.print(f"Backend: [bold]{backend_url}[/bold]\n")

    # ── Load traces ──────────────────────────────────────────────────────────
    if source == "live":
        inputs = _load_live(days, limit, fqn_filter)
    elif source == "file":
        fp = Path(file_path) if file_path else TRACES_JSON_PATH
        inputs = _load_file(fp)
    else:
        console.print(f"[red]Unknown source '{source}'. Use 'live' or 'file'.[/red]")
        raise typer.Exit(1)

    if not inputs:
        console.print("[yellow]No traces to evaluate. Exiting.[/yellow]")
        raise typer.Exit(0)

    # ── Show table ───────────────────────────────────────────────────────────
    max_rows = len(inputs) if show_all else 30
    _show_traces_table(inputs, max_rows=max_rows)

    # ── Select trace ─────────────────────────────────────────────────────────
    trace = _pick_trace(inputs, index)
    _show_trace_detail(trace)

    if not trace.system_prompt.strip():
        console.print("[red]Selected trace has no system prompt. Cannot run pipeline.[/red]")
        raise typer.Exit(1)

    # Use explicit --model override or backend default — not the trace's model
    resolved_model = (model or "").strip() or None

    # ── Confirm ──────────────────────────────────────────────────────────────
    if not yes:
        console.print(
            f"\n[bold]Ready to run pipeline[/bold] on this trace.\n"
            f"  Model:   [cyan]{resolved_model or 'backend default'}[/cyan]\n"
            f"  Backend: [cyan]{backend_url}[/cyan]\n"
        )
        confirmed = typer.confirm("Proceed?", default=True)
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    all_results: dict = {
        "span_id": trace.span_id,
        "model": resolved_model,
        "original_system_prompt": trace.system_prompt,
        "user_message": trace.user_message,
        "original_assistant_response": trace.trace_output,
    }

    # ── Step 1: Get recommendations ──────────────────────────────────────────
    console.rule("[yellow]Step 1/3 — Get Recommendations[/yellow]")
    try:
        with console.status("[yellow]Analyzing prompt…[/yellow]"):
            recommendations, total_score, raw1 = step_get_recommendations(
                backend_url, trace.system_prompt.strip(), resolved_model,
                session_id, max_tokens, temperature,
            )
    except Exception as exc:
        console.print(f"[red]Step 1 failed: {exc}[/red]")
        raise typer.Exit(1)

    if not recommendations:
        console.print("[red]No recommendations returned. Cannot proceed.[/red]")
        raise typer.Exit(1)

    criteria = {}
    try:
        from cli.pipeline import extract_criteria_scores
        criteria = extract_criteria_scores(raw1)
    except Exception:
        pass

    _show_recommendations(recommendations, total_score, criteria)
    all_results["total_score"] = total_score
    all_results["recommendations"] = recommendations

    # ── Step 2: Apply recommendations ────────────────────────────────────────
    console.rule("[green]Step 2/3 — Apply Recommendations[/green]")
    try:
        with console.status("[green]Enhancing prompt…[/green]"):
            enhanced_prompt, raw2 = step_apply_recommendations(
                backend_url, trace.system_prompt.strip(), recommendations,
                resolved_model, session_id, max_tokens, temperature,
            )
    except Exception as exc:
        console.print(f"[red]Step 2 failed: {exc}[/red]")
        raise typer.Exit(1)

    if not enhanced_prompt:
        console.print("[red]No enhanced prompt returned. Cannot proceed.[/red]")
        raise typer.Exit(1)

    _show_enhanced_prompt(enhanced_prompt)
    all_results["enhanced_prompt"] = enhanced_prompt

    # ── Step 3: LLM judge ─────────────────────────────────────────────────────
    console.rule("[magenta]Step 3/3 — LLM Judge[/magenta]")
    try:
        with console.status("[magenta]Running judge comparison…[/magenta]"):
            judge_result, raw3 = step_llm_judge(
                backend_url,
                trace.system_prompt.strip(),
                enhanced_prompt,
                trace.user_message,
                trace.span_id,
                resolved_model,
                session_id,
                max_tokens,
                temperature,
            )
    except Exception as exc:
        console.print(f"[red]Step 3 failed: {exc}[/red]")
        raise typer.Exit(1)

    if not judge_result:
        console.print("[red]No judge result returned.[/red]")
        raise typer.Exit(1)

    _show_judge_results(judge_result)
    all_results["judge_result"] = judge_result

    console.rule("[bold green]Pipeline Complete[/bold green]")

    # ── Save output ───────────────────────────────────────────────────────────
    if save_output:
        out_path = Path(save_output)
        out_path.write_text(json.dumps(all_results, indent=2, default=str))
        console.print(f"[green]✓[/green] Results saved to [bold]{out_path}[/bold]")


if __name__ == "__main__":
    app()
