#!/usr/bin/env python3
"""
StudyMind AI — Main Entry Point
================================
A RAG-powered, agentic study assistant that answers questions,
generates quizzes, and summarises topics grounded in your own notes.

Usage
-----
    python main.py                        # Interactive mode (default)
    python main.py --ingest <dir>         # Ingest a directory of notes
    python main.py --query "What is RAG?" # Single-query mode
    python main.py --demo                 # Run built-in demo queries
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.logger_config import setup_logging
from src.rag_engine import RAGEngine
from src.agent import StudyMindAgent
from src.guardrails import Guardrails

# ---------------------------------------------------------------------------
# Initialise logging before anything else
# ---------------------------------------------------------------------------
setup_logging(log_dir="logs", log_level=logging.CRITICAL)
logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def print_header():
    """Render the app masthead."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]StudyMind AI[/bold cyan]\n"
        "[white]RAG-powered study assistant with agent trace, sources, and guardrails[/white]",
        border_style="cyan",
        padding=(1, 4),
    ))


def print_notice(title: str, message: str, style: str = "yellow") -> None:
    """Render a short status or warning panel."""
    console.print(Panel(message, title=title, border_style=style, padding=(1, 2)))


def has_openai_key() -> bool:
    """Return True when the generation API key looks configured."""
    return os.environ.get("OPENAI_API_KEY", "").strip().startswith("sk-")

def print_response(response):
    """Pretty-print an AgentResponse object."""
    console.print()
    console.rule(f"[bold cyan]Query[/bold cyan] [white]{response.query}[/white]", style="cyan")

    trace = Table(
        title="Agent Trace",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
        expand=True,
    )
    trace.add_column("Step", justify="right", width=4)
    trace.add_column("Action", style="bold")
    trace.add_column("Result")
    trace.add_column("Confidence", justify="right", width=11)
    for step in response.steps:
        confidence_style = "green" if step.confidence >= 0.7 else ("yellow" if step.confidence >= 0.4 else "red")
        trace.add_row(
            str(step.step_number),
            step.action,
            step.output_summary,
            f"[{confidence_style}]{step.confidence:.0%}[/{confidence_style}]",
        )
    console.print(trace)

    conf = response.overall_confidence
    conf_style = "green" if conf >= 0.7 else ("yellow" if conf >= 0.45 else "red")
    sources = ", ".join(response.sources_used) if response.sources_used else "No retrieved sources"
    summary = Table.grid(expand=True)
    summary.add_column(ratio=1)
    summary.add_column(ratio=2)
    summary.add_row("[bold]Task[/bold]", response.task_type)
    summary.add_row("[bold]Overall confidence[/bold]", f"[{conf_style}]{conf:.0%}[/{conf_style}]")
    summary.add_row("[bold]Sources[/bold]", sources)
    console.print(Panel(summary, title="Run Summary", border_style=conf_style, padding=(1, 2)))

    console.print(Panel(
        Markdown(response.final_answer),
        title="Answer",
        border_style="cyan",
        padding=(1, 2),
    ))


# ---------------------------------------------------------------------------
# Core application logic
# ---------------------------------------------------------------------------

def build_system(notes_dir: str = "data/sample_notes") -> tuple:
    """Initialise RAG engine, agent, and guardrails; ingest notes."""
    console.print("[bold cyan]Initializing StudyMind AI...[/bold cyan]")

    with console.status("[cyan]Loading retriever and agent...[/cyan]", spinner="dots"):
        rag = RAGEngine()
        agent = StudyMindAgent(rag)
        guard = Guardrails()

    notes_path = Path(notes_dir)
    if notes_path.is_dir():
        with console.status(f"[cyan]Indexing notes from {notes_path.resolve()}...[/cyan]", spinner="dots"):
            results = rag.ingest_directory(str(notes_path))
        total_chunks = sum(results.values())
        table = Table(
            title=f"Knowledge Base: {len(results)} file(s), {total_chunks} chunks",
            box=box.SIMPLE,
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Source file")
        table.add_column("Chunks", justify="right")
        for fname, count in results.items():
            table.add_row(fname, str(count))
        console.print(table)
    else:
        print_notice(
            "Knowledge Base",
            f"Notes directory '{notes_dir}' was not found. Running without local study notes.",
            style="yellow",
        )

    return rag, agent, guard


def handle_query(query: str, agent: StudyMindAgent, guard: Guardrails) -> bool:
    """
    Validate and process a single query.

    Returns True if the query was processed, False if it was blocked.
    """
    # Sanitise
    clean_query = guard.sanitize_query(query)

    # Validate
    is_valid, validation_msg = guard.validate_input(clean_query)
    if not is_valid:
        print_notice("Guardrail", validation_msg, style="red")
        return False
    if validation_msg:
        print_notice("Note", validation_msg, style="yellow")

    # Run agent
    try:
        with console.status("[cyan]Running retrieval, generation, and verification...[/cyan]", spinner="dots"):
            response = agent.run(clean_query)
    except RuntimeError as exc:
        print_notice("Error", str(exc), style="red")
        logger.error("Agent failed for query '%s': %s", clean_query[:60], exc)
        return False

    # Confidence check
    _, conf_warning = guard.check_output_confidence(response.overall_confidence)
    if conf_warning:
        print_notice("Confidence Warning", conf_warning, style="yellow")

    print_response(response)
    return True


def interactive_mode(agent: StudyMindAgent, guard: Guardrails):
    """Run the interactive REPL loop."""
    print_header()
    commands = Table(box=box.SIMPLE, show_header=False)
    commands.add_column("Command", style="bold cyan")
    commands.add_column("Purpose")
    commands.add_row("/quiz <topic>", "Generate quiz questions")
    commands.add_row("/summary <topic>", "Get a concise summary")
    commands.add_row("/explain <topic>", "Get a detailed explanation")
    commands.add_row("/help", "Show commands")
    commands.add_row("/quit", "Exit")
    console.print(Panel(commands, title="Commands", border_style="cyan", padding=(1, 2)))

    while True:
        try:
            raw = console.input("[bold cyan]StudyMind > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Goodbye![/bold]")
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            console.print("[bold]Goodbye![/bold]")
            break
        elif cmd == "/help":
            console.print(commands)
            continue

        # Translate shorthand commands to natural language
        if raw.startswith("/quiz "):
            raw = "Generate a quiz on " + raw[6:]
        elif raw.startswith("/summary "):
            raw = "Summarize " + raw[9:]
        elif raw.startswith("/explain "):
            raw = "Explain " + raw[9:]

        handle_query(raw, agent, guard)


def demo_mode(agent: StudyMindAgent, guard: Guardrails):
    """Run a set of built-in demo queries to showcase the system."""
    print_header()
    console.print(Panel("Running built-in demo queries.", border_style="yellow", padding=(1, 2)))

    demo_queries = [
        "What is overfitting and how can it be prevented?",
        "Explain the attention mechanism in transformers.",
        "Summarize the key concepts of Retrieval-Augmented Generation.",
        "Generate a quiz on supervised learning.",
        "What are the ethical concerns with AI bias?",
    ]

    for query in demo_queries:
        console.rule(f"[bold yellow]Demo Query[/bold yellow] {query}", style="yellow")
        handle_query(query, agent, guard)
        console.input("[dim]Press Enter to continue to the next demo query...[/dim]")


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="StudyMind AI — RAG-Powered Study Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ingest", metavar="DIR",
        help="Directory of notes to ingest (default: data/sample_notes)",
    )
    parser.add_argument(
        "--query", metavar="QUERY",
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run built-in demo queries",
    )
    parser.add_argument(
        "--notes-dir", metavar="DIR", default="data/sample_notes",
        help="Directory of study notes to load at startup (default: data/sample_notes)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    notes_dir = args.ingest if args.ingest else args.notes_dir

    if args.query:
        guard = Guardrails()
        clean_query = guard.sanitize_query(args.query)
        is_valid, validation_msg = guard.validate_input(clean_query)
        if not is_valid:
            print_header()
            print_notice("Guardrail", validation_msg, style="red")
            return

    if not has_openai_key():
        print_header()
        print_notice(
            "OpenAI API Key",
            "Set OPENAI_API_KEY to a real OpenAI key before running generation. "
            "The key should start with 'sk-'.",
            style="red",
        )
        return

    rag, agent, guard = build_system(notes_dir=notes_dir)

    if args.query:
        handle_query(args.query, agent, guard)
    elif args.demo:
        demo_mode(agent, guard)
    else:
        interactive_mode(agent, guard)


if __name__ == "__main__":
    main()
