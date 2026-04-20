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
import sys
from pathlib import Path

from src.logger_config import setup_logging
from src.rag_engine import RAGEngine
from src.agent import StudyMindAgent
from src.guardrails import Guardrails

# ---------------------------------------------------------------------------
# Initialise logging before anything else
# ---------------------------------------------------------------------------
setup_logging(log_dir="logs", log_level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour helpers (graceful fallback on non-TTY)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def cyan(t):    return _c(t, "96")
def green(t):   return _c(t, "92")
def yellow(t):  return _c(t, "93")
def red(t):     return _c(t, "91")
def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
DIVIDER = "─" * 70

def print_header():
    print()
    print(bold(cyan("╔══════════════════════════════════════════════════════════════════╗")))
    print(bold(cyan("║              StudyMind AI — RAG-Powered Study Assistant          ║")))
    print(bold(cyan("╚══════════════════════════════════════════════════════════════════╝")))
    print()

def print_response(response):
    """Pretty-print an AgentResponse object."""
    print()
    print(DIVIDER)
    print(bold(f"Query: {response.query}"))
    print(DIVIDER)

    # Reasoning trace
    print(bold(cyan("\n[Reasoning Trace]")))
    for step in response.steps:
        icon = "✓" if step.confidence >= 0.7 else ("⚠" if step.confidence >= 0.4 else "✗")
        print(f"  Step {step.step_number}: {bold(step.action)}")
        print(f"    → {step.output_summary}")
        print(dim(f"    Confidence: {step.confidence:.0%}  {icon}"))

    # Sources
    if response.sources_used:
        print(bold(cyan("\n[Sources Used]")))
        for src in response.sources_used:
            print(f"  • {src}")

    # Confidence
    conf = response.overall_confidence
    conf_color = green if conf >= 0.7 else (yellow if conf >= 0.45 else red)
    print(bold(cyan("\n[Overall Confidence]")), conf_color(f"{conf:.0%}"))

    # Final answer
    print(bold(cyan("\n[Answer]")))
    print(response.final_answer)
    print()


# ---------------------------------------------------------------------------
# Core application logic
# ---------------------------------------------------------------------------

def build_system(notes_dir: str = "data/sample_notes") -> tuple:
    """Initialise RAG engine, agent, and guardrails; ingest notes."""
    print(bold("\nInitialising StudyMind AI..."))

    rag = RAGEngine()
    agent = StudyMindAgent(rag)
    guard = Guardrails()

    # Ingest notes
    notes_path = Path(notes_dir)
    if notes_path.is_dir():
        print(f"Ingesting notes from: {notes_path.resolve()}")
        results = rag.ingest_directory(str(notes_path))
        total_chunks = sum(results.values())
        print(green(f"  ✓ Ingested {len(results)} file(s) → {total_chunks} chunks loaded into knowledge base."))
        for fname, count in results.items():
            print(dim(f"    {fname}: {count} chunks"))
    else:
        print(yellow(f"  ⚠ Notes directory '{notes_dir}' not found. Running without knowledge base."))

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
        print(red(f"\n[Guardrail] {validation_msg}"))
        return False
    if validation_msg:
        print(yellow(f"\n[Note] {validation_msg}"))

    # Run agent
    try:
        response = agent.run(clean_query)
    except RuntimeError as exc:
        print(red(f"\n[Error] {exc}"))
        logger.error("Agent failed for query '%s': %s", clean_query[:60], exc)
        return False

    # Confidence check
    _, conf_warning = guard.check_output_confidence(response.overall_confidence)
    if conf_warning:
        print(yellow(f"\n[Confidence Warning] {conf_warning}"))

    print_response(response)
    return True


def interactive_mode(agent: StudyMindAgent, guard: Guardrails):
    """Run the interactive REPL loop."""
    print_header()
    print("Type your study question, or use one of these commands:")
    print(dim("  /quiz <topic>    — Generate quiz questions on a topic"))
    print(dim("  /summary <topic> — Get a concise summary of a topic"))
    print(dim("  /explain <topic> — Get a detailed explanation"))
    print(dim("  /help            — Show this help message"))
    print(dim("  /quit or /exit   — Exit the application"))
    print()

    while True:
        try:
            raw = input(bold(cyan("StudyMind > "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(bold("\nGoodbye!"))
            break

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            print(bold("Goodbye!"))
            break
        elif cmd == "/help":
            print(dim("  /quiz <topic>    — Generate quiz questions on a topic"))
            print(dim("  /summary <topic> — Get a concise summary of a topic"))
            print(dim("  /explain <topic> — Get a detailed explanation"))
            print(dim("  /quit or /exit   — Exit the application"))
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
    print(bold(yellow("Running demo queries...\n")))

    demo_queries = [
        "What is overfitting and how can it be prevented?",
        "Explain the attention mechanism in transformers.",
        "Summarize the key concepts of Retrieval-Augmented Generation.",
        "Generate a quiz on supervised learning.",
        "What are the ethical concerns with AI bias?",
    ]

    for query in demo_queries:
        print(bold(f"\n{'='*70}"))
        print(bold(f"Demo Query: {query}"))
        print(bold(f"{'='*70}"))
        handle_query(query, agent, guard)
        input(dim("  [Press Enter to continue to next demo query...]"))


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

    rag, agent, guard = build_system(notes_dir=notes_dir)

    if args.query:
        handle_query(args.query, agent, guard)
    elif args.demo:
        demo_mode(agent, guard)
    else:
        interactive_mode(agent, guard)


if __name__ == "__main__":
    main()
