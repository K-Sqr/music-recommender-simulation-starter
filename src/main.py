"""Command-line entry point for the Music Recommender Simulation.

Project 4 turns this from a 3-profile demo into a small CLI that wires
together the agent, the RAG knowledge base, the guardrails, and the
tabulated display layer. A startup menu offers three modes:

    1) Interactive agent (you type a natural-language vibe)
    2) Demo agent run (3 preset NL queries, ideal for Loom recordings)
    3) Legacy Project 3 profiles (original rule-based recommender)

Run ``python -m src.main`` to launch the menu, or pass ``--mock`` to skip
Claude entirely and use the deterministic mock parser.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional but recommended
    load_dotenv = None  # type: ignore[assignment]

from src.agent import AgentTrace, load_default_kb, run_agent
from src.display import render_legacy_results_table, render_results_table
from src.guardrails import get_logger
from src.recommender import load_songs, recommend_songs

DEMO_QUERIES: List[str] = [
    "I want something chill for studying late at night",
    "Pump me up for a workout, fast tempo only",
    "Romantic R&B for a quiet candle-lit dinner",
]

LEGACY_PROFILES: List[Tuple[str, dict]] = [
    ("High-Energy Pop", {"genre": "pop", "mood": "happy", "energy": 0.9}),
    ("Chill Lofi", {"genre": "lofi", "mood": "chill", "energy": 0.35}),
    ("Deep Intense Rock", {"genre": "rock", "mood": "intense", "energy": 0.92}),
]


def _print_trace_table(trace: AgentTrace, header: str) -> None:
    """Pretty-print a finished agent trace and its guardrail summary."""
    print()
    print(render_results_table(
        trace.recommendations,
        flags_by_id=trace.guardrail_flags,
        title=f"=== {header} ({trace.mode}) ===",
    ))
    print()


def _run_interactive(songs, kb, *, use_mock: bool) -> None:
    """Prompt for a single NL query and execute the agent on it."""
    print("\n--- Interactive Agent ---")
    print("Describe the kind of music you want (e.g. 'something chill for studying').")
    try:
        query = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n(no input received)")
        return

    if not query:
        print("(empty input, returning to menu)")
        return

    trace = run_agent(query, songs, kb, k=5, use_mock=use_mock, verbose=True)
    if trace.error:
        print(f"\nAgent could not run: {trace.error}")
        return
    _print_trace_table(trace, header=f"Recommendations for: {query!r}")


def _run_demo(songs, kb, *, use_mock: bool) -> None:
    """Run all three preset NL queries back-to-back for screenshots/Loom."""
    print("\n--- Demo Agent Run (3 preset NL queries) ---")
    for q in DEMO_QUERIES:
        print("\n" + "=" * 76)
        print(f"QUERY: {q}")
        print("=" * 76)
        trace = run_agent(q, songs, kb, k=5, use_mock=use_mock, verbose=True)
        if trace.error:
            print(f"Agent could not run for query: {trace.error}")
            continue
        _print_trace_table(trace, header=f"Recommendations for: {q!r}")


def _run_legacy(songs) -> None:
    """Run the original Project 3 hardcoded profiles via the rule-based scorer."""
    print("\n--- Legacy Project 3 Profiles (rule-based recommender) ---")
    for profile_name, user_prefs in LEGACY_PROFILES:
        triples = recommend_songs(user_prefs, songs, k=5)
        print()
        print(render_legacy_results_table(
            triples,
            title=f"=== {profile_name} ({user_prefs}) ===",
        ))
    print()


def _menu_loop(songs, kb, *, use_mock: bool) -> None:
    """Top-level menu driver. Loops until the user picks Quit."""
    options = {
        "1": "Interactive agent (type your own NL vibe)",
        "2": "Demo agent run (3 preset NL queries -- recommended for Loom)",
        "3": "Legacy Project 3 profiles (rule-based recommender)",
        "q": "Quit",
    }
    while True:
        print("\n=== Music Recommender Simulation -- Project 4 ===")
        if use_mock:
            print("(running in --mock mode: deterministic, no API calls)")
        for key, label in options.items():
            print(f"  [{key}] {label}")
        try:
            choice = input("Select an option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if choice == "1":
            _run_interactive(songs, kb, use_mock=use_mock)
        elif choice == "2":
            _run_demo(songs, kb, use_mock=use_mock)
        elif choice == "3":
            _run_legacy(songs)
        elif choice in {"q", "quit", "exit"}:
            print("Goodbye.")
            return
        else:
            print(f"(unknown option {choice!r})")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags. ``--mock`` skips Claude calls entirely."""
    parser = argparse.ArgumentParser(
        description="Music Recommender Simulation -- Project 4 CLI",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Use the deterministic rule-based parser instead of calling Claude. "
            "Recommended for grading without an API key."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["menu", "interactive", "demo", "legacy"],
        default="menu",
        help=(
            "Skip the menu and run a single mode directly. Defaults to 'menu'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Application entry point."""
    args = parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()  # picks up .env if present so ANTHROPIC_API_KEY is loaded

    logger = get_logger()
    logger.info("starting main mode=%s mock=%s", args.mode, args.mock)

    songs = load_songs("data/songs.csv")
    kb = load_default_kb()

    print(f"Loaded {len(songs)} songs and {len(kb.documents)} knowledge-base entries.")

    if args.mode == "menu":
        _menu_loop(songs, kb, use_mock=args.mock)
        return
    if args.mode == "interactive":
        _run_interactive(songs, kb, use_mock=args.mock)
        return
    if args.mode == "demo":
        _run_demo(songs, kb, use_mock=args.mock)
        return
    if args.mode == "legacy":
        _run_legacy(songs)
        return


if __name__ == "__main__":
    main(sys.argv[1:])
