"""Test harness for the Music Recommender agent (Project 4 reliability stretch).

Runs the full agent pipeline against:

- 5 representative natural-language vibe queries
- 3 edge cases (empty / whitespace / gibberish)

For each input it scores:

- ``valid``: did input validation behave correctly (accept good / reject bad)?
- ``consistent``: did the output guardrail flag any explanation as
  inconsistent with the score breakdown?
- ``distinct_genres``: how many different genres landed in the top-5
  (a coverage / filter-bubble proxy)?
- ``distinct_artists``: how many different artists landed in the top-3
  (a direct check on the artist diversity penalty)?
- ``confidence``: ``(top_score - median_score) / top_score`` -- how
  decisively the top result beat the median.

Run::

    python eval.py            # uses live Claude if ANTHROPIC_API_KEY is set
    python eval.py --mock     # deterministic, no API key needed

The summary is printed as a tabulated report and an overall pass/fail line.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tabulate import tabulate

from src.agent import AgentTrace, load_default_kb, run_agent
from src.guardrails import get_logger
from src.recommender import load_songs


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """One row in the eval harness."""

    label: str
    query: str
    should_validate: bool = True  # True => valid input, False => expected reject
    expected_genre: str | None = None  # optional sanity check on top result


GOOD_CASES: List[TestCase] = [
    TestCase(
        label="chill_study",
        query="I want something chill for studying late at night",
        expected_genre=None,  # multiple acceptable: lofi, ambient
    ),
    TestCase(
        label="workout_pump",
        query="Pump me up for a workout, fast tempo only",
        expected_genre=None,  # pop, rock, metal all acceptable
    ),
    TestCase(
        label="romantic_dinner",
        query="Romantic R&B for a quiet candle-lit dinner",
        expected_genre="r&b",
    ),
    TestCase(
        label="oldschool_metal",
        query="Old-school metal, the heavier the better",
        expected_genre="metal",
    ),
    TestCase(
        label="world_uplifting",
        query="Uplifting world music with global influences",
        expected_genre="world",
    ),
]

EDGE_CASES: List[TestCase] = [
    TestCase(label="empty_string", query="", should_validate=False),
    TestCase(label="whitespace", query="    \t  ", should_validate=False),
    TestCase(label="gibberish", query="asdfqwerty", should_validate=False),  # 1 token
]


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------


def _confidence(trace: AgentTrace) -> float:
    """``(top - median) / top`` for the recommendation scores. 0 if empty."""
    scores = [r["score"] for r in trace.recommendations]
    if not scores:
        return 0.0
    top = max(scores)
    if top <= 0:
        return 0.0
    median = statistics.median(scores)
    return round(max(0.0, (top - median) / top), 3)


def _distinct(values: List[str]) -> int:
    return len({v for v in values if v})


def evaluate_case(
    case: TestCase,
    songs: List[Dict],
    kb,
    *,
    use_mock: bool,
) -> Dict:
    """Run one case through the agent and compute the eval metrics."""
    logger = get_logger()
    logger.info("eval case=%s mock=%s", case.label, use_mock)

    start = time.time()
    trace = run_agent(case.query, songs, kb, k=5, use_mock=use_mock, verbose=False)
    elapsed = round(time.time() - start, 2)

    valid_correct = trace.valid == case.should_validate

    consistent = not any(trace.guardrail_flags.values()) if trace.recommendations else True

    genres_top5 = [r["song"].get("genre", "") for r in trace.recommendations]
    artists_top3 = [r["song"].get("artist", "") for r in trace.recommendations[:3]]

    expected_match = True
    if case.expected_genre and trace.recommendations:
        top_genre = trace.recommendations[0]["song"].get("genre", "")
        expected_match = top_genre == case.expected_genre

    confidence = _confidence(trace)

    # Overall pass: input validation behaved correctly AND, for valid inputs,
    # the agent produced consistent recs that hit the expected genre (when
    # specified). Edge cases pass simply by being correctly rejected.
    if not case.should_validate:
        passed = valid_correct
    else:
        passed = (
            valid_correct
            and consistent
            and bool(trace.recommendations)
            and expected_match
        )

    return {
        "label": case.label,
        "query": case.query if case.query else "(empty)",
        "valid_correct": valid_correct,
        "consistent": consistent,
        "expected_match": expected_match,
        "distinct_genres_top5": _distinct(genres_top5),
        "distinct_artists_top3": _distinct(artists_top3),
        "confidence": confidence,
        "elapsed_s": elapsed,
        "mode": trace.mode,
        "passed": passed,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_report(results: List[Dict]) -> str:
    """Render the per-case + summary table for the harness output."""
    rows = []
    for r in results:
        rows.append(
            [
                r["label"],
                ("OK" if r["valid_correct"] else "FAIL"),
                ("OK" if r["consistent"] else "FLAG"),
                ("OK" if r["expected_match"] else "MISS"),
                r["distinct_genres_top5"],
                r["distinct_artists_top3"],
                r["confidence"],
                r["elapsed_s"],
                r["mode"],
                ("PASS" if r["passed"] else "FAIL"),
            ]
        )
    table = tabulate(
        rows,
        headers=[
            "Case",
            "Validation",
            "Consistency",
            "Expected",
            "Genres top5",
            "Artists top3",
            "Confidence",
            "Latency (s)",
            "Mode",
            "Result",
        ],
        tablefmt="grid",
    )

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    confidences = [r["confidence"] for r in results if r["passed"] and r["confidence"]]
    avg_conf = round(statistics.mean(confidences), 3) if confidences else 0.0
    avg_genres = round(
        statistics.mean(r["distinct_genres_top5"] for r in results if r["distinct_genres_top5"]) or 0.0,
        2,
    ) if any(r["distinct_genres_top5"] for r in results) else 0.0

    summary = (
        f"\nSUMMARY: {passed}/{total} passed | "
        f"avg confidence (passing cases) {avg_conf} | "
        f"avg distinct genres top-5 {avg_genres}"
    )
    return f"{table}{summary}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags. ``--mock`` runs without an API key."""
    parser = argparse.ArgumentParser(description="Music Recommender eval harness.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with the deterministic rule-based parser (no API key required).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Run the harness and print a report. Returns 0 on full pass, 1 otherwise."""
    args = parse_args(argv)

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    logger = get_logger()
    logger.info("eval harness starting mock=%s", args.mock)

    songs = load_songs("data/songs.csv")
    kb = load_default_kb()
    print(
        f"Eval harness loaded {len(songs)} songs and {len(kb.documents)} KB docs. "
        f"Mock mode: {args.mock}"
    )

    results: List[Dict] = []
    for case in GOOD_CASES + EDGE_CASES:
        print(f"  running case: {case.label} ...")
        results.append(evaluate_case(case, songs, kb, use_mock=args.mock))

    print()
    print(render_report(results))

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"  - {r['label']} (mode={r['mode']}): {r['query']!r}")
            for sid, flags in r["trace"].guardrail_flags.items():
                for f in flags:
                    print(f"      flag song {sid}: {f}")
        return 1
    print("\nAll cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
