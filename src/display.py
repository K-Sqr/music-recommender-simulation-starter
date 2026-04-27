"""Pretty-printers for recommender output (Project 3 visual stretch).

The agent and the legacy CLI both emit results as a list of dicts shaped
like::

    {"song": <song dict>, "score": float, "mode": str, "explanation": str,
     "breakdown": {...}, "reasons": [...]}

This module turns that list into a tabulated table that fits in a terminal
and, more importantly, in a Loom recording.
"""
from __future__ import annotations

import textwrap
from typing import Dict, Iterable, List

from tabulate import tabulate

# Width allocated to the explanation column when wrapping.
_EXPLANATION_WIDTH = 60


def _wrap(text: str, width: int = _EXPLANATION_WIDTH) -> str:
    """Soft-wrap a long string so the table stays readable in narrow terminals."""
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _flag_marker(flags: Iterable[str]) -> str:
    """Return a short human-readable marker for guardrail flags on a song."""
    flags = list(flags or [])
    return f"FLAG x{len(flags)}" if flags else "OK"


def render_results_table(
    recs: List[Dict],
    *,
    flags_by_id: Dict[str, List[str]] | None = None,
    title: str | None = None,
    table_format: str = "grid",
) -> str:
    """Render ranked recommendations as a tabulated string.

    Columns: Rank, Title, Artist, Genre, Score, Mode, Guardrail, Explanation.
    """
    flags_by_id = flags_by_id or {}
    rows: List[List[str]] = []
    for i, r in enumerate(recs, start=1):
        song = r["song"]
        sid = str(song.get("id", ""))
        rows.append(
            [
                i,
                song.get("title", ""),
                song.get("artist", ""),
                song.get("genre", ""),
                f"{float(r.get('score', 0.0)):.2f}",
                r.get("mode", ""),
                _flag_marker(flags_by_id.get(sid, [])),
                _wrap(r.get("explanation") or ", ".join(r.get("reasons", []))),
            ]
        )

    table = tabulate(
        rows,
        headers=[
            "#",
            "Title",
            "Artist",
            "Genre",
            "Score",
            "Mode",
            "Check",
            "Why",
        ],
        tablefmt=table_format,
    )

    if title:
        return f"{title}\n{table}"
    return table


def render_legacy_results_table(
    triples,
    *,
    title: str | None = None,
    table_format: str = "grid",
) -> str:
    """Render the Project 3 ``recommend_songs`` 3-tuple output as a table.

    Useful for the legacy menu option in main.py so even the original
    rule-based mode benefits from the visual stretch.
    """
    rows: List[List[str]] = []
    for i, (song, score, explanation) in enumerate(triples, start=1):
        rows.append(
            [
                i,
                song.get("title", ""),
                song.get("artist", ""),
                song.get("genre", ""),
                f"{float(score):.2f}",
                "legacy_p3",
                _wrap(explanation),
            ]
        )

    table = tabulate(
        rows,
        headers=["#", "Title", "Artist", "Genre", "Score", "Mode", "Why"],
        tablefmt=table_format,
    )
    if title:
        return f"{title}\n{table}"
    return table
