"""Reliability layer: input validation, output consistency checks, and logging.

The guardrails sit at two boundaries:

1. **Input** -- :func:`validate_user_input` rejects empty/whitespace/abusive
   input before it ever reaches the agent.
2. **Output** -- :func:`check_explanation_consistency` parses the agent's
   natural-language explanation and verifies that every dimension it
   mentions actually contributed to the song's score. If the agent says
   "matches your high-energy vibe" but the energy term contributed ~0, we
   raise a flag.

Both layers funnel into a shared rotating file logger at ``logs/agent.log``
so eval.py and main.py share a single auditable trail.
"""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

# Public so callers don't need to import logging just to silence noisy libs.
LOGGER_NAME = "music_agent"

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_logger_initialised = False


def setup_logger(
    log_path: str | Path = "logs/agent.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure (idempotently) the shared file logger and return it."""
    global _logger_initialised
    logger = logging.getLogger(LOGGER_NAME)

    if _logger_initialised:
        return logger

    logger.setLevel(level)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_path, maxBytes=512_000, backupCount=2, encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers if pytest reloads the module
    logger.handlers = [handler]
    logger.propagate = False
    _logger_initialised = True
    logger.info("Logger initialised at %s", log_path)
    return logger


def get_logger() -> logging.Logger:
    """Return the shared logger, initialising it lazily if needed."""
    return setup_logger()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# Tiny obvious blocklist; we are not trying to be a content moderation API,
# just to refuse the most clearly unwanted prompts before paying for a
# Claude call.
_BLOCKLIST = {
    "kill", "bomb", "suicide", "racist", "slur",
}

_MIN_TOKENS = 2
_MAX_LEN = 500


def validate_user_input(text: str) -> Tuple[bool, str]:
    """Return ``(is_valid, reason)`` for a raw NL query.

    Reasons are short, user-facing, and safe to print directly to the CLI.
    """
    if text is None:
        return False, "input is missing"
    if not isinstance(text, str):
        return False, "input must be a string"
    stripped = text.strip()
    if not stripped:
        return False, "input is empty"
    if len(stripped) > _MAX_LEN:
        return False, f"input too long (max {_MAX_LEN} characters)"
    tokens = re.findall(r"[A-Za-z']+", stripped)
    if len(tokens) < _MIN_TOKENS:
        return False, "please describe the vibe in a few words"

    lower = stripped.lower()
    for bad in _BLOCKLIST:
        if re.search(rf"\b{re.escape(bad)}\b", lower):
            return False, "input contains disallowed content"

    return True, "ok"


# ---------------------------------------------------------------------------
# Output consistency check
# ---------------------------------------------------------------------------

# Map dimensions to the regex patterns we expect the agent to use when it
# claims that dimension drove a recommendation. Patterns are word-bounded
# and tuned to avoid overlapping with each other (e.g. "mood" must not
# match "mood tag", and "dance" alone is too ambiguous so we require
# "danceabil*" or a true dance-related stem).
_DIMENSION_PATTERNS: Dict[str, List[str]] = {
    "genre": [r"\bgenre\b", r"\bstyle match\b", r"\bstylistic\b"],
    # \bmood\b but NOT followed by optional space/hyphen + "tag(s)"
    "mood": [r"\bmood\b(?!\s*[- ]?tags?)"],
    "mood_tags": [r"\bmood[- ]?tags?\b", r"\btags?\b", r"\batmosphere\b"],
    "energy": [
        r"\benergy\b",
        r"\benergetic\b",
        r"\bhigh-energy\b",
        r"\blow-energy\b",
        r"\bintensity\b",
    ],
    "danceability": [r"\bdanceabilit\w*\b", r"\bdanceable\b", r"\bgroove\b", r"\bgroovy\b"],
    "acousticness": [r"\bacoustic(ness)?\b", r"\bunplugged\b"],
    "popularity": [r"\bpopular(ity)?\b", r"\bmainstream\b", r"\bwell-known\b"],
    "decade": [
        r"\bdecade\b",
        r"\bera\b",
        r"\bold-school\b",
        r"\bretro\b",
        r"\bnostalgic\b",
        r"\bvintage\b",
        r"\b(19[5-9]0s|20[0-2]0s)\b",
    ],
}

# Below this absolute contribution we treat a dimension as "did not really
# drive the recommendation" -- mentioning it counts as an over-claim.
_MIN_CONTRIBUTION = 0.1
# Only flag a *missing* mention when a dimension dominates this much.
_MISSED_MENTION_FLOOR = 1.0


def _strip_prefix(explanation: str) -> str:
    """Drop the optional "<title> by <artist>:" prefix before keyword matching.

    Without this strip, a song title like "Velvet Slow Dance" would
    accidentally trigger the danceability check.
    """
    if not explanation:
        return ""
    if ":" in explanation:
        # Use partition to get only the segment after the FIRST colon.
        _, _, rest = explanation.partition(":")
        return rest.strip()
    return explanation


def check_explanation_consistency(
    explanation: str,
    breakdown: Dict[str, float],
) -> List[str]:
    """Return a list of human-readable flags where text and math disagree.

    An empty list means the explanation is consistent with the score
    breakdown. Flags include both *over-claims* (text mentions a dimension
    that contributed little) and *missed mentions* (a high-impact dimension
    that the agent did not justify).
    """
    flags: List[str] = []
    if not explanation:
        return ["explanation is empty"]

    text = _strip_prefix(explanation).lower()
    if not text:
        return ["explanation has no content after the title prefix"]

    # Over-claim detection: any dimension whose patterns appear in the text
    # but contributed less than the threshold.
    for dim, patterns in _DIMENSION_PATTERNS.items():
        mentioned = any(re.search(p, text) for p in patterns)
        if not mentioned:
            continue
        contrib = abs(float(breakdown.get(dim, 0.0)))
        if contrib < _MIN_CONTRIBUTION:
            flags.append(
                f"explanation references '{dim}' but its score contribution was "
                f"only {contrib:.2f}"
            )

    # Missed-mention detection: dimension contributed materially but the
    # explanation never references it. Only triggered for dominant dims to
    # avoid flagging every minor term.
    for dim, patterns in _DIMENSION_PATTERNS.items():
        contrib = float(breakdown.get(dim, 0.0))
        if contrib < _MISSED_MENTION_FLOOR:
            continue
        mentioned = any(re.search(p, text) for p in patterns)
        if not mentioned:
            flags.append(
                f"'{dim}' contributed {contrib:.2f} to the score but is not mentioned "
                "in the explanation"
            )

    return flags


def log_guardrail_results(
    label: str,
    flags: List[str],
) -> None:
    """Log guardrail flags at the right severity for downstream analysis."""
    logger = get_logger()
    if flags:
        logger.warning("guardrail flags for %s: %s", label, "; ".join(flags))
    else:
        logger.info("guardrail clean for %s", label)
