"""Multi-step agentic workflow for the Music Recommender.

The agent takes a natural-language query (e.g. "I want something chill for
studying late at night"), grounds it with a RAG retrieval over the local
knowledge base, picks scoring weights and a ranking mode, then invokes the
recommender as a tool to produce a ranked, explained list.

Pipeline (each step is printed to stdout so it shows up in a Loom recording
and is also written to ``logs/agent.log``):

    [STEP 1/5] Validate input
    [STEP 2/5] Retrieve context from RAG knowledge base
    [STEP 3/5] Plan + assign scoring weights (Claude or mock)
    [STEP 4/5] Tool call: recommend_songs (deterministic Python)
    [STEP 5/5] Generate per-song explanations + guardrail check

Two execution modes are supported:

- **Live**: uses ``anthropic.Anthropic`` with ``claude-sonnet-4-20250514``
  and Claude's native tool-use API so the agent literally calls
  ``recommend_songs`` as a tool.
- **Mock**: deterministic rule-based parser. No API key required. Useful
  for the eval harness, CI, and graders without a key.

Both modes return the same shaped result so downstream code (display,
guardrails, eval) is mode-agnostic.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .guardrails import (
    check_explanation_consistency,
    get_logger,
    log_guardrail_results,
    validate_user_input,
)
from .knowledge_base import KnowledgeBase
from .recommender import (
    RANKING_MODES,
    default_weights,
    recommend_songs_weighted,
)

CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class AgentTrace:
    """Captured artefacts from a single agent run, useful for eval and Loom."""

    query: str
    valid: bool
    validation_reason: str
    kb_results: List[Dict] = field(default_factory=list)
    plan: Dict = field(default_factory=dict)
    recommendations: List[Dict] = field(default_factory=list)
    guardrail_flags: Dict[str, List[str]] = field(default_factory=dict)
    mode: str = "agent"  # "agent" (live), "mock", or "error"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool schema exposed to Claude
# ---------------------------------------------------------------------------

RECOMMEND_TOOL_SCHEMA = {
    "name": "recommend_songs",
    "description": (
        "Score and rank the local song catalog given a ranking mode, scoring "
        "weights, and any directly-targeted attributes. Use after you have "
        "decided how to interpret the user's vibe."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": list(RANKING_MODES.keys()),
                "description": (
                    "Which preset to use as the base weight vector. "
                    "'genre_first' boosts genre, 'mood_first' boosts mood "
                    "and mood-tag overlap, 'energy_similarity' boosts energy "
                    "and danceability."
                ),
            },
            "weights": {
                "type": "object",
                "description": (
                    "Optional per-dimension overrides on top of the mode "
                    "preset. Keys: genre, mood, mood_tags, energy, "
                    "danceability, acousticness, popularity, decade."
                ),
                "additionalProperties": {"type": "number"},
            },
            "user_prefs": {
                "type": "object",
                "description": (
                    "Targeted attributes parsed from the query. Optional "
                    "keys: genre, mood, mood_tags (list of strings), "
                    "energy (0-1), danceability (0-1), acousticness (0-1), "
                    "decade (e.g. '2020s'), min_popularity (0-100)."
                ),
                "properties": {
                    "genre": {"type": "string"},
                    "mood": {"type": "string"},
                    "mood_tags": {"type": "array", "items": {"type": "string"}},
                    "energy": {"type": "number"},
                    "danceability": {"type": "number"},
                    "acousticness": {"type": "number"},
                    "decade": {"type": "string"},
                    "min_popularity": {"type": "number"},
                },
                "additionalProperties": False,
            },
            "k": {
                "type": "integer",
                "description": "How many songs to return (default 5).",
                "default": 5,
            },
        },
        "required": ["mode", "user_prefs"],
    },
}


# ---------------------------------------------------------------------------
# Mock parser (used by --mock and as fallback when Claude is unavailable)
# ---------------------------------------------------------------------------

_MOCK_KEYWORD_MAP: List[Tuple[List[str], Dict]] = [
    (
        ["study", "studying", "homework", "reading", "focus", "deep work", "concentrate"],
        {"mood": "focused", "energy": 0.35, "mood_tags": ["study", "mellow", "focus"], "mode": "mood_first"},
    ),
    (
        ["chill", "relax", "calm", "mellow", "wind down", "rainy", "late night", "late-night"],
        {"mood": "chill", "energy": 0.3, "mood_tags": ["mellow", "late-night", "calm"], "mode": "mood_first"},
    ),
    (
        ["sleep", "meditat", "ambient", "dream"],
        {"mood": "calm", "energy": 0.2, "mood_tags": ["sleep", "meditative"], "mode": "mood_first"},
    ),
    (
        ["workout", "gym", "pump", "run", "running", "hype", "lift", "training"],
        {"mood": "intense", "energy": 0.92, "mood_tags": ["workout", "hype", "driving"], "mode": "energy_similarity"},
    ),
    (
        ["metal", "heavy", "headbang"],
        {"genre": "metal", "mood": "intense", "energy": 0.9, "mood_tags": ["aggressive", "heavy"], "mode": "genre_first"},
    ),
    (
        ["rock"],
        {"genre": "rock", "mood": "intense", "energy": 0.85, "mood_tags": ["driving", "aggressive"], "mode": "genre_first"},
    ),
    (
        ["pop"],
        {"genre": "pop", "mood": "happy", "energy": 0.8, "mood_tags": ["uplifting", "feel-good"], "mode": "genre_first"},
    ),
    (
        ["lofi", "lo-fi", "lo fi"],
        {"genre": "lofi", "mood": "chill", "energy": 0.4, "mood_tags": ["study", "mellow"], "mode": "genre_first"},
    ),
    (
        ["jazz"],
        {"genre": "jazz", "mood": "relaxed", "energy": 0.4, "mood_tags": ["acoustic", "cafe"], "mode": "genre_first"},
    ),
    (
        ["classical", "piano", "strings"],
        {"genre": "classical", "mood": "calm", "energy": 0.2, "mood_tags": ["acoustic", "reading"], "mode": "genre_first"},
    ),
    (
        ["r&b", "rnb", "soul", "romantic", "dinner", "candle", "evening", "slow dance"],
        {"genre": "r&b", "mood": "romantic", "energy": 0.5, "mood_tags": ["romantic", "evening"], "mode": "mood_first"},
    ),
    (
        ["hip hop", "hip-hop", "rap"],
        {"genre": "hip hop", "mood": "focused", "energy": 0.7, "mood_tags": ["urban", "deep-work"], "mode": "genre_first"},
    ),
    (
        ["world", "global", "cultural", "uplifting"],
        {"genre": "world", "mood": "uplifting", "energy": 0.6, "mood_tags": ["global", "cultural", "uplifting"], "mode": "mood_first"},
    ),
    (
        ["synthwave", "neon", "night drive", "retro"],
        {"genre": "synthwave", "mood": "moody", "energy": 0.7, "mood_tags": ["nostalgic", "late-night", "driving"], "mode": "genre_first"},
    ),
]

_DECADE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b80s\b|\b1980s\b|\beighties\b", "1980s"),
    (r"\b90s\b|\b1990s\b|\bnineties\b", "1990s"),
    (r"\b2000s\b|\baughts\b", "2000s"),
    (r"\b2010s\b", "2010s"),
    (r"\b2020s\b|\bmodern\b|\bcurrent\b", "2020s"),
    (r"\bold[- ]school\b|\bold school\b|\bvintage\b|\bnostalgic\b|\bclassic\b", "1990s"),
]


def mock_plan(query: str) -> Dict:
    """Rule-based intent parser used by ``--mock`` mode and as a fallback.

    Returns a plan dict shaped exactly like the live Claude planner so the
    rest of the pipeline does not branch on mode.
    """
    q = (query or "").lower()
    plan: Dict = {
        "user_prefs": {},
        "weights": {},
        "mode": "genre_first",
        "rationale": "rule-based mock parser",
    }

    matched_signals: List[str] = []

    for keywords, payload in _MOCK_KEYWORD_MAP:
        if any(kw in q for kw in keywords):
            matched_signals.append(payload.get("mode", ""))
            for k, v in payload.items():
                if k == "mode":
                    plan["mode"] = v
                else:
                    if k == "mood_tags":
                        existing = set(plan["user_prefs"].get("mood_tags", []))
                        existing.update(v)
                        plan["user_prefs"]["mood_tags"] = sorted(existing)
                    else:
                        plan["user_prefs"][k] = v

    for pattern, decade in _DECADE_PATTERNS:
        if re.search(pattern, q):
            plan["user_prefs"]["decade"] = decade
            break

    if not plan["user_prefs"]:
        # Empty fallback: default to a moderate mood-first profile.
        plan["user_prefs"] = {
            "mood_tags": ["mellow"],
            "energy": 0.5,
        }
        plan["mode"] = "mood_first"
        plan["rationale"] = "no clear signals; defaulting to neutral mood-first plan"

    # Pick a mode by majority signal if multiple matched
    if matched_signals:
        # frequency vote ignoring blanks
        counts: Dict[str, int] = {}
        for m in matched_signals:
            if m:
                counts[m] = counts.get(m, 0) + 1
        if counts:
            plan["mode"] = max(counts, key=counts.get)

    return plan


def mock_explanations(recs: List[Dict], plan: Dict) -> List[str]:
    """Generate per-song explanations from the score breakdown.

    The mock mode reuses the breakdown so explanations stay consistent
    with the math by construction (the guardrail still runs as a check).
    """
    explanations: List[str] = []
    for r in recs:
        song = r["song"]
        breakdown = r.get("breakdown", {})
        sorted_terms = sorted(
            ((d, v) for d, v in breakdown.items() if abs(v) >= 0.1),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        if sorted_terms:
            phrases = []
            # Top 4 keeps explanations focused while still mentioning any
            # dominant dimension above the missed-mention floor.
            for dim, val in sorted_terms[:4]:
                phrases.append(_dimension_to_phrase(dim, val, song, plan))
            text = (
                f"{song['title']} by {song['artist']}: "
                + "; ".join(phrases)
                + "."
            )
        else:
            text = (
                f"{song['title']} by {song['artist']}: weak overall match, "
                "included only because the catalog is small."
            )
        explanations.append(text)
    return explanations


def _dimension_to_phrase(dim: str, value: float, song: Dict, plan: Dict) -> str:
    """Translate a (dimension, contribution) pair into human English.

    Wording is carefully chosen to avoid keyword overlap with the
    guardrail's per-dimension regexes (e.g. we do not say "vibe" in the
    mood phrase because "vibe" is a mood_tags signal).
    """
    prefs = plan.get("user_prefs", {})
    if dim == "genre":
        return f"genre matches your '{song['genre']}' request (+{value:.2f})"
    if dim == "mood":
        return f"mood matches your '{song['mood']}' preference (+{value:.2f})"
    if dim == "mood_tags":
        overlap = sorted(set(song.get("mood_tags") or []) & set(prefs.get("mood_tags") or []))
        tag_str = ", ".join(overlap) if overlap else "shared mood tags"
        return f"shared mood tags [{tag_str}] (+{value:.2f})"
    if dim == "energy":
        return f"energy {song.get('energy', 0):.2f} sits close to your target (+{value:.2f})"
    if dim == "danceability":
        return f"danceability {song.get('danceability', 0):.2f} aligns with the request (+{value:.2f})"
    if dim == "acousticness":
        return f"acousticness {song.get('acousticness', 0):.2f} matches your acoustic preference (+{value:.2f})"
    if dim == "popularity":
        return f"popularity {song.get('popularity', 0):.0f}/100 (+{value:.2f})"
    if dim == "decade":
        return f"released in the {song.get('release_decade', '?')} era you asked for (+{value:.2f})"
    if dim == "artist_penalty":
        return f"diversity penalty applied ({value:+.2f})"
    return f"{dim} contributed {value:+.2f}"


# ---------------------------------------------------------------------------
# Live Claude planner (uses tool-use API)
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """You are the planning brain of a music recommender.
You receive a user's natural-language vibe request plus retrieved context
from a small local knowledge base about genres, moods, decades, and artists.

Your job is to call the `recommend_songs` tool exactly once with:
- `mode`: one of {modes}
- `user_prefs`: an object with the parsed targets (mood, mood_tags, energy,
  decade, etc.). Only include fields you are confident about.
- `weights`: optional small overrides on the mode preset.

After the tool returns the top recommendations, write ONE final assistant
message containing a JSON array under the key `explanations`. Each item:
{{"id": <song_id>, "text": "<one or two sentence explanation>"}}

Rules:
- Mention only score dimensions that actually drove the recommendation.
- Keep each explanation under 220 characters.
- Do not invent songs. Use only the ones returned by the tool.
- Output ONLY valid JSON in your final message, no surrounding prose.
"""


def _format_planner_user_message(query: str, kb_block: str, song_count: int) -> str:
    return (
        f"User vibe request: {query!r}\n\n"
        f"Retrieved knowledge-base context (top results):\n{kb_block}\n\n"
        f"Catalog size: {song_count} songs. Pick a mode, set weights, and "
        f"call recommend_songs."
    )


def _live_plan_and_explain(
    query: str,
    kb_block: str,
    songs: List[Dict],
    k: int,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Run the Claude tool-use loop. Returns (plan, recs, explanations).

    Each iteration of the loop is printed for observability. Raises if
    Anthropic SDK / API key is unavailable so the caller can fall back to
    mock mode.
    """
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError("anthropic package not installed") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = Anthropic(api_key=api_key)
    logger = get_logger()

    system = _PLANNER_SYSTEM_PROMPT.format(modes=list(RANKING_MODES.keys()))
    messages: List[Dict] = [
        {
            "role": "user",
            "content": _format_planner_user_message(query, kb_block, len(songs)),
        }
    ]

    plan: Dict = {}
    recs: List[Dict] = []
    explanations: List[Dict] = []

    for turn in range(4):  # safety cap on tool-use loop
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            tools=[RECOMMEND_TOOL_SCHEMA],
            messages=messages,
        )
        logger.info("claude turn=%d stop_reason=%s", turn, resp.stop_reason)

        # Print any text (planning thoughts) for the Loom recording
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                txt = block.text.strip()
                if txt:
                    print(f"      AGENT THOUGHT: {txt[:300]}")

        # Append the assistant message back into the conversation as required by the API
        assistant_content = [
            _block_to_dict(b) for b in resp.content
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use" and block.name == "recommend_songs":
                    tool_input = block.input or {}
                    plan = {
                        "mode": tool_input.get("mode", "genre_first"),
                        "user_prefs": tool_input.get("user_prefs", {}),
                        "weights": tool_input.get("weights", {}),
                        "rationale": "claude tool-use call",
                    }
                    print(
                        f"      AGENT TOOL CALL: recommend_songs(mode={plan['mode']}, "
                        f"prefs_keys={list(plan['user_prefs'].keys())})"
                    )
                    recs = recommend_songs_weighted(
                        plan["user_prefs"],
                        songs,
                        k=int(tool_input.get("k", k)),
                        weights=plan["weights"],
                        mode=plan["mode"],
                    )
                    summary = _summarise_recs_for_tool_result(recs)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(summary),
                        }
                    )
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                continue
            break

        if resp.stop_reason == "end_turn":
            # Final assistant message should contain explanations JSON
            text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            full_text = "\n".join(text_blocks).strip()
            explanations = _parse_explanations(full_text)
            break

    return plan, recs, explanations


def _block_to_dict(block) -> Dict:
    """Convert an Anthropic SDK content block back into a serialisable dict.

    The SDK accepts the same dict shape it returns, so re-serialising is
    safe for sending back in the next ``messages.create`` call.
    """
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": getattr(block, "tool_use_id", None),
            "content": getattr(block, "content", ""),
        }
    return {"type": btype or "text", "text": str(block)}


def _summarise_recs_for_tool_result(recs: List[Dict]) -> Dict:
    """Compact JSON summary of the recommender output for the agent."""
    return {
        "recommendations": [
            {
                "id": r["song"]["id"],
                "title": r["song"]["title"],
                "artist": r["song"]["artist"],
                "genre": r["song"]["genre"],
                "mood": r["song"]["mood"],
                "energy": r["song"]["energy"],
                "popularity": r["song"].get("popularity"),
                "release_decade": r["song"].get("release_decade"),
                "score": r["score"],
                "breakdown": {
                    k: round(float(v), 3) for k, v in r["breakdown"].items()
                },
                "mode": r["mode"],
            }
            for r in recs
        ]
    }


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


def _parse_explanations(text: str) -> List[Dict]:
    """Pull the explanations array out of Claude's final text message."""
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict) and "explanations" in parsed:
        items = parsed["explanations"]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    out: List[Dict] = []
    for it in items:
        if isinstance(it, dict) and "text" in it:
            out.append({"id": it.get("id"), "text": str(it["text"])})
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    query: str,
    songs: List[Dict],
    kb: KnowledgeBase,
    *,
    k: int = 5,
    use_mock: bool = False,
    verbose: bool = True,
) -> AgentTrace:
    """Execute the full 5-step agent pipeline and return a structured trace.

    Set ``use_mock=True`` to run without an API key. Either way the steps
    are printed when ``verbose=True`` so a Loom recording captures the
    intermediate reasoning.
    """
    logger = get_logger()
    trace = AgentTrace(query=query, valid=False, validation_reason="")

    # ---- Step 1: validate ------------------------------------------------
    if verbose:
        print("[STEP 1/5] Validating user input ...")
    ok, reason = validate_user_input(query)
    trace.valid = ok
    trace.validation_reason = reason
    logger.info("validation query=%r ok=%s reason=%s", query, ok, reason)
    if not ok:
        if verbose:
            print(f"      INPUT REJECTED: {reason}")
        trace.error = reason
        trace.mode = "error"
        return trace
    if verbose:
        print(f"      OK ({reason})")

    # ---- Step 2: retrieve KB --------------------------------------------
    if verbose:
        print("[STEP 2/5] Retrieving context from RAG knowledge base ...")
    kb_results = kb.retrieve(query, k=4)
    trace.kb_results = kb_results
    logger.info("kb hits=%d for query=%r", len(kb_results), query)
    if verbose:
        if kb_results:
            for r in kb_results:
                print(f"      RETRIEVED [{r['type']}:{r['name']}] score={r['score']}")
        else:
            print("      (no relevant context retrieved; agent will rely on the query alone)")
    kb_block = kb.format_for_prompt(kb_results)

    # ---- Step 3: plan + assign weights ----------------------------------
    if verbose:
        print("[STEP 3/5] Planning + assigning scoring weights ...")
    plan: Dict = {}
    recs: List[Dict] = []
    raw_explanations: List[Dict] = []

    used_live = False
    if not use_mock:
        try:
            plan, recs, raw_explanations = _live_plan_and_explain(
                query, kb_block, songs, k=k
            )
            used_live = True
            trace.mode = "agent"
        except Exception as e:  # noqa: BLE001 (broad fallback is intentional)
            logger.warning("live agent failed (%s); falling back to mock", e)
            if verbose:
                print(f"      live agent unavailable ({e}); falling back to mock parser")

    if not used_live:
        plan = mock_plan(query)
        trace.mode = "mock"
        if verbose:
            print(
                f"      PLAN: mode={plan['mode']} prefs={json.dumps(plan['user_prefs'], sort_keys=True)}"
            )

        # ---- Step 4: tool call (mock path) ------------------------------
        if verbose:
            print("[STEP 4/5] Tool call: recommend_songs(...)")
        recs = recommend_songs_weighted(
            plan["user_prefs"],
            songs,
            k=k,
            weights=plan.get("weights"),
            mode=plan.get("mode"),
        )
        if verbose:
            for r in recs:
                print(
                    f"      -> {r['song']['title']} by {r['song']['artist']} "
                    f"[{r['mode']}] score={r['score']:.2f}"
                )

        # ---- Step 5: explanations (mock path) ---------------------------
        if verbose:
            print("[STEP 5/5] Generating per-song explanations ...")
        explanation_texts = mock_explanations(recs, plan)
        for rec, text in zip(recs, explanation_texts):
            rec["explanation"] = text
    else:
        if verbose:
            print(
                f"      PLAN: mode={plan.get('mode')} "
                f"prefs={json.dumps(plan.get('user_prefs', {}), sort_keys=True)}"
            )
            print("[STEP 4/5] Tool call: recommend_songs(...)")
            for r in recs:
                print(
                    f"      -> {r['song']['title']} by {r['song']['artist']} "
                    f"[{r['mode']}] score={r['score']:.2f}"
                )
            print("[STEP 5/5] Generating per-song explanations ...")
        # Map Claude explanations onto the rec list by song id; fall back to
        # the deterministic mock explanation if Claude missed an entry.
        by_id: Dict[int, str] = {}
        for item in raw_explanations:
            if item.get("id") is not None:
                try:
                    by_id[int(item["id"])] = item["text"]
                except (TypeError, ValueError):
                    continue
        fallback_texts = mock_explanations(recs, plan)
        for rec, fallback in zip(recs, fallback_texts):
            sid = rec["song"]["id"]
            rec["explanation"] = by_id.get(sid, fallback)

    # ---- Guardrail: explanation vs score consistency --------------------
    flags_by_id: Dict[str, List[str]] = {}
    for rec in recs:
        sid = str(rec["song"]["id"])
        flags = check_explanation_consistency(rec.get("explanation", ""), rec.get("breakdown", {}))
        if flags:
            flags_by_id[sid] = flags
        log_guardrail_results(
            label=f"{rec['song']['title']} (id={sid})",
            flags=flags,
        )
    trace.guardrail_flags = flags_by_id
    if verbose and flags_by_id:
        print("      GUARDRAIL FLAGS:")
        for sid, flags in flags_by_id.items():
            for f in flags:
                print(f"        - song {sid}: {f}")
    elif verbose:
        print("      GUARDRAIL: explanations consistent with score breakdowns")

    trace.plan = plan
    trace.recommendations = recs
    return trace


# ---------------------------------------------------------------------------
# Convenience: load knowledge base from default path
# ---------------------------------------------------------------------------


def load_default_kb(path: str | Path = "data/knowledge_base.json") -> KnowledgeBase:
    """Load the bundled knowledge base."""
    return KnowledgeBase.from_json(path)
