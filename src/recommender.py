"""Core recommender for the Music Recommender Simulation (Project 3 + Project 4).

Provides two parallel layers used by different parts of the codebase:

- Functional layer: ``load_songs``, ``score_song``, ``recommend_songs``.
  Used by ``src/main.py`` and the agent tool. Backwards-compatible with the
  Project 3 starter signature.
- OOP layer: ``Song``, ``UserProfile``, ``Recommender``. Used by
  ``tests/test_recommender.py``. Signatures and default scoring preserved.

Project 4 extensions added here:

- New attributes ``popularity``, ``release_decade``, ``mood_tags`` consumed
  alongside the existing ``danceability``, ``acousticness``, ``tempo_bpm``,
  and ``valence`` columns.
- A weighted scorer ``score_song_weighted`` driven by an optional weights
  dict so the agent can dial dimensions up or down per query.
- Three named ranking modes (``genre_first``, ``mood_first``,
  ``energy_similarity``) for the multi-mode stretch.
- An artist diversity penalty applied after scoring to fight filter bubbles.
"""
from typing import List, Dict, Tuple, Optional, Iterable
from dataclasses import dataclass, field
import csv


# ---------------------------------------------------------------------------
# Dataclasses (OOP layer required by tests)
# ---------------------------------------------------------------------------

@dataclass
class Song:
    """Represents a song and its attributes. Required by tests/test_recommender.py."""

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Project 4 extensions: optional with safe defaults so older test
    # constructors that omit them keep working.
    popularity: float = 50.0
    release_decade: str = ""
    mood_tags: List[str] = field(default_factory=list)


@dataclass
class UserProfile:
    """Represents a user's taste preferences. Required by tests/test_recommender.py."""

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ---------------------------------------------------------------------------
# Ranking mode presets (Project 3 stretch)
# ---------------------------------------------------------------------------
#
# Each preset is a full weight vector over the dimensions the weighted scorer
# understands. Presets are tuned so each mode produces visibly different
# rankings on the same query.
RANKING_MODES: Dict[str, Dict[str, float]] = {
    "genre_first": {
        "genre": 3.0,
        "mood": 1.0,
        "mood_tags": 0.5,
        "energy": 1.0,
        "danceability": 0.3,
        "acousticness": 0.3,
        "popularity": 0.3,
        "decade": 0.5,
    },
    "mood_first": {
        "genre": 1.0,
        "mood": 3.0,
        "mood_tags": 1.5,
        "energy": 1.0,
        "danceability": 0.5,
        "acousticness": 0.5,
        "popularity": 0.3,
        "decade": 0.3,
    },
    "energy_similarity": {
        "genre": 0.5,
        "mood": 0.5,
        "mood_tags": 0.5,
        "energy": 3.0,
        "danceability": 1.5,
        "acousticness": 0.3,
        "popularity": 0.3,
        "decade": 0.3,
    },
}


def default_weights() -> Dict[str, float]:
    """Return a copy of the genre-first preset (sensible neutral default)."""
    return dict(RANKING_MODES["genre_first"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tag_list(value: object) -> List[str]:
    """Parse a semicolon-separated tag string from CSV into a clean list."""
    if isinstance(value, list):
        return [str(t).strip().lower() for t in value if str(t).strip()]
    if not value:
        return []
    return [t.strip().lower() for t in str(value).split(";") if t.strip()]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity between two iterables of strings (case-insensitive)."""
    sa = {str(x).strip().lower() for x in a if str(x).strip()}
    sb = {str(x).strip().lower() for x in b if str(x).strip()}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _decade_match(song_decades: str, target_decade: Optional[str]) -> float:
    """Return 1.0 if any song decade matches target_decade, else 0.0.

    ``release_decade`` may be a single decade ("2020s") or multiple separated
    by ``;`` so songs that span eras can still match across them.
    """
    if not target_decade or not song_decades:
        return 0.0
    parts = {p.strip().lower() for p in str(song_decades).split(";") if p.strip()}
    return 1.0 if str(target_decade).strip().lower() in parts else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file, coercing numeric fields and tag lists.

    Backwards compatible with the Project 3 schema. New columns
    (``popularity``, ``release_decade``, ``mood_tags``) are optional; missing
    values fall back to safe defaults so old CSVs still load.
    """
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            row["popularity"] = float(row.get("popularity") or 50.0)
            row["release_decade"] = (row.get("release_decade") or "").strip()
            row["mood_tags"] = _parse_tag_list(row.get("mood_tags"))
            songs.append(row)
    return songs


def songs_dicts_to_objects(songs: List[Dict]) -> List[Song]:
    """Convert dict-format songs (from load_songs) into Song dataclass objects."""
    out: List[Song] = []
    for s in songs:
        out.append(
            Song(
                id=int(s["id"]),
                title=s["title"],
                artist=s["artist"],
                genre=s["genre"],
                mood=s["mood"],
                energy=float(s["energy"]),
                tempo_bpm=float(s["tempo_bpm"]),
                valence=float(s["valence"]),
                danceability=float(s["danceability"]),
                acousticness=float(s["acousticness"]),
                popularity=float(s.get("popularity", 50.0)),
                release_decade=str(s.get("release_decade", "")),
                mood_tags=_parse_tag_list(s.get("mood_tags")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Project 3 starter scorer (kept as-is for backwards compatibility / tests)
# ---------------------------------------------------------------------------

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song against simple user preferences (Project 3 contract).

    Inputs use the original P3 keys (``genre``, ``mood``, ``energy``).
    Returns ``(score, reasons)`` exactly as Project 3 expected so existing
    callers and graders keep working.
    """
    score = 0.0
    reasons: List[str] = []

    if song["genre"] == user_prefs["genre"]:
        score += 2.0
        reasons.append("genre match (+2.0)")

    if song["mood"] == user_prefs["mood"]:
        score += 1.0
        reasons.append("mood match (+1.0)")

    energy_score = round(1.0 - abs(song["energy"] - user_prefs["energy"]), 2)
    score += energy_score
    reasons.append(f"energy proximity (+{energy_score:.2f})")

    return score, reasons


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """Rank songs by ``score_song`` (Project 3 contract).

    Returns a list of ``(song_dict, score, explanation_string)`` 3-tuples
    sorted by score descending, truncated to ``k``.
    """
    scored: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Project 4 weighted scorer (used by the agent + ranking modes)
# ---------------------------------------------------------------------------

def score_song_weighted(
    user_prefs: Dict,
    song: Dict,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Score a song using a flexible weight dict and the extended attributes.

    ``user_prefs`` may include any subset of:
        ``genre``, ``mood``, ``mood_tags`` (list), ``energy``,
        ``danceability``, ``acousticness``, ``decade``, ``min_popularity``.

    Returns ``(total_score, per_dimension_breakdown, reason_strings)``. The
    per-dimension breakdown is what the output guardrail uses to verify that
    the agent's natural-language explanation actually reflects the math.
    """
    if weights is None:
        weights = default_weights()

    breakdown: Dict[str, float] = {}
    reasons: List[str] = []

    # Genre (exact match, weighted)
    if user_prefs.get("genre") and song.get("genre") == user_prefs["genre"]:
        contrib = float(weights.get("genre", 0.0))
        breakdown["genre"] = contrib
        reasons.append(f"genre match {song['genre']} (+{contrib:.2f})")
    else:
        breakdown["genre"] = 0.0

    # Mood (exact match, weighted)
    if user_prefs.get("mood") and song.get("mood") == user_prefs["mood"]:
        contrib = float(weights.get("mood", 0.0))
        breakdown["mood"] = contrib
        reasons.append(f"mood match {song['mood']} (+{contrib:.2f})")
    else:
        breakdown["mood"] = 0.0

    # Mood tag overlap (Jaccard similarity, weighted)
    requested_tags = user_prefs.get("mood_tags") or []
    song_tags = song.get("mood_tags") or []
    if requested_tags and song_tags:
        sim = _jaccard(requested_tags, song_tags)
        contrib = sim * float(weights.get("mood_tags", 0.0))
        breakdown["mood_tags"] = contrib
        if contrib > 0:
            reasons.append(
                f"mood-tag overlap {sim:.2f} (+{contrib:.2f})"
            )
    else:
        breakdown["mood_tags"] = 0.0

    # Energy proximity
    if "energy" in user_prefs and song.get("energy") is not None:
        prox = round(1.0 - abs(float(song["energy"]) - float(user_prefs["energy"])), 2)
        contrib = prox * float(weights.get("energy", 0.0))
        breakdown["energy"] = contrib
        reasons.append(f"energy proximity {prox:.2f} (+{contrib:.2f})")
    else:
        breakdown["energy"] = 0.0

    # Danceability proximity
    if "danceability" in user_prefs and song.get("danceability") is not None:
        prox = round(
            1.0 - abs(float(song["danceability"]) - float(user_prefs["danceability"])), 2
        )
        contrib = prox * float(weights.get("danceability", 0.0))
        breakdown["danceability"] = contrib
        if contrib > 0:
            reasons.append(f"danceability proximity {prox:.2f} (+{contrib:.2f})")
    else:
        breakdown["danceability"] = 0.0

    # Acousticness proximity
    if "acousticness" in user_prefs and song.get("acousticness") is not None:
        prox = round(
            1.0 - abs(float(song["acousticness"]) - float(user_prefs["acousticness"])), 2
        )
        contrib = prox * float(weights.get("acousticness", 0.0))
        breakdown["acousticness"] = contrib
        if contrib > 0:
            reasons.append(f"acousticness proximity {prox:.2f} (+{contrib:.2f})")
    else:
        breakdown["acousticness"] = 0.0

    # Popularity (normalised 0-1)
    pop_norm = float(song.get("popularity", 50.0)) / 100.0
    contrib = pop_norm * float(weights.get("popularity", 0.0))
    breakdown["popularity"] = contrib
    if contrib > 0:
        reasons.append(f"popularity {pop_norm:.2f} (+{contrib:.2f})")

    # Decade match
    target_decade = user_prefs.get("decade")
    decade_hit = _decade_match(song.get("release_decade", ""), target_decade)
    contrib = decade_hit * float(weights.get("decade", 0.0))
    breakdown["decade"] = contrib
    if contrib > 0:
        reasons.append(f"decade match {target_decade} (+{contrib:.2f})")

    # Optional hard-ish filter: subtract a small amount when below minimum popularity
    min_pop = user_prefs.get("min_popularity")
    if isinstance(min_pop, (int, float)) and song.get("popularity", 50.0) < float(min_pop):
        penalty = -0.5
        breakdown["min_popularity_penalty"] = penalty
        reasons.append(f"below requested popularity ({penalty:+.2f})")

    total = round(sum(breakdown.values()), 4)
    return total, breakdown, reasons


def recommend_songs_weighted(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    mode: Optional[str] = None,
    artist_penalty: float = 0.5,
) -> List[Dict]:
    """Score, sort, and diversify songs using the weighted scorer.

    If ``mode`` is set, its preset weights are merged on top of ``weights``
    so the agent can pick a mode and still nudge individual dimensions.

    Returns a list of dicts (one per recommended song) shaped for the agent
    and the display layer:

        {"song", "score", "breakdown", "reasons", "mode"}
    """
    effective: Dict[str, float] = default_weights()
    if mode and mode in RANKING_MODES:
        effective.update(RANKING_MODES[mode])
    if weights:
        effective.update({k_: float(v) for k_, v in weights.items()})

    scored: List[Dict] = []
    for song in songs:
        total, breakdown, reasons = score_song_weighted(user_prefs, song, effective)
        scored.append(
            {
                "song": song,
                "score": total,
                "breakdown": breakdown,
                "reasons": reasons,
                "mode": mode or "custom",
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    diversified = apply_artist_penalty(scored, penalty=artist_penalty)
    return diversified[:k]


# ---------------------------------------------------------------------------
# Diversity / fairness (Project 3 stretch)
# ---------------------------------------------------------------------------

def apply_artist_penalty(
    ranked: List[Dict],
    penalty: float = 0.5,
) -> List[Dict]:
    """Penalise repeat artists in a ranked list to fight filter bubbles.

    For each occurrence of an artist beyond the first, subtract ``penalty``
    from that song's score, then re-sort. The original score is preserved in
    ``original_score`` for transparency, and a reason string is appended so
    the user can see when diversity adjusted the ordering.
    """
    seen: Dict[str, int] = {}
    for item in ranked:
        artist = (item.get("song") or {}).get("artist", "")
        count = seen.get(artist, 0)
        item["original_score"] = item.get("score")
        if count > 0:
            adjustment = -penalty * count
            item["score"] = round(item["score"] + adjustment, 4)
            item.setdefault("reasons", []).append(
                f"artist diversity penalty ({adjustment:+.2f}, {artist} seen {count}x)"
            )
            item.setdefault("breakdown", {})["artist_penalty"] = adjustment
        seen[artist] = count + 1

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# OOP Recommender (used by tests; kept stable)
# ---------------------------------------------------------------------------

class Recommender:
    """OOP wrapper required by ``tests/test_recommender.py``.

    ``_score`` keeps the Project 3 formula so the test asserting that a
    pop/happy song outranks a lofi/chill song for a pop/happy user still
    passes. ``_score_weighted`` is the Project 4 path used when an explicit
    weights dict is supplied.
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 2.0
        if song.mood == user.favorite_mood:
            score += 1.0
        score += round(1.0 - abs(song.energy - user.target_energy), 2)
        if user.likes_acoustic and song.acousticness > 0.6:
            score += 0.5
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top-k songs sorted by score descending."""
        return sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation string for one (user, song) pair."""
        reasons: List[str] = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre match ({song.genre})")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood match ({song.mood})")
        energy_gap = round(abs(song.energy - user.target_energy), 2)
        reasons.append(f"energy proximity (gap: {energy_gap})")
        if user.likes_acoustic and song.acousticness > 0.6:
            reasons.append(f"acoustic bonus ({song.acousticness:.2f})")
        return ", ".join(reasons) if reasons else "no strong match"
