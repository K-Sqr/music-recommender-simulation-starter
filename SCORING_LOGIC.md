# Scoring Logic — Music Recommender

## score_song (functional layer — used by recommend_songs and main.py)

**Signature:** `score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]`

user_prefs keys: `"genre"`, `"mood"`, `"energy"`
song keys: `"genre"`, `"mood"`, `"energy"` (already converted to float by load_songs)

### Point Rules

| Condition | Points | Reason string to append |
|---|---|---|
| song["genre"] == user_prefs["genre"] | +2.0 | `"genre match (+2.0)"` |
| song["mood"] == user_prefs["mood"] | +1.0 | `"mood match (+1.0)"` |
| energy proximity (always runs) | 0.0 – 1.0 | `"energy proximity (+X.XX)"` |

### Energy Proximity Formula
```python
energy_score = round(1.0 - abs(song["energy"] - user_prefs["energy"]), 2)
```
Always add this to the score. Always append its reason string.

### Example
```python
score_song(
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    {"genre": "pop", "mood": "sad", "energy": 0.82, "title": "Sunrise City"}
)
# Returns: (3.98, ["genre match (+2.0)", "energy proximity (+0.98)"])
# mood did NOT match so no mood reason
```

---

## recommend_songs (functional layer)

**Signature:** `recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]`

```python
def recommend_songs(user_prefs, songs, k=5):
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = ", ".join(reasons)
        scored.append((song, score, explanation))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
```

Use `sorted()` not `.sort()` — it returns a new list without mutating the original.

---

## Recommender.recommend (OOP layer — used by tests)

UserProfile has: `favorite_genre`, `favorite_mood`, `target_energy`, `likes_acoustic`
Song has: `genre`, `mood`, `energy`, `acousticness` (all actual field names from dataclass)

### Scoring inside Recommender.recommend
```python
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
    """Returns top-k songs sorted by score descending."""
    return sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)[:k]
```

The test asserts `results[0].genre == "pop"` and `results[0].mood == "happy"` for a pop/happy/0.8 energy user — this scoring logic satisfies that.

---

## Recommender.explain_recommendation (OOP layer — used by tests)

Must return a non-empty string. List the matching attributes.

```python
def explain_recommendation(self, user: UserProfile, song: Song) -> str:
    """Returns a human-readable explanation of why a song was recommended."""
    reasons = []
    if song.genre == user.favorite_genre:
        reasons.append(f"genre match ({song.genre})")
    if song.mood == user.favorite_mood:
        reasons.append(f"mood match ({song.mood})")
    energy_gap = round(abs(song.energy - user.target_energy), 2)
    reasons.append(f"energy proximity (gap: {energy_gap})")
    return ", ".join(reasons) if reasons else "no strong match"
```

---

## load_songs

```python
import csv

def load_songs(csv_path: str) -> List[Dict]:
    """Loads songs from CSV, converting numeric fields to float/int."""
    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = int(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs
```
