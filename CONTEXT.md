# Music Recommender Simulation — Project Context

## What I'm Building
A content-based music recommendation system in Python. It takes a user taste profile and scores/ranks songs from a CSV catalog by relevance. This is a graded submission — it needs to be clean, documented, and explainable.

## Actual Project Structure (from starter repo)
```
music-recommender-simulation-starter/
├── data/
│   └── songs.csv                  # 10 songs with: id, title, artist, genre, mood,
│                                  # energy, tempo_bpm, valence, danceability, acousticness
├── src/
│   ├── recommender.py             # Core logic — Song, UserProfile, Recommender class,
│                                  # plus load_songs, score_song, recommend_songs functions
│   └── main.py                    # CLI runner — calls load_songs + recommend_songs, prints output
├── tests/
│   └── test_recommender.py        # Tests the OOP layer (Recommender class)
├── model_card.md                  # Documents how the system works, limits, bias (9 sections)
└── README.md                      # System explanation + screenshots + experiments
```

## Two Parallel Implementation Layers
The starter has BOTH a functional layer AND an OOP layer. Both need to be implemented.

### Functional layer (used by main.py)
- `load_songs(csv_path)` → List[Dict]
- `score_song(user_prefs: Dict, song: Dict)` → Tuple[float, List[str]]
- `recommend_songs(user_prefs, songs, k=5)` → List[Tuple[Dict, float, str]]

### OOP layer (used by tests)
- `Song` dataclass — already defined, do not change field names
- `UserProfile` dataclass — already defined, do not change field names
- `Recommender` class with:
  - `recommend(user: UserProfile, k=5)` → List[Song], sorted by score descending
  - `explain_recommendation(user: UserProfile, song: Song)` → str (non-empty)

## UserProfile Shape (from starter)
```python
UserProfile(
    favorite_genre="pop",
    favorite_mood="happy",
    target_energy=0.8,
    likes_acoustic=False
)
```

## main.py User Profile Shape (Dict, different from OOP)
```python
user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
```

## main.py Expected Output Format
```python
for rec in recommendations:
    song, score, explanation = rec   # must be a 3-tuple
    print(f"{song['title']} - Score: {score:.2f}")
    print(f"Because: {explanation}")
```

## songs.csv Columns
id, title, artist, genre, mood, energy (float), tempo_bpm (int), valence (float), danceability (float), acousticness (float)

## Run Commands
```bash
python -m src.main     # runs the CLI
pytest                 # runs tests/test_recommender.py
```

## What the Grader Checks
- score_song returns (float, List[str]) — reasons only appear when points scored
- recommend_songs returns List of 3-tuples: (song_dict, score, explanation_string)
- Recommender.recommend() returns songs sorted by score, pop/happy song ranked first in tests
- Recommender.explain_recommendation() returns a non-empty string
- At least 3 distinct user profiles tested in main.py
- model_card.md all 9 sections filled out
- README.md has terminal screenshots for each profile
- Multiple meaningful git commits
