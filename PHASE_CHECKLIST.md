# Phase Checklist — Music Recommender

Update the STATUS line before every Composer session.

## Phase 1 & 2: Setup + Design
- [ ] Repo forked and cloned
- [x] songs.csv reviewed — columns understood
- [x] Added 5-10 new songs to songs.csv (diverse genres and moods beyond the starter 10)
- [x] README.md "How The System Works" section written
- [x] Algorithm recipe documented in README.md

## Phase 3: Implementation
- [x] load_songs() works — `python -m src.main` prints songs loading
- [x] score_song() returns (float, List[str]) — reasons only when points scored
- [x] recommend_songs() returns List of (song_dict, score, explanation_str) 3-tuples
- [x] Recommender.recommend() implemented — sorted by score, passes pytest
- [x] Recommender.explain_recommendation() returns non-empty string, passes pytest
- [x] main.py defines 3 user profiles: "High-Energy Pop", "Chill Lofi", "Deep Intense Rock"
- [x] CLI output is clean: title, score, reasons printed for each profile
- [x] Docstrings on all functions
- [ ] Committed with meaningful message

## Phase 4: Evaluation
- [x] Ran all 3 profiles, took terminal screenshots
- [x] Screenshots added to README.md
- [x] Ran at least one experiment (e.g. doubled energy weight, halved genre weight)
- [x] Experiment results noted in README.md "Experiments You Tried" section
- [x] model_card.md Limitations section filled out (3-5 sentences on genre bias + dataset size)

## Phase 5: Model Card + Reflection
- [x] model_card.md all 9 sections complete
- [x] README.md Reflection section written (2 paragraphs)
- [ ] Repo is public
- [ ] All changes pushed

---

## Current Status
> Update this before each session.
**STATUS: Phase 5 — documentation complete; pending screenshots/push**
