# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

**VibeMatch Classroom Recommender v1**

---

## 2. Intended Use

This system recommends top songs from a small local catalog based on user preferences for genre, mood, and energy level. It is intended for classroom exploration of recommendation logic, not for production use with real users.

It assumes users can be represented by a simple taste profile and that song metadata is accurate. The model is designed for transparency and learning, not personalization at commercial scale.

---

## 3. How It Works (Short Explanation)

Each song receives a score based on how closely it matches the user profile. Genre match contributes the largest weight, mood match contributes a smaller bonus, and energy proximity contributes a continuous value based on distance from the user target energy.

The system ranks all songs by score and returns the top results. In the OOP layer, users who prefer acoustic music can get a small bonus for songs with higher acousticness.

Because scoring is rule-based and not learned from user behavior, recommendations are easy to explain but less adaptive than modern machine learning recommenders.

---

## 4. Data

The catalog contains **15 songs** stored in `data/songs.csv`. It includes multiple genres and moods such as pop, rock, lofi, ambient, jazz, synthwave, indie pop, world, hip hop, classical, r&b, and metal.

I added 5 songs beyond the starter set to improve diversity across energy levels and styles. Even with those additions, the dataset is still very small and does not represent global musical taste.

This data mostly reflects a handcrafted sample rather than real listening populations, so results are useful for simulation but not for broad user claims.

---

## 5. Strengths

- Transparent scoring with clear reasons makes recommendations easy to audit.
- The model behaves predictably for users with clear genre/mood preferences.
- Energy proximity helps avoid rigid yes/no matching and gives smoother ranking.
- The same logic works in both functional and OOP implementations, making testing straightforward.

---

## 6. Limitations and Bias

This recommender is limited by catalog size and handcrafted metadata. With only 15 songs, many user tastes cannot be represented, which leads to repetitive or weak matches.

The fixed weights can over-prioritize genre and under-value other dimensions of music preference. If a user has cross-genre taste, the system may still push them toward a single dominant label.

Some genres, moods, and cultural music contexts are underrepresented in the dataset. In a real product, this could unfairly reduce exposure for less represented artists or listener communities.

The model also ignores important factors such as lyrics, language, release era, and listening context (study, workout, commute), which can bias recommendations toward simplistic interpretations of taste.

---

## 7. Evaluation

I evaluated the system with three profiles:

1. High-Energy Pop (`genre=pop`, `mood=happy`, `energy=0.9`)
2. Chill Lofi (`genre=lofi`, `mood=chill`, `energy=0.35`)
3. Deep Intense Rock (`genre=rock`, `mood=intense`, `energy=0.92`)

For each profile, I checked whether top songs matched expected genre/mood and had close energy values. The outputs were aligned with expectations and included readable reason strings.

I also ran unit tests in `tests/test_recommender.py` to confirm the OOP recommender returns sorted songs and non-empty explanations.

---

## 8. Future Work

- Add more songs and broader metadata coverage to reduce narrow-catalog bias.
- Learn weights from user feedback instead of fixed manual constants.
- Add diversity constraints so top-k results are not overly similar.
- Incorporate additional features like tempo ranges, valence bands, and lyric themes.
- Improve explanations by including confidence and trade-off details.

---

## 9. Personal Reflection

This project showed me how recommendation behavior can change dramatically from small adjustments in scoring weights. A simple formula can feel intelligent, but it still encodes design choices that may favor some users more than others.

I also learned that explainability is essential. Even in a toy system, showing why a song was recommended makes it easier to debug, trust, and critique the model. Building this changed how I view real music platforms: good recommendations are not only about prediction quality, but also about fairness, transparency, and representation.
