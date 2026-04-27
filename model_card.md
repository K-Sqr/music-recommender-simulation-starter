# Model Card -- Music Recommender Simulation (Project 4)

## 1. Model name

**VibeMatch Agent v2** -- an agent-driven extension of the Project 3
content-based recommender ("VibeMatch Classroom Recommender v1").

## 2. Intended use

This system recommends songs from a small local catalog given a free-form
natural-language description of a user's vibe (e.g. *"I want something chill
for studying late at night"*). It is intended for classroom exploration of
recommender design, agentic workflows, and reliability guardrails -- not
for production music personalisation.

The system assumes a single user per session, a small static catalog, and
that song metadata is correct. It does not learn from feedback and does not
make claims about real-world listener preferences.

## 3. How the system works (plain language)

Each query travels through a 5-step agent pipeline:

1. **Validate input.** Empty, whitespace-only, single-word, overly long, or
   blocklisted queries are rejected before any AI call runs.
2. **Retrieve context.** A tiny TF-IDF retriever picks the top relevant
   notes from `data/knowledge_base.json` (genre, mood, decade, and context
   articles plus brief artist bios).
3. **Plan and weight.** The Claude agent (model `claude-sonnet-4-20250514`)
   reads the query plus the retrieved context and uses Anthropic's
   tool-use API to call the `recommend_songs` tool with a chosen ranking
   mode and per-dimension scoring weights. A deterministic mock parser
   serves the same role when the API key is unavailable.
4. **Score and rank.** The recommender computes a per-dimension breakdown
   for every song using a weighted formula across genre, mood, mood-tag
   overlap, energy proximity, danceability proximity, acousticness
   proximity, popularity, and release decade. An artist diversity penalty
   then subtracts 0.5 per repeat artist before the list is re-sorted.
5. **Explain and check.** Claude (or the mock template) writes per-song
   explanations from the breakdown. An output guardrail then parses each
   explanation and flags any dimension it mentions whose actual score
   contribution is below 0.10, plus any dimension above 1.0 that the
   explanation never mentioned.

Because the underlying scoring is rule-based, every recommendation is
auditable: you can trace exactly how many points each dimension
contributed for any (user query, song) pair.

## 4. Dataset

The catalog is `data/songs.csv` -- **20 songs** spanning pop, rock, lofi,
ambient, jazz, synthwave, indie pop, world, hip hop, classical, R&B, and
metal. Each song has the following attributes:

| Attribute | Type | Notes |
|---|---|---|
| `id`, `title`, `artist` | identifiers | |
| `genre`, `mood` | string | exact-match dimensions |
| `energy` | float 0-1 | proximity-scored |
| `tempo_bpm` | int | reference metadata |
| `valence` | float 0-1 | reference metadata |
| `danceability` | float 0-1 | proximity-scored when targeted |
| `acousticness` | float 0-1 | proximity-scored when targeted |
| `popularity` | float 0-100 | normalised, weighted into the score |
| `release_decade` | string ("1990s;2000s") | optional multi-decade match |
| `mood_tags` | semicolon list | scored via Jaccard overlap with requested tags |

The Project 3 catalog had 15 songs and only used `genre`, `mood`, and
`energy` in scoring (plus `acousticness` for a small bonus). Project 4
expands the dataset to 20 songs and brings five more attributes
(`popularity`, `release_decade`, `mood_tags`, `danceability`,
`acousticness`) into active scoring.

The knowledge base in `data/knowledge_base.json` adds **34 short
documents** describing genres, moods, decades, listening contexts, and
artist bios. These are retrieved per query and inserted into the agent's
planning prompt so its weight choices are grounded in real-ish musical
context, not just keyword matching.

## 5. Algorithmic approach

The scoring engine is a **weighted linear combination** of independent
similarity terms. For a song *s* and parsed user preferences *u*:

```
score(s, u) = w_genre * 1[s.genre == u.genre]
            + w_mood  * 1[s.mood  == u.mood]
            + w_mood_tags * jaccard(s.mood_tags, u.mood_tags)
            + w_energy * (1 - |s.energy - u.energy|)
            + w_danceability * (1 - |s.danceability - u.danceability|)
            + w_acousticness * (1 - |s.acousticness - u.acousticness|)
            + w_popularity * (s.popularity / 100)
            + w_decade * 1[u.decade in s.release_decade.split(';')]
```

The agent picks one of three named **ranking modes**, which are presets
over `w_*`:

- `genre_first` (boosts genre)
- `mood_first` (boosts mood + mood-tag overlap)
- `energy_similarity` (boosts energy + danceability)

Then it can override any individual `w_*` per query. After scoring, an
**artist diversity penalty** subtracts 0.5 from each song's score for every
prior occurrence of the same artist higher in the list, and the list is
re-sorted.

In short: rule-based scoring at the bottom, LLM-driven weight selection in
the middle, and post-hoc fairness adjustment at the top.

## 6. Strengths

- Every recommendation has a transparent per-dimension score breakdown,
  so explanations can be audited line-by-line.
- The agent's intermediate reasoning steps are visible in stdout, which
  makes both demos and debugging straightforward.
- Three named modes plus per-query weight overrides mean the same catalog
  produces visibly different rankings for different vibes.
- The artist diversity penalty pushes back on filter-bubble behaviour
  without hard-filtering otherwise-strong recommendations.
- A deterministic mock mode means the eval harness reproduces with no API
  key, which is essential for grading.

## 7. Limitations and biases

- **Catalog size (20 songs).** Many user tastes cannot be represented;
  some queries hit no good match and rely on energy proximity alone.
  This is the single biggest limitation.
- **Genre imbalance.** Rock, pop, and lofi each have multiple entries,
  while classical and hip hop have only two. Niche tastes get fewer
  options and therefore weaker top-K diversity.
- **Hand-curated metadata.** All `mood`, `mood_tags`, `popularity`, and
  `release_decade` values are author-assigned, not learned from real
  listener data. Errors and biases in those labels propagate directly
  into rankings.
- **Popularity bias.** A non-zero popularity weight in every preset means
  more popular songs are slightly favoured at the margin, which can
  systematically under-serve niche tracks.
- **LLM hallucination risk.** The Claude agent could in principle write an
  explanation that does not match the score breakdown. The output
  guardrail catches this for the dimensions it knows about, but it cannot
  catch every possible misclaim (e.g. a stylistic adjective that does not
  map to any scored dimension).
- **English-language assumption.** Both the validation regex and the KB
  documents assume English. Non-English vibe descriptions will retrieve
  no context and likely route through the fallback default plan.

## 8. Diversity, fairness, and filter bubbles

The recommender includes an **artist diversity penalty** that subtracts
0.5 per prior occurrence of the same artist from a song's score before the
final sort. This addresses two real concerns:

- **Filter-bubble exposure.** Without the penalty, a single artist with two
  on-vibe tracks (e.g. LoRoom for the chill-study query) can dominate
  positions 1 and 2 of every chill-leaning playlist, pushing other artists
  out. The penalty preserves their visibility while still letting the
  highest-scoring track lead.
- **Auditable nudge.** The adjustment is added to the per-song breakdown
  as `artist_penalty`, so users can see exactly when and why diversity
  changed the ordering. There is no hidden re-ranking.

Concrete example from `python -m src.main --mode demo --mock`:

> Query: *"I want something chill for studying late at night"*
> Without the penalty, LoRoom would take #1 ("Midnight Coding", 5.27) and
> #2 ("Focus Flow", 1.83). The penalty drops "Focus Flow" by 0.5 to 1.33,
> which is now below "Spacewalk Thoughts" by Orbit Bloom (4.29) and
> "Library Rain" by Paper Lanterns (4.55), so two other artists move ahead.

## 9. Improvement ideas

- **Larger and richer catalog.** Even doubling the dataset to 40 songs
  would meaningfully reduce the energy-proximity tie-breaking that
  dominates niche queries. Ideally we would also pull lyrics, language,
  and instrument tags.
- **Embedding-based KB retrieval.** TF-IDF works for 30 docs; for a real
  product we would swap in sentence-transformer embeddings so synonyms
  ("calm" vs "mellow", "intense" vs "aggressive") match without exact
  keywords.
- **Learned weights.** Replace the three hand-tuned ranking modes with
  weights learned from labelled (query, ranking) pairs. Even a tiny
  logistic-regression model would let the agent fall back to a learned
  prior when the query is ambiguous.
- **Per-explanation faithfulness scoring.** Today the guardrail flags
  binary mismatches; a richer score (BLEU between explanation and
  templated breakdown, for example) would let us track regressions over
  time.
- **Stricter input policy.** The blocklist is a placeholder; a real
  product would use Anthropic's content moderation tooling.

## 10. AI collaboration reflection

I built this with substantial help from an AI coding assistant.

**Helpful suggestion.** When I described the agent design, the assistant
suggested using Anthropic's *native tool-use* API rather than asking
Claude to emit a JSON blob that I would then parse. That suggestion turned
out to be exactly right: the tool-use loop made the recommender feel like
a real tool the agent calls (which is what the rubric's agentic stretch
asks for), and it cut the brittleness of "is this JSON well-formed"
parsing out of the hot path. It also made the tool boundary the natural
place to log and to instrument.

**Flawed suggestion.** When I asked for a guardrail keyword set, the
first proposal listed `"vibe"` under both *general atmosphere* and
*mood_tags*. I accepted it, then my own mock-mode explanation template
also used the word *"vibe"* in a generic phrase ("matches your 'intense'
vibe"). The result was that every passing run produced a wave of false
positives -- the guardrail flagged itself. That was a real reliability
bug: a check that fires on its own templated output is worse than no
check at all. I tightened the keyword set, dropped *"vibe"* from
mood_tags, added a song-title prefix-stripper so titles like "Velvet Slow
Dance" stop triggering the danceability check, and re-ran the harness
clean. Lesson: any guardrail needs to be tested against the system it
guards, not just against adversarial input.

## 11. Ethics: misuse and prevention

The system is not powerful enough to do meaningful damage on its own (it
recommends songs from a 20-track catalog), but in a scaled-up version a
few real misuse vectors exist:

- **Manipulation through curated catalogs.** A bad-faith operator could
  load a catalog whose `mood_tags`, `popularity`, and `release_decade`
  fields are tilted to push specific artists or labels regardless of
  user intent. Mitigation: surface the per-dimension breakdown (as we
  already do) so the user can see why each track ranked where it did,
  and require dataset provenance metadata.
- **Reinforcing filter bubbles.** Without the diversity penalty, a
  recommender that always picks the same on-vibe artist for similar
  queries would narrow listener exposure over time. The artist penalty
  is the first line of defence here; richer diversity (genre,
  decade, language) would be the next step.
- **Hallucinated justifications.** An LLM-driven explanation could in
  principle give a confident, false reason for a recommendation, which
  is its own form of misuse (especially in regulated domains beyond
  music). The output guardrail in this system is the prevention strategy:
  every textual claim is checked against the math, and mismatches are
  visibly flagged in the same table the user reads.
- **Privacy.** This system does not collect or persist any user data
  beyond the in-process query and a local log file. A productionised
  version should make logging opt-in, redact PII, and add a clear
  retention policy.

In all of these, the design choice is the same one I made throughout the
project: **show the math and flag the disagreements rather than hide
them**.

## 12. Testing summary

- Unit tests (`python -m pytest`): **2/2 passing**.
- Eval harness (`python eval.py --mock`): **8/8 passing** (5 NL inputs +
  3 edge cases). Average confidence on passing cases ~0.39, average
  distinct genres in top-5 = 3.4.
- Manual demo (`python -m src.main --mode demo --mock`): all three preset
  queries produce ranked, explained, guardrail-clean output with the
  artist diversity penalty visibly active.
