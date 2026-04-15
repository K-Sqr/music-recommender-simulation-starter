# Reflection: Profile Comparison Notes

## Pairwise Output Comparisons

### High-Energy Pop vs Chill Lofi

High-Energy Pop surfaces upbeat songs with higher energy values, while Chill Lofi prioritizes lower-energy tracks with calm/chill moods. This makes sense because the target energy values are far apart (`0.9` vs `0.35`), so the energy proximity term strongly separates the top results. The genre bonus also pushes pop songs up for the first profile and lofi songs up for the second profile.

### High-Energy Pop vs Deep Intense Rock

Both profiles prefer high energy, so some energetic tracks overlap near the top of both lists. The key difference is mood and genre preference: High-Energy Pop favors `happy` and `pop`, while Deep Intense Rock favors `intense` and `rock`. That is why songs like `Sunrise City` dominate the pop profile and `Storm Runner` dominates the rock profile.

### Chill Lofi vs Deep Intense Rock

These two profiles produce very different results because they differ on all major preference signals: genre, mood, and target energy. Chill Lofi ranks mellow tracks (`Library Rain`, `Midnight Coding`) because they match both lofi/chill and low energy. Deep Intense Rock ranks aggressive, high-energy tracks first because the scoring rewards intense mood and close energy near `0.92`.

## What I Learned From The Comparisons

The profile tests show that this scoring system is easy to interpret: when inputs change, ranking changes in predictable ways. It also shows a limitation: fixed weights can make one feature (like genre) dominate even when another song has a better vibe match on energy.
