# External reproduction audit results

Date: 2026-08-25
Final run: `external_reproduction_benchmark_v1_0a95702214`

Four read-only reviews used the same preregistered audit prompt:

- Cursor Grok 4.6 slow, maximum thinking;
- OpenRouter `stealth/ox-alpha`, maximum thinking;
- Cursor Opus 5 slow, maximum thinking;
- Cursor Kimi K3, maximum thinking.

## Consensus

No reviewer found a P0. The annual Ryan Davis and current xRAPM headline
correlations were independently reproduced. The initial multi-window result
and the game-level home coefficient were not safe until fixed.

## Material findings and disposition

1. **Ryan multi-window convention — fixed.** Ryan's `2018-23` label means the
   five NBA seasons ending 2019 through 2023, not six seasons ending 2018
   through 2023. Opus 5 and Kimi K3 both caught this independently using rookie,
   retirement, and annual-file checks. The runner now maps the labels to exact
   three- and five-year CourtSignal windows. Correct pooled net correlations
   are `.980` for three-year windows and `.957` for five-year windows.
2. **Game-level home term — fixed.** The initial implementation contained both
   an all-ones column and a fitted intercept, so its near-zero `home_effect` was
   an identification artifact. ox-alpha and Grok both found it. The model now
   uses the home-margin intercept only; the fitted home advantage is `+1.696`
   points per 100. Player correlations and the selected lambda are unchanged.
3. **Pre-2000 season parsing — fixed.** ox-alpha found that two-digit ending
   years could resolve to the wrong century. Tests now cover `1997-98`,
   `1999-00`, `2014-19`, and Ryan's window mapping.
4. **Luck-adjusted RAPM label — fixed.** Grok and Kimi noted that exact
   player-season keys do not make luck-adjusted and normal RAPM the same
   estimand. Those rows are now labeled `different_estimand`.
5. **RAPTOR tails — fixed in reporting.** All reviewers found that low-minute
   on/off tails explain Pearson `.436` versus rank correlation `.829`; this is
   not a sign or join failure. The artifact now reports the full sample and a
   1,000-minute stratum. Qualified net Pearson/rank correlation is
   `.917/.912` across 2,449 player-seasons. The source is called the RAPTOR
   on/off component, not raw on/off.
6. **PBPStats units — fixed.** Grok found that `TotalPoss` is offense plus
   defense possessions. Raw on-court margin now divides by average team
   possessions, producing conventional points-per-100 units. Correlation is
   unchanged.

## Safe final claims

- Ryan Davis annual normal net RAPM, 2014-2023: n=5,217, Pearson `.967`, rank
  correlation `.962`, CourtSignal-on-reference scale slope `1.391`.
- Ryan Davis exact five-year windows: n=5,513, Pearson `.957`, rank `.948`.
- Ryan Davis exact three-year windows: n=5,869, Pearson `.980`, rank `.970`.
- xRAPM 2024-2026: n=687, Pearson `.897`, rank `.888`, with an explicit unequal
  season-weight caveat.
- The local legacy AuPM formula reproduces its stored column to `1.78e-15`.

None of these correlations replaces chronological future-game validation.
DARKO WOWY, RAPTOR on/off, AuPM, PBPStats on-court margin, and game-level PM
remain different-estimand diagnostics.
