# Shot-quality matchups and lineup-combination RAPM

**Status:** preregistered research projects. Neither output belongs in the
public ratings, SPM inputs, or AIO prior until it wins a frozen, forward test.

## Why these are two separate projects

The current matchup Elo model observes offensive player–defender assignment
aggregates: possession overlap, field-goal attempts/makes, threes, free throws,
points, assists, turnovers, and blocks. It does **not** identify the defender
on each individual shot. The play-by-play archive has shot location and result,
but no primary defender assignment for every shot. Combining the two naively
would falsely attribute an individual shot's quality to a defender.

The current matchup Elo should therefore remain an exploratory assignment model.
It is useful for finding questions, not for a defensive leaderboard.

## Project A — Shot quality before matchup ratings

### First frozen v0 result

`expected_shot_quality_v1_f5d343a852` was trained on the complete 2023–24
season (217,570 shots), isotonic-calibrated on 2024–25 (218,084), and scored
once on the untouched 2025–26 season (218,722). Years here are **season-end
years**, matching the site.

| 2026 untouched test | Brier | Log loss | Mean prediction | Actual make rate |
|---|---:|---:|---:|---:|
| Base model | .23167 | .65573 | .47155 | .47107 |
| Calibrated overall | **.23075** | **.65413** | .46981 | .47107 |
| Rim | .21499 | .62048 | .66433 | .67104 |
| Non-rim | .23701 | .66749 | .39263 | .39172 |

The calibration set improves both held-out scores modestly. The remaining rim
gap means this is a transparent v0 opportunity baseline, not a finished public
expected-shot metric. Its player outputs are stored only as a research artifact
and still do not contain defender credit.

### Available event data

The CDN event archive for 2023–26 has one row per action with `gameId`,
`actionNumber`, `orderNumber`, clock, period, shooter `personId`, team,
`x`/`y`, `xLegacy`/`yLegacy`, `area`, `areaDetail`, `shotDistance`,
`actionType`, `subType`, shot value, and `shotResult`. This is enough for a
simple, reproducible expected-field-goal model. It lacks defender distance,
contest type, shot-clock state, and a per-shot primary defender.

### v0 expected-shot model

**Grain:** a non-heave two- or three-point field-goal attempt.

**Outcome:** made field goal, with expected points `shot_value × P(make)`.

**Pre-shot features only:** season, two-versus-three value, court location
(`x`, `y`, distance, angle, and smooth location terms), NBA area/area-detail,
action type/subtype, quarter/OT, game clock, and home/offense indicator.

**Excluded:** free throws, post-shot score, rebound/assist/block result,
descriptions, future events, player identity, defender identity, and team
outcome. These exclusions stop the model from learning a shooter's talent or a
defender's result as if it were shot quality.

**Fit contract:** train only on earlier seasons; validate calibration and Brier
score by season and separately for rim/non-rim. A flexible logistic GAM or
regularized spline logistic regression is adequate for v0; a neural model adds
complexity before it adds useful evidence.

### Outputs

For every shooter-season, publish only descriptive totals:

| Quantity | Definition |
|---|---|
| Expected points | Sum of pre-shot expected points |
| Shot-quality creation | Expected points per attempt minus league average, split rim/non-rim |
| Shotmaking above quality | Actual field-goal points minus expected points, split rim/non-rim |

This directly addresses the current big-man problem: a player is not penalized
for taking high-value rim shots. High-quality attempts become opportunity;
conversion above or below that opportunity is a separate residual.

### Matchup extension: blocked until a shot-level join exists

The desired observation is `(shooter, defender, shot)` with `xPts`, actual
points, and a primary/credited defender. The current aggregate matchup table
cannot supply it. Do **not** allocate a player's season shot quality to his
defenders in proportion to overlap; that merely assumes the answer.

Once a rights-reviewed event source supplies a shot-level defender assignment,
fit two linked ridge models on shot-level residuals:

```
quality_ij = xPts_ij - league_xPts(zone_ij)
make_ij    = actualPts_ij - xPts_ij

quality_ij = offense_quality[i] - defense_suppression[j] + context + error
make_ij    = offense_shotmaking[i] - defense_contest[j] + context + error
```

Both models will be run separately for rim and non-rim attempts. Report a
minimum exposure, shrinkage, and out-of-sample calibration. The model is not a
claim that the nearest defender caused every outcome.

### Intermediate fallback, explicitly weaker

Before a defender-at-shot source exists, we may use current matchup FGA/FGM and
a shooter's **own season-level** shot-quality profile to make a coarse matchup
residual. It cannot distinguish which defenders induced which locations, so it
is a diagnostic only and cannot be called suppression or contest skill.

The source audit is maintained in
[SHOT_LEVEL_DEFENDER_SOURCE_AUDIT.md](../data/SHOT_LEVEL_DEFENDER_SOURCE_AUDIT.md).

## Project B — 2–5 player combination RAPM

### Estimand

The additive RAPM model estimates player coefficients. Combination RAPM asks a
different question: after the five individual player terms, does a recurring
offensive or defensive pair/trio/group systematically over- or under-perform?
It estimates **lineup interaction residuals**, not a causal chemistry score.

### Design

1. Start from terminal-lineup possession rows under the normal RAPM contract.
2. Preserve additive player and home columns exactly.
3. Add separate same-side interaction columns for observed 2-player groups;
   only later add 3-, 4-, and 5-player columns.
4. Include a group only if it clears a preregistered exposure threshold and
   appears in more than one game. Never infer a group from an unobserved
   lineup.
5. Use order-specific ridge shrinkage, centered at zero:

   `lambda_2 < lambda_3 < lambda_4 < lambda_5`.

   Choose the four penalties on development seasons only, using whole-game
   held-out margin error. Higher-order groups must shrink more because their
   exposure and identifiability collapse rapidly.
6. Fit offensive and defensive group interactions with the same sign
   convention as RAPM. A net interaction is offense plus defense.

### Sequence and gates

| Stage | Candidate | Promotion gate |
|---|---|---|
| 1 | Two-player interactions | Improvement on held-out game margin without degrading player-term stability |
| 2 | Three-player interactions | Beats Stage 1 on untouched selection seasons and remains sign-stable across adjacent windows |
| 3 | Four-player interactions | Same gate, plus minimum multi-season recurrence |
| 4 | Five-player lineup effects | Research-only; requires a clear out-of-sample win and no leakage from lineup selection |

Use rolling three-season source windows and one future season as the first
evaluation. The all-history model is a descriptive stability check, not the
selection criterion. Every baseline and challenger must score the exact same
games and lineups. Results are game-clustered for uncertainty; possession-row
resampling is prohibited.

### Explicit non-goals

- Do not put group terms into current normal RAPM, SPM, or AIO.
- Do not publish five-man rankings as player value.
- Do not treat a high group coefficient as evidence that coaching, role,
  opponent, or roster construction caused it.

## Immediate next implementation

1. Build and calibrate the shooter-only expected-shot v0 on 2023–25; reserve
   2026 for an untouched calibration report.
2. Locate or acquire a permitted shot-level defender-assignment source before
   attempting defender-specific expected shot quality.
3. Run pair-only combination RAPM first, with frozen filtering and game-held-out
   validation. Stop if it does not beat additive RAPM.
