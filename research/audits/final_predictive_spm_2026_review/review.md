# Final predictive SPM workstream review

Date: 2026-08-26

Scope: target-horizon selection, predictive current-strength AIO, luck-adjusted
RAPM, current player skills, and the localhost-only skill UI. Four independent
read-only reviewers inspected statistics, temporal leakage and lineage,
basketball semantics, and UI behavior.

## Findings and resolutions

| Review | Finding | Resolution |
| --- | --- | --- |
| Statistical | Thirteen selected age arms were labeled but not applied to final skill estimates. | Fixed. The selected age curve produces the preseason estimate, which is then updated with named-season observations. All 7,254 comparable 2026 age-arm rows changed; median absolute change is 0.130 in each skill's native unit. |
| UI | The 2026 game chart included playoffs and used a different Beta update from the frozen annual estimator. | Fixed. Only regular-season games enter. Each FT/3P series stops at the exact prefix matching the frozen annual makes and attempts, uses the artifact's preseason estimate and precision, and must end at the annual posterior. Non-reconciling series are withheld. |
| Basketball | Offensive and defensive rebound sources can exceed 100%, so grouped-binomial likelihood created negative failures. | Fixed. Both are scored as rates with opportunity-weighted RMSE and MAE. Their selected hyperparameters happened to remain unchanged. |
| Leakage | Predictive-SPM manifests embedded absolute machine paths. | Fixed in both saved manifests and future writes; paths are repository-relative. |
| Leakage | The target-horizon run chose its checkpoint identity before hashing statistical player sheets. | Fixed. All player sheets are loaded and hashed before the run directory or resume decision is selected. The completed historical artifact is retained as prior evidence; future resumes use the corrected identity contract. |
| Leakage | AIO accepted any parquet whose method was `raw`. | Fixed. It now requires the preregistered prior-run ID and verifies that every training cutoff precedes its forecast season. |
| UI | Rapid comparison clicks could display the earlier request, source values were called raw, and the radar lacked a comparison key. | Fixed with request identity, `Source`/`Game` labels, and a visible player-color key. |
| Basketball | Shot-quality and spacing labels overstated what was measured. | Fixed. Shot quality now states defender-distance and two-versus-three-point mix; the profile blend is named `Shooting context`. |

## Remaining limits

- The AIO bootstrap intervals are conditional on the frozen selected candidate;
  selection is not repeated inside each draw.
- Age penalties are tested after the best non-age decay/prior/exposure arm is
  locked, not over a fully joint grid.
- Historical skill curves are post-hoc stabilized histories under parameters
  selected through 2024, not estimates that were issued contemporaneously.
- Skill outputs remain research-only and localhost-only. Defensive outcomes are
  observational. Continuous-skill uncertainty is not calibrated.
- Seasons 2025 and 2026 are reused diagnostics. Season 2027 remains untouched.

## Decision

The corrected current-skill artifact passes the registered research gates. The
predictive current-strength AIO remains a research champion within its frozen
candidate set, not a confirmed or public model. Normal RAPM remains the public
reference; the luck-adjusted arms remain null results.
