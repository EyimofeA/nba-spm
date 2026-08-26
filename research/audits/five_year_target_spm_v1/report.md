# Five-year-target SPM audit

Decision: pass as the research SPM replacement; do not promote publicly before
the untouched 2027 test.

The selected run is `five_year_target_spm_v1_65550acb79`. It uses 8,608 matched
five-year player windows, 127 offense features, and 68 defense features. The
historical five-year target rebuild agrees with the reference artifact within
`1.50e-7` points per 100. No 2027 row was loaded. Offense plus defense equals
net exactly in every SPM and AIO row.

The five-year-prior AIO beats the annual-prior AIO on next-season game-margin
RMSE in all five scored seasons and improves mean RMSE from `14.4705` to
`14.4005`. It also beats the zero-prior one-year RAPM mean of `14.5697`.
Development and reused-diagnostic paired game intervals both exclude zero.

The standalone SPM tradeoff is real. Training on stable five-year labels raises
next-year one-season-RAPM net RMSE from `1.7543` to `1.9832`, while correlation
rises from `.4106` to `.4232`. This model is selected for its downstream AIO
use, not because it mimics noisy annual RAPM more closely.

Known limits:

- 2025 and 2026 are reused diagnostics, not untouched confirmation;
- the feature-complete run followed a common-feature pilot, so its interval is
  not selection-aware;
- test-season lineups are observed, so this is player-rating validation rather
  than a roster-only season forecast;
- some 2025--26 defense fields are pooled with earlier observed seasons because
  the annual current-season sources are incomplete.
