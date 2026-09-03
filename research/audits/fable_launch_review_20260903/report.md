# CourtSignal Fable review verification

Date: 2026-09-03. Base source: `4b92a418`.

Eight Fable 5.1 reviews and follow-ups completed through the authenticated
Cursor CLI. Third-party allowance moved from 58.01818% used to 99.80909% used.
No further request was started with the remaining 0.19091%; billing settings
were not changed. All findings below are local. No commit, push, deployment,
model fit, or public-data regeneration was performed.

## Confirmed correction

The WP target builder reordered period-local possession numbers across
quarters. A four-possession fixture reproduces the error even though
conservation passes. The pinned checkpoint contains 853,715 backward steps
across 2014–2023 and none in 2024–2026. Rebuilding credit from its existing
probability states changes zero modern credit rows.

The source repair and cache guard pass 23 focused tests. Existing checkpoints,
ratings, and public assets have not changed. Historical WP comparisons need a
corrected rerun. The 2024–2026 annual boards are unchanged at their fixed
parameters, but the historical justification for those parameters is no longer
sound. No finding here authorizes a new penalty grid or a promotion.

Run the read-only audit from the repository root:

```sh
.venv/bin/python research/audits/fable_launch_review_20260903/audit_wp_chronology.py
.venv/bin/python -m pytest tests/test_repository_boundaries.py tests/test_rapm.py tests/test_win_probability_rapm.py tests/test_wp_target_checkpoint.py -q
```

The audit reports only aggregates and the input hash. It never rewrites the
checkpoint. The research log records the correction to the earlier log-odds
and lambda claims.

## Advisory reports

- `statistics.md`: initial source review, independently confirms the sorting
  concern and distinguishes fixed reused selection from chronological selection.
- `product.md`: source-level UI review, identifies stale selection/data risks
  and percentiles that depend on name filters. Browser behavior still needs
  reproduction before feature fixes are called complete.
- `reconstructions.md`: source-level reconstruction review. In-sample RAPTOR
  agreement is not forward validation. Missing-feature, duplicate-row, and
  source-scope claims require local data checks or primary-source confirmation.

These reports preserve the model's response, including uncertain claims and
errors. They are not ground truth. Follow-up reviews are checking its proposed
remedies, mathematical claims, and incomplete source observations.

## Reconstruction checks

Local schema checks found 172 numeric RAPTOR inputs, including `OffPoss`,
`DefPoss`, and eight matchup fields. No season or age feature appeared in the
selected list. The review's suggestion of those extra inputs was hypothetical.
The official by-team file does contain season type and box offense/defense/net
columns, so an RS-only source comparison is feasible without new data.

All 1997–2026 PIPM input sheets have positive, nonmissing pace. Missing pace is
a guard gap, not a demonstrated explanation of the current disagreement.
Four seasons contain 64 rows for one player each:

| Season | Player | Distinct consumed PIPM input rows |
| --- | --- | --- |
| 2014 | Jordan Hamilton | 1 |
| 2015 | Quincy Pondexter | 1 |
| 2018 | Corey Brewer | 1 |
| 2023 | Mikal Bridges | 1 |

These rows differ in other source columns but duplicate the exact inputs
consumed by PIPM. Deduplicating those inputs before the lineup join avoids
arbitrary traded-team selection. Require a one-to-one player join afterward.
The existing builder repeats these inputs in its season weighting and output;
the size of the resulting rating change has not been fitted. No PIPM or RAPTOR
artifact was changed during these checks.

## Browser fixes and checks

The local review branch fixes search-dependent percentiles in RAPM Lab and
Reconstructions. Jokić's 2026 annual RAPM defense remains at the 76th percentile
after filtering to his name. Before the fix, that search changed it to 100th.
The DARKO reconstruction retains its 97th defensive percentile under the same
check. The reconstruction table was also inspected at a 390 by 844 viewport.
Impact columns remain in its existing horizontal scroll area.

Role-map data now carries the season and side of its request. An intentionally
delayed local response showed a loading message with zero chart and export
button elements after a season change. The chart and export button returned
after that response arrived. RAPM data carries its request URL and ignores
responses from abandoned selections. A delayed 2025 response did not replace
the 2024 board selected immediately afterward. The 2024 Jokić row remained
`+3.60 / +1.24 / +4.84` after the delayed response completed.

The browser checks used a local forwarding server that delayed only selected
JSON responses. It did not modify payloads. The normal client build and all 40
client tests pass. Lint has zero errors and the existing headshot-image warning.
The Ponytail complexity review found no new dependency or unnecessary helper.
The other UI suggestions in the advisory reports remain unimplemented.

## PULSE review disposition

`pulse.md` traces PULSE's source dependencies and does not find a WP-target
consumer in the PULSE rating path. Its conditional concern about the published
prior is not present in the pinned configuration: `pulse_v1.yml` explicitly
sets `prior_scale: 1`, with penalties 3000/4500/300. The chronological validation
loop trains before the rating season and scores the following season. Published
ratings instead use the final descriptive prior mapping; these are different
evidence products, not interchangeable validation rows.

The reviewer did not receive every imported module or the artifacts and says
so in its report. Its release-lineage, cache-identity, mixed-season missing-value,
and averaged prior/update concerns remain source-level findings to verify. No
additional PULSE model or publication change was made in response to them.

The statistics follow-up proposes extra fixed arms, model selection, and an
uncertainty bootstrap. Those are advice, not an accepted experiment contract.
No rerun, grid, bootstrap, or 2027 use was authorized or executed by this review.
Its byte-identical publication suggestion is also stronger than a numerical
tolerance check can establish and is not an accepted release requirement.

## Further advisory work

- `statistics-followup.md` and `product-followup.md` critique the initial
  reviews using the verified chronology and browser evidence.
- `reconstructions-followup.md` gives repair sketches and source-agreement
  diagnostics. Its exact-input deduplication, finite-pair metrics, regular-season
  alignment, and immutable-run suggestions are useful next checks, not merged
  code. The proposed metric helper lacks zero-variance/zero-error guards. Its
  claim that duplicated-player corrections necessarily lower correlation or
  direct R-squared is not established: changes in squared error include cross
  terms and can go either way. The simple common-offset derivation holds with
  fixed features; changing league pace also changes those features. Numerical
  consequences still require a controlled recomputation. Do not copy the
  proposed many-to-one join merely to accommodate a test; enforce one-to-one
  joins after input uniqueness is established.
- `current-strength.md` distinguishes the dated statistical-center foundation
  from a complete dated AIO. Missing dated likelihood is a planned capability,
  not evidence that the published PULSE model is broken. Its temporal/feature
  contract fixtures are candidates for follow-up. The proposed six-arm scoring
  study and nested parameter searches are not accepted or run. Some underlying
  builders and artifacts were absent from its snapshot. The response includes
  an interrupted draft followed by its completed review; it is preserved as
  received rather than presented as a verified specification.

The next consequential action is to withdraw or supersede the affected rolling
WP board and comparison evidence, then regenerate under a separately recorded
contract. The source fixes here do not silently repair already published data.
