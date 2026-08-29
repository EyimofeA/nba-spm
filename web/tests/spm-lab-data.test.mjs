import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const lab = JSON.parse(
  readFileSync(new URL("../local-data/spm-lab.json", import.meta.url), "utf8"),
);

test("SPM Lab uses same-season stabilization and preserves 2027", () => {
  assert.equal(lab.scope, "localhost_only");
  assert.equal(lab.stabilization.prior_season_stabilization, false);
  assert.equal(lab.stabilization.future_season_stabilization, false);
  assert.deepEqual(lab.seasons, [2021, 2022, 2023, 2024, 2025, 2026]);
  assert.deepEqual(lab.selection_gate.untouched_confirmation_season, 2027);
});

test("SPM Lab exposes decisions, validation, and named rating rows", () => {
  assert.equal(lab.decisions.length, 8);
  assert.deepEqual(
    lab.decisions.filter((row) => row.selected).map((row) => row.group),
    ["bbi_passing", "raptor_shot_defense", "raptor_matchup_volume"],
  );
  assert.deepEqual(lab.validation.map((row) => row.test_season), [2022, 2023, 2024, 2025, 2026]);
  assert.ok(lab.ratings.length > 5_000);
  assert.ok(lab.ratings.every((row) => row.PLAYER_NAME));
  assert.ok(lab.ratings.every((row) => Math.abs(row.selected_offense + row.selected_defense - row.selected_net) < 1e-9));
});

test("SPM Lab defaults can render the public metric benchmark", () => {
  assert.deepEqual(lab.comparison.common_seasons, [2021, 2022, 2023, 2024]);
  assert.equal(lab.comparison.minimum_metric_year_minutes, 250);
  assert.equal(lab.comparison.replacement_value, -2);
  assert.equal(lab.comparison.minutes_mode, "observed_next_season");
  assert.equal(lab.comparison.projected_minutes_status, "not_run_no_archived_projection_file");
  assert.equal(lab.comparison.team_win_summary.length, 10);
  assert.equal(lab.comparison.team_win_folds.length, 40);
  assert.equal(lab.comparison.team_win_summary[0].metric, "mamba");
  assert.ok(lab.comparison.pairwise_correlations.length >= 10 * 10 * 3);
  assert.ok(lab.comparison.definitions.some((row) => row.metric === "epm" && row.included));
  assert.ok(lab.comparison.definitions.some((row) => row.metric === "site_aio" && row.included));
  assert.ok(!lab.comparison.definitions.some((row) => row.metric === "old_aio"));
});

test("SPM Lab exposes the frozen Box15 prior, posterior, and comparisons", () => {
  assert.equal(lab.box15.run_id, "final_box_feature_ladder_v1_8bb26f12e7");
  assert.deepEqual(lab.box15.seasons, [2021, 2022, 2023, 2024, 2025, 2026]);
  assert.deepEqual(lab.box15.correlation_seasons, [2021, 2022, 2023, 2024]);
  assert.equal(lab.box15.minimum_minutes, 250);
  assert.ok(lab.box15.leaderboard.length > 3_000);
  assert.ok(lab.box15.leaderboard.every((row) => row.PLAYER_NAME));
  assert.ok(lab.box15.leaderboard.every((row) => Math.abs(row.prior_offense + row.prior_defense - row.prior_net) < 1e-9));
  assert.ok(lab.box15.leaderboard.every((row) => Math.abs(row.posterior_offense + row.posterior_defense - row.posterior_net) < 1e-9));
  assert.equal(lab.box15.correlations.length, 30);
  assert.ok(lab.box15.correlations.some((row) => row.metric === "site_aio"));
  assert.ok(lab.box15.correlations.some((row) => row.metric === "full_spm_aio"));
});

test("SPM Lab exposes the exact feature catalog and weight ablation", () => {
  assert.equal(lab.weighting.quality.rows, 6942);
  assert.equal(lab.weighting.feature_catalog.length, 170);
  assert.equal(lab.weighting.feature_catalog.filter((row) => row.offense_input).length, 127);
  assert.equal(lab.weighting.feature_catalog.filter((row) => row.defense_input).length, 68);
  assert.ok(lab.weighting.feature_catalog.every((row) => row.description.length > 10));
  assert.equal(lab.weighting.summary.length, 12);
});
