import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const lab = JSON.parse(
  readFileSync(new URL("../local-data/rapm-lab.json", import.meta.url)),
);

test("local RAPM Lab pairs every result with its test and decision", () => {
  assert.equal(lab.scope, "localhost_only");
  assert.ok(lab.experiments.length >= 10);
  for (const row of lab.experiments) {
    assert.ok(row.title.length > 0);
    assert.ok(row.test.length > 20);
    assert.ok(row.result.length > 10);
    assert.ok(row.decision.length > 10);
    assert.ok(row.run_id.length > 0);
  }
});

test("aligned horizon and single-season sweeps are exposed", () => {
  const single = lab.experiments.find(
    (row) => row.id === "single-season-rapm-sweep",
  );
  const horizons = lab.experiments.find(
    (row) => row.id === "rapm-horizon-age-sweep",
  );
  assert.match(single.result, /3000\/6000/);
  assert.match(single.decision, /3000\/4500\/300/);
  assert.match(horizons.result, /4 years/);
  assert.match(horizons.result, /7 years/);
});

test("rubber-band payload carries the frozen actual-clock estimate", () => {
  assert.equal(lab.rubberband.selected_spec, "six_minute_clip15");
  assert.equal(lab.rubberband.coefficients.length, 8);
  assert.equal(lab.rubberband.curve.length, 8 * 61);
  assert.ok(lab.rubberband.bootstrap.lower_95 > 0);
  assert.ok(
    lab.rubberband.selection_winner_vs_runner_up.lower_95 < 0 &&
      lab.rubberband.selection_winner_vs_runner_up.upper_95 > 0,
  );
});

test("rubber-band comparison includes possession progress and player ratings", () => {
  assert.equal(lab.rubberband.comparison_coefficients.length, 16);
  assert.deepEqual(
    [...new Set(lab.rubberband.comparison_coefficients.map((row) => row.basis))].sort(),
    ["actual_clock", "possession_progress"],
  );
  assert.equal(lab.rubberband.rapm_evaluation.length, 5);
  assert.ok(lab.rubberband.ratings.length >= 300);
  assert.ok(
    lab.rubberband.comparison_run_id.startsWith("rubberband_progress_rapm_v2_"),
  );
});

test("JE replication exposes exact score states and a held-out rejection", () => {
  assert.equal(lab.rubberband.je.curve.length, 51);
  assert.equal(
    lab.rubberband.je.curve.find((row) => row.margin === 0)
      .effect_points_per_100_vs_tie,
    0,
  );
  assert.equal(lab.rubberband.je.evaluation.length, 3);
  assert.ok(
    lab.rubberband.je.bootstrap_vs_normal.neutral_player_only.lower_95 > 0,
  );
  assert.ok(
    lab.leaderboards.some(
      (board) => board.experiment_id === "rubberband-je",
    ),
  );
});

test("saved RAPM tests expose real leaderboards", () => {
  assert.ok(lab.leaderboards.length >= 10);
  for (const board of lab.leaderboards) {
    assert.ok(board.title.length > 0);
    assert.ok(board.columns.length >= 2);
    assert.ok(board.rows.length > 0);
    for (const column of board.columns) {
      assert.ok(column.key.length > 0);
      assert.ok(column.label.length > 0);
    }
  }
  for (const experimentId of [
    "rubberband-rapm",
    "standalone-units",
    "factors",
    "win-probability",
    "coach",
    "point-channels",
  ]) {
    assert.ok(
      lab.leaderboards.some((board) => board.experiment_id === experimentId),
      `Missing leaderboard for ${experimentId}`,
    );
  }
});

test("reconstruction lab publishes generated rows and source agreement only", () => {
  assert.deepEqual(
    lab.replications.map((row) => row.metric),
    [
      "CourtSignal DARKO WOWY reconstruction",
      "CourtSignal RAPTOR reconstruction",
      "CourtSignal PIPM reconstruction",
    ],
  );
  assert.equal(lab.replications[0].status, "exact_reconstruction");
  assert.equal(lab.replications[1].status, "methodology_aligned_reconstruction");
  assert.equal(lab.replications[2].status, "methodology_aligned_reconstruction");
  assert.equal(lab.replications[0].r_squared, 1);
  assert.ok(lab.replications[1].r_squared > 0.75);
  assert.ok(lab.replications[2].r_squared > 0.65);
  assert.ok(lab.replication_leaderboards.length >= 15);
  assert.ok(lab.replication_leaderboards.every((row) => row.rows.length > 0));
  assert.ok(lab.replication_leaderboards.every((row) => row.columns.length >= 4));
  assert.ok(lab.replication_leaderboards.every((row) => row.source === "CourtSignal reconstruction"));
  const payload = JSON.stringify(lab.replication_leaderboards);
  assert.doesNotMatch(payload, /reference_|official|xrapm|pipm_net|bpm_net/i);
  const raptor = lab.replication_leaderboards.filter((row) => row.metric === "CourtSignal RAPTOR reconstruction");
  assert.deepEqual(raptor.map((row) => row.season).sort((a, b) => a - b), [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]);
  assert.deepEqual(raptor[0].columns.map((column) => column.key), [
    "player", "team", "box_offense", "box_defense", "box_net",
    "onoff_offense", "onoff_defense", "onoff_net",
    "offense", "defense", "net", "exposure",
  ]);
  const darko = lab.replication_leaderboards.filter((row) => row.metric === "CourtSignal DARKO WOWY reconstruction");
  assert.equal(Math.min(...darko.map((row) => row.season)), 1957);
  assert.equal(Math.max(...darko.map((row) => row.season)), 2026);
  const pipm = lab.replication_leaderboards.filter((row) => row.metric === "CourtSignal PIPM reconstruction");
  assert.equal(Math.min(...pipm.map((row) => row.season)), 1997);
  assert.equal(Math.max(...pipm.map((row) => row.season)), 2026);
});

test("large unit leaderboards carry both tails", () => {
  for (const id of ["unit-2", "unit-3", "unit-4", "unit-5"]) {
    const board = lab.leaderboards.find((candidate) => candidate.id === id);
    assert.ok(board);
    assert.equal(board.rows.length, 200);
    assert.equal(board.rows.filter((row) => row.sample === "Top 100").length, 100);
    assert.equal(
      board.rows.filter((row) => row.sample === "Bottom 100").length,
      100,
    );
  }
});

test("new full-span and five-point results are exposed without WP unit inflation", () => {
  assert.ok(lab.age.run_id.startsWith("age_adjusted_full_1997_2026_v1_"));
  assert.equal(lab.rubberband.five_point.curve.length, 11);
  assert.equal(
    lab.rubberband.five_point.curve.find((row) => row.margin === 0)
      .effect_points_per_100_vs_tie,
    0,
  );
  const rolling = lab.leaderboards.find(
    (board) => board.id === "rolling-wp-ratings",
  );
  assert.ok(rolling);
  assert.ok(rolling.rows.length > 1000);
  assert.ok(
    Math.max(...rolling.rows.map((row) => Math.abs(row.net_wp_percentage_points_per_100))) < 20,
  );
  assert.equal(Math.min(...rolling.rows.map((row) => row.window_end)), 2018);
  assert.equal(Math.max(...rolling.rows.map((row) => row.window_end)), 2026);
  assert.ok(rolling.rows.every((row) => typeof row.player_name === "string" && row.player_name.length > 0));
  const wpPriorTest = lab.experiments.find((row) => row.id === "wp-spm-aio");
  assert.equal(wpPriorTest.status, "lost");
  assert.ok(lab.leaderboards.some((row) => row.id === "wp-spm-aio-2026"));
});

test("new research tables expose fixed years and external comparison columns", () => {
  const production = lab.leaderboards.find((row) => row.id === "production-5y-ratings");
  const ryan = lab.leaderboards.find((row) => row.id === "ryan-davis-ratings");
  assert.ok(production.rows.every((row) => Number.isInteger(row.window_start) && Number.isInteger(row.window_end)));
  assert.deepEqual(ryan.columns.map((column) => column.key).slice(1, 7), [
    "target_offense",
    "target_defense",
    "target_net",
    "ryan_offense",
    "ryan_defense",
    "ryan_net",
  ]);
});

test("teammate, play-channel, and actual-age challenger tables are explicit", () => {
  const teammate = lab.leaderboards.find(
    (row) => row.id === "teammate-effect-ratings",
  );
  const scoring = lab.leaderboards.find(
    (row) => row.id === "observable-scoring-channels",
  );
  const finishes = lab.leaderboards.find(
    (row) => row.id === "observable-finish-channels",
  );
  const currentAge = lab.leaderboards.find(
    (row) => row.id === "time-decay-actual-age-ratings",
  );
  assert.deepEqual(teammate.columns.map((column) => column.key).slice(1), [
    "teammate_scoring",
    "teammate_turnovers",
    "teammate_assists",
    "teammate_steals",
    "teammate_blocks",
    "teammate_oreb",
    "teammate_dreb",
  ]);
  assert.ok(scoring.rows.length >= 300);
  assert.ok(finishes.columns.some((column) => column.key === "playtype_other_points_net"));
  assert.ok(currentAge.rows.every((row) => Number.isFinite(row.age_net_adjustment)));
  const timeExperiment = lab.experiments.find(
    (row) => row.id === "time-decay-actual-age",
  );
  assert.equal(timeExperiment.status, "lost");
});

test("joint actual-clock rubber-band RAPM is explicit and has a leaderboard", () => {
  const experiment = lab.experiments.find(
    (row) => row.id === "rubberband-joint-clock",
  );
  const board = lab.leaderboards.find(
    (row) => row.id === "rubberband-joint-clock-ratings",
  );
  assert.equal(experiment.status, "lost");
  assert.match(experiment.test, /unchanged/i);
  assert.ok(board.rows.length >= 400);
  assert.deepEqual(board.columns.map((column) => column.key), [
    "player_name",
    "normal_net",
    "joint_offense",
    "joint_defense",
    "joint_net",
    "joint_net_change",
  ]);
});

test("age and score controls use one blocked comparison and expose ratings", () => {
  const experiment = lab.experiments.find(
    (row) => row.id === "age-score-context",
  );
  const board = lab.leaderboards.find(
    (row) => row.id === "age-score-context-ratings",
  );
  assert.equal(experiment.status, "lost");
  assert.equal(lab.rubberband.score_signal.selection_winner.shape, "signed_buckets");
  assert.equal(lab.rubberband.age_score.quality.score_columns, 10);
  assert.equal(lab.rubberband.age_score.quality.season_2027_loaded, false);
  assert.equal(
    lab.rubberband.age_score.diagnostic.passes_reused_diagnostic_gate.player_only.age_only,
    false,
  );
  assert.ok(board.rows.length >= 400);
  assert.deepEqual(board.columns.map((column) => column.key), [
    "player_name",
    "normal_net",
    "age_net",
    "score_net",
    "combined_offense",
    "combined_defense",
    "combined_net",
    "combined_net_change",
  ]);
});
