import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const data = JSON.parse(readFileSync(new URL("../public/data/external-benchmark.json", import.meta.url), "utf8"));
const view = readFileSync(new URL("../app/views/ResearchView.tsx", import.meta.url), "utf8");

test("external comparison pins current PULSE and separates MAMBA coverage", () => {
  assert.ok(view.includes(data.run_id));
  const main = data.panels.find((panel) => panel.scope === "main");
  const mamba = data.panels.find((panel) => panel.scope === "with_mamba");
  assert.equal(main.games, 13209);
  assert.equal(main.outcome_start, 2016);
  assert.equal(main.outcome_end, 2026);
  assert.equal(main.rows.length, 9);
  assert.ok(main.rows.every((row) => row.folds === 11 && Number.isFinite(row.aggregate_rmse)));
  assert.equal(mamba.outcome_end, 2025);
  assert.ok(mamba.rows.some((row) => row.candidate === "MAMBA"));
  assert.ok(!main.rows.some((row) => row.candidate === "MAMBA"));
  for (const name of ["PULSE", "RAPM", "xRAPM", "EPM", "DARKO DPM", "LEBRON", "BPM 2.0", "CourtSignal PIPM reconstruction", "CourtSignal RAPTOR reconstruction"]) {
    assert.ok(main.rows.some((row) => row.candidate === name), name);
  }
  assert.match(view, /not an out-of-time test/);
  assert.match(view, /not untouched confirmation/);
  assert.match(view, /Full-coverage PULSE test/);
  assert.equal(data.rich_prior_test.outcome_start, 2016);
  assert.equal(data.rich_prior_test.outcome_end, 2026);
  assert.equal(data.rich_prior_test.games, 13199);
  assert.equal(data.rich_prior_test.rows.length, 2);
  assert.ok(data.rich_prior_test.rows.every((row) => row.folds === 11));
  assert.ok(data.rich_prior_test.pulse_minus_rich_mse < 0);
  assert.ok(data.rich_prior_test.upper_95 < 0);
  assert.doesNotMatch(JSON.stringify(data), /"(?:PLAYER_ID|game_id|offense_coefficient|defense_coefficient|actual_margin|predicted_margin)"|\/Users\//);
});
