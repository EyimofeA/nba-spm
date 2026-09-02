import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const catalog = JSON.parse(
  readFileSync(new URL("../public/data/rapm/catalog.json", import.meta.url), "utf8"),
);

test("public RAPM release contains selected estimands and no tuning sweep", () => {
  const ids = catalog.estimands.map((estimand) => estimand.id);
  for (const required of [
    "annual", "rolling-three", "rolling-five", "full-history",
    "same-age-27", "current-time-decay",
    "current-age-time-decay", "luck-adjusted", "coach", "win-probability", "units",
    "game-level-pm", "point-channels", "six-factor-annual", "teammate-effects",
    "teammate-efg", "observable-scoring-channels", "observable-finish-channels",
  ]) assert.ok(ids.includes(required), required);
  assert.ok(ids.some((id) => id.startsWith("factor-shooting_ts-")));
  assert.ok(ids.some((id) => id.startsWith("factor-opponent_oreb_prevention-")));
  assert.ok(!ids.includes("full-history-actual-age"));
  assert.ok(!ids.some((id) => /lambda|sweep|old-vs|rerun/i.test(id)));
});

test("WP-RAPM publishes only the latest three rolling endpoints", () => {
  const wp = catalog.estimands.find((estimand) => estimand.id === "win-probability");
  assert.deepEqual(wp.periods.map((period) => period.id), ["2024", "2025", "2026"]);
  assert.match(wp.note, /rolling five-season fit/);
});

test("game-level and point-channel boards use source-backed rows", () => {
  const gamePm = JSON.parse(
    readFileSync(new URL("../public/data/rapm/game-level-pm-2024-2026.json", import.meta.url), "utf8"),
  );
  const channels = JSON.parse(
    readFileSync(new URL("../public/data/rapm/point-channels-2024-2026.json", import.meta.url), "utf8"),
  );
  assert.ok(gamePm.length >= 500);
  assert.ok(gamePm.every((row) => Number.isFinite(row.net) && Number.isFinite(row.minutes)));
  assert.equal(channels.length, 1029);
  assert.ok(channels.every((row) =>
    Math.abs(row.one_point_net + row.two_point_net + row.three_plus_net - row.net) < 0.001,
  ));
});

test("teammate eFG and observable channel boards carry fitted values", () => {
  const teammateEfg = JSON.parse(
    readFileSync(new URL("../public/data/rapm/teammate-efg-2024-2026.json", import.meta.url), "utf8"),
  );
  const scoring = JSON.parse(
    readFileSync(new URL("../public/data/rapm/observable-scoring-channels-2024-2026.json", import.meta.url), "utf8"),
  );
  const finishing = JSON.parse(
    readFileSync(new URL("../public/data/rapm/observable-finish-channels-2024-2026.json", import.meta.url), "utf8"),
  );
  assert.equal(teammateEfg.length, 802);
  assert.equal(scoring.length, 802);
  assert.equal(finishing.length, 802);
  assert.ok(teammateEfg.every((row) => Number.isFinite(row.teammate_efg_net)));
  assert.ok(scoring.every((row) => Number.isFinite(row.rim_points_net)));
  assert.ok(finishing.every((row) => Number.isFinite(row.playtype_drive_points_net)));
});

test("research payload carries the same-game WP-RAPM comparison", () => {
  const curves = JSON.parse(
    readFileSync(new URL("../public/data/rapm/research-curves.json", import.meta.url), "utf8"),
  );
  const comparison = curves.wp_rapm_vs_pulse;
  assert.equal(comparison.games, 4911);
  assert.deepEqual(comparison.outcome_seasons, [2023, 2024, 2025, 2026]);
  assert.ok(
    comparison.summary.find((row) => row.model === "PULSE").equal_season_rmse <
      comparison.summary.find((row) => row.model === "WP-RAPM").equal_season_rmse,
  );
  assert.ok(
    comparison.summary.find((row) => row.model === "RAPM").equal_season_rmse <
      comparison.summary.find((row) => row.model === "WP-RAPM").equal_season_rmse,
  );
  assert.equal(
    comparison.paired_comparisons.find((row) => row.left === "WP-RAPM" && row.right === "RAPM").probability_left_better,
    0,
  );
});

test("public research payload excludes local season diagnostics", () => {
  const curves = JSON.parse(
    readFileSync(new URL("../public/data/rapm/research-curves.json", import.meta.url), "utf8"),
  );
  assert.equal("pulse_by_season" in curves, false);
});

test("every lazy RAPM shard exists", () => {
  for (const estimand of catalog.estimands) {
    for (const period of estimand.periods) {
      const relative = period.url.replace(/^\/data\//, "../public/data/");
      const payload = JSON.parse(readFileSync(new URL(relative, import.meta.url), "utf8"));
      assert.equal(payload.length, period.rows, `${estimand.id} ${period.id}`);
    }
  }
});
