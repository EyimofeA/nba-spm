import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createHash } from "node:crypto";

const catalog = JSON.parse(
  readFileSync(new URL("../public/data/rapm/catalog.json", import.meta.url), "utf8"),
);

test("public RAPM release contains selected estimands and no tuning sweep", () => {
  const ids = catalog.estimands.map((estimand) => estimand.id);
  for (const required of [
    "annual", "rolling-three", "rolling-five", "current-time-decay",
    "luck-adjusted", "coach", "win-probability",
    "log-odds-win-probability", "units",
    "game-level-pm", "point-channels", "six-factor-annual", "teammate-effects",
    "teammate-efg", "observable-scoring-channels", "observable-finish-channels",
  ]) assert.ok(ids.includes(required), required);
  assert.ok(ids.some((id) => id.startsWith("factor-shooting_ts-")));
  assert.ok(ids.some((id) => id.startsWith("factor-opponent_oreb_prevention-")));
  assert.ok(!ids.some((id) => id.includes("age") || id === "full-history"));
  assert.ok(!ids.some((id) => /lambda|sweep|old-vs|rerun/i.test(id)));
  assert.match(catalog.estimands.find((item) => item.id === "luck-adjusted").note, /not the reference RAPM/);
});

test("WP-RAPM publishes only the latest three rolling endpoints", () => {
  const wp = catalog.estimands.find((estimand) => estimand.id === "win-probability");
  assert.deepEqual(wp.periods.map((period) => period.id), ["2024", "2025", "2026"]);
  assert.match(wp.note, /rolling five-season fit/);
  assert.match(wp.unit, /0–1 scale/);
  assert.doesNotMatch(wp.unit, /percentage points/);
  assert.match(wp.note, /overtime is folded into final regulation credit/);
});

test("log-odds WP-RAPM is labeled as a descriptive three-season rating", () => {
  const wp = catalog.estimands.find((estimand) => estimand.id === "log-odds-win-probability");
  assert.deepEqual(wp.periods.map((period) => period.id), ["2024", "2025", "2026"]);
  assert.match(wp.note, /descriptive/i);
  assert.match(wp.note, /not a forecast/i);
  const current = JSON.parse(
    readFileSync(new URL("../public/data/rapm/log-odds-win-probability-2026.json", import.meta.url), "utf8"),
  );
  assert.ok(current.every((row) => row.Poss_Off > 0 && row.Poss_Def > 0));
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
  assert.equal(comparison.run_id, catalog.lineage.wp_run);
  assert.match(comparison.run_id, /^wp_chronology_release_v2_/);
  assert.match(comparison.target, /Official final game margin/);
  assert.deepEqual(comparison.outcome_seasons, [2022, 2023, 2024, 2025, 2026]);
  assert.deepEqual(comparison.summary.map((row) => row.model).sort(), ["Log-odds WP-RAPM", "PULSE", "RAPM", "WP-RAPM"]);
  assert.ok(comparison.games > 0);
  assert.ok(comparison.summary.every((row) => row.games === comparison.games && row.folds === 5 && Number.isFinite(row.equal_season_rmse)));
  assert.match(comparison.warning, /earlier outcomes only/);
  assert.match(comparison.warning, /Reused historical/);
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
      const relative = period.url.split("?")[0].replace(/^\/data\//, "../public/data/");
      const payload = JSON.parse(readFileSync(new URL(relative, import.meta.url), "utf8"));
      assert.equal(payload.length, period.rows, `${estimand.id} ${period.id}`);
    }
  }
});

test("RAPM release bytes match the manifest", () => {
  const version = readFileSync(new URL("../app/lib/rapmRelease.ts", import.meta.url), "utf8");
  assert.ok(version.includes(JSON.stringify(catalog.lineage.wp_run)));
  for (const [name, expected] of Object.entries(catalog.files)) {
    const bytes = readFileSync(new URL(`../public/data/rapm/${name}`, import.meta.url));
    assert.equal(bytes.length, expected.bytes, name);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected.sha256, name);
  }
});

test("WP public coefficients have named players and additive totals", () => {
  for (const estimand of catalog.estimands.filter((item) => ["win-probability", "log-odds-win-probability"].includes(item.id))) {
    for (const period of estimand.periods) {
      assert.match(period.url, /\?v=[a-f0-9]{12}$/);
      const url = new URL(period.url, "https://courtsignalnba.pages.dev");
      assert.equal(url.searchParams.get("v"), catalog.files[url.pathname.split("/").at(-1)].sha256.slice(0, 12));
      const relative = period.url.split("?")[0].replace(/^\/data\//, "../public/data/");
      const rows = JSON.parse(readFileSync(new URL(relative, import.meta.url), "utf8"));
      assert.ok(rows.every((row) => row.PLAYER_NAME && row.Poss_Off > 0 && row.Poss_Def > 0));
      assert.ok(rows.every((row) => [row.offense, row.defense, row.net].every(Number.isFinite) && Math.abs(row.offense + row.defense - row.net) < .0002));
      assert.ok(rows.every((row) => estimand.id === "win-probability"
        ? row.window_end === Number(period.id) && row.window_start === Number(period.id) - 4
        : row.Season === Number(period.id)));
    }
  }
});
