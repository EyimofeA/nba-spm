import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const lab = JSON.parse(
  readFileSync(new URL("../local-data/matchup-lab.json", import.meta.url), "utf8"),
);

test("Matchup Lab stays local and carries the chronological comparison", () => {
  assert.equal(lab.scope, "localhost_only");
  assert.equal(lab.latest_season, 2026);
  assert.deepEqual(lab.quality, {
    points_conserved: true,
    shot_level_defender_assignments_invented: false,
    unique_game_scorer_defender_keys: true,
  });
  assert.ok(lab.players.length > 500);
  assert.ok(lab.channels.length > 3_000);
  assert.ok(lab.pairs.length > 1_000);
  assert.ok(lab.validation.some((row) => row.model === "contextual_hierarchical"));
  assert.ok(lab.validation.some((row) => row.model === "sequential_residual_elo"));
});

test("matchup components and unique scorer-defender pairs reconcile", () => {
  for (const row of lab.players) {
    for (const prefix of ["raw", "scorer_adjusted", "contextual", "sequential"]) {
      const offense = row[`${prefix}_offense`];
      const defense = row[`${prefix}_defense`];
      const net = row[`${prefix}_net`];
      if ([offense, defense, net].some((value) => value == null)) continue;
      assert.ok(Math.abs(offense + defense - net) < 2e-6);
    }
  }
  const keys = lab.pairs.map((row) => `${row.SCORER_ID}-${row.DEFENDER_ID}`);
  assert.equal(new Set(keys).size, keys.length);
});
