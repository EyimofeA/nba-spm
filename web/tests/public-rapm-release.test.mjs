import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const catalog = JSON.parse(
  readFileSync(new URL("../public/data/rapm/catalog.json", import.meta.url), "utf8"),
);

const REQUIRED = [
  "annual",
  "rolling-three",
  "rolling-five",
  "full-history",
  "win-probability",
];
const FORBIDDEN = [
  "same-age-27",
  "full-history-actual-age",
  "current-time-decay",
  "current-age-time-decay",
  "luck-adjusted",
  "coach",
  "units",
  "wp_pulse",
];

test("public RAPM release keeps the published contract and omits rejected lab boards", () => {
  const ids = catalog.estimands.map((estimand) => estimand.id);
  for (const required of REQUIRED) assert.ok(ids.includes(required), required);
  for (const forbidden of FORBIDDEN) {
    assert.ok(!ids.includes(forbidden), `unpublished estimand leaked: ${forbidden}`);
  }
  assert.ok(ids.some((id) => id.startsWith("factor-shooting_ts-")));
  assert.ok(ids.some((id) => id.startsWith("factor-opponent_oreb_prevention-")));
  assert.ok(!ids.some((id) => /lambda|sweep|old-vs|rerun/i.test(id)));
});

test("rejected RAPM lab shards are not fetchable", () => {
  const leaked = [
    "same-age-27-1997-2026.json",
    "current-time-decay-2022-2026.json",
    "current-age-time-decay-2022-2026.json",
    "luck-adjusted-2024-2026.json",
    "coach-1997-2026.json",
    "units-2.json",
    "research-curves.json",
  ];
  for (const name of leaked) {
    const path = new URL(`../public/data/rapm/${name}`, import.meta.url);
    assert.equal(existsSync(path), false, name);
  }
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
