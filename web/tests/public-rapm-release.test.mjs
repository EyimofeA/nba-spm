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
    "same-age-27", "full-history-actual-age", "current-time-decay",
    "current-age-time-decay", "luck-adjusted", "coach", "win-probability", "units",
  ]) assert.ok(ids.includes(required), required);
  assert.ok(ids.some((id) => id.startsWith("factor-shooting_ts-")));
  assert.ok(ids.some((id) => id.startsWith("factor-opponent_oreb_prevention-")));
  assert.ok(!ids.some((id) => /lambda|sweep|old-vs|rerun/i.test(id)));
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
