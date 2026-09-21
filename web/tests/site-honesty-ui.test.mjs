import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

const ratings = readFileSync(new URL("../app/views/RatingsView.tsx", import.meta.url), "utf8");
const research = readFileSync(new URL("../app/views/ResearchView.tsx", import.meta.url), "utf8");
const player = readFileSync(new URL("../app/views/PlayerView.tsx", import.meta.url), "utf8");
const catalog = JSON.parse(
  readFileSync(new URL("../public/data/rapm/catalog.json", import.meta.url), "utf8"),
);
const rapmDir = new URL("../public/data/rapm/", import.meta.url);

test("Ratings board labels PULSE as descriptive refit with backcast seasons", () => {
  assert.match(ratings, /final descriptive mapping trained through 2026/);
  assert.match(ratings, /1997–2013 are backcast/);
  assert.match(ratings, /2014–2026[\s\S]*are refits/);
});

test("Research states both scoring targets and the rank reversal", () => {
  assert.match(research, /excludes technical free throws/);
  assert.match(research, /official final margins including technical free throws/);
  assert.match(research, /ranks fifth behind xRAPM, EPM, DARKO DPM, and LEBRON/);
});

test("public RAPM horizons name stint penalties", () => {
  for (const id of ["annual", "rolling-three", "rolling-five"]) {
    const note = catalog.estimands.find((item) => item.id === id).note;
    assert.match(note, /3000.*4500.*300/);
    assert.match(note, /Stint-aggregated/);
  }
  assert.match(catalog.estimands.find((item) => item.id === "annual").note, /Not terminal-lineup/);
});

test("player page date-stamps stale roles and softens factor copy", () => {
  assert.match(player, /roleSeason\.Season < currentSeason/);
  assert.match(player, /Factor allocation/);
  assert.doesNotMatch(player, /Where the rating comes from/);
  assert.match(player, /mixture changes when matchup/);
});

test("orphan age-conditioned and extra WP shards are not published", () => {
  const files = readdirSync(rapmDir);
  assert.ok(!files.some((name) => name.startsWith("full-history-actual-age-")));
  assert.deepEqual(
    files.filter((name) => name.startsWith("win-probability-") && name.endsWith(".json")).sort(),
    ["win-probability-2024.json", "win-probability-2025.json", "win-probability-2026.json"],
  );
});
