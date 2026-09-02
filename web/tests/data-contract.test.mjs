import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

const read = (name) => JSON.parse(readFileSync(new URL(`../public/data/${name}`, import.meta.url)));
const catalog = read("catalog.json");
const { resolveModel } = await import("../app/lib/data.ts");

/** Every client source concatenated, so these checks cover the whole app. */
function sources(dir) {
  let text = "";
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) text += sources(new URL(`${entry.name}/`, dir));
    else if (/\.tsx?$/.test(entry.name)) text += readFileSync(new URL(entry.name, dir), "utf8");
  }
  return text;
}
const client = sources(new URL("../app/", import.meta.url));

test("the catalog offers exactly the models the client can render", () => {
  assert.deepEqual(
    catalog.catalog.models.map(({ id, prefix }) => ({ id, prefix })),
    [
      { id: "pulse", prefix: "pulse_" },
      { id: "rapm", prefix: "rapm_" },
    ],
  );
  for (const model of catalog.catalog.models) {
    assert.ok(client.includes(`prefix: "${model.prefix}"`), `client is missing ${model.prefix}`);
  }
});

test("the snapshot carries no win probability and no stabilized roles", () => {
  const payload = JSON.stringify(catalog);
  assert.ok(!("win_probability" in catalog.validation));
  assert.doesNotMatch(payload, /brier|espn|stabiliz/i);
  assert.doesNotMatch(client, /win probability|brier/i);
  assert.doesNotMatch(JSON.stringify(read("roles-offense-2024.json")), /stable_role/);
  assert.doesNotMatch(JSON.stringify(read("ratings-00.json")), /stabilized/);
});

test("an unavailable model falls back to a model carried by the season", () => {
  assert.equal(resolveModel(read("leaderboard-2026.json"), "unknown").id, "pulse");
  assert.equal(resolveModel(read("leaderboard-2024.json"), "rapm").id, "rapm");
});

test("every production payload matches the release manifest", () => {
  const manifest = read("snapshot-manifest.json");
  assert.equal(manifest.schema_version, "nba_impact_release_v1");
  for (const [name, expected] of Object.entries(manifest.files)) {
    const payload = readFileSync(new URL(`../public/data/${name}`, import.meta.url));
    assert.equal(payload.byteLength, expected.bytes, `${name} byte count changed`);
    assert.equal(
      createHash("sha256").update(payload).digest("hex"),
      expected.sha256,
      `${name} hash changed`,
    );
  }
});

test("season tables expose only the validated model coverage", () => {
  const seasons = Array.from({ length: 30 }, (_, index) => 1997 + index);
  assert.deepEqual(catalog.catalog.seasons, seasons);
  assert.deepEqual(
    Object.fromEntries(catalog.catalog.models.map((model) => [model.id, model.seasons])),
    {
      pulse: seasons,
      rapm: seasons,
    },
  );
  for (const season of catalog.catalog.seasons) {
    const rows = read(`leaderboard-${season}.json`);
    assert.ok(rows.length > 100, `season ${season} is too small`);
    for (const row of rows) {
      assert.equal(row.Season, season);
      assert.equal(typeof row.PLAYER_NAME, "string");
      for (const prefix of ["rapm_", "pulse_", "pulse_prior_", "lineup_update_"]) {
        assert.equal(typeof row[`${prefix}net`], "number");
        assert.ok(
          Math.abs(row[`${prefix}offense`] + row[`${prefix}defense`] - row[`${prefix}net`]) < 0.001,
          `${prefix} offense plus defense must equal net`,
        );
      }
      for (const component of ["offense", "defense", "net"]) {
        assert.ok(
          Math.abs(row[`pulse_prior_${component}`] + row[`lineup_update_${component}`] - row[`pulse_${component}`]) < 0.001,
          `PULSE prior plus lineup update must equal PULSE ${component}`,
        );
      }
      assert.ok(row.Poss_Off >= 0 && row.Poss_Def >= 0);
    }
  }
});

test("production data excludes local-only research payloads", () => {
  const names = readdirSync(new URL("../public/data/", import.meta.url));
  assert.ok(!names.some((name) => /matchup-elo|shot-quality-lineup|rapm-lab|spm-lab|projection-/.test(name)));
});

test("the player index points at existing shards", () => {
  const index = read("players.json");
  assert.ok(index.length > 1000);
  for (const item of index) assert.ok(item.shard >= 0 && item.shard < catalog.shards);
  const shard = read(`ratings-${String(index[0].shard).padStart(2, "0")}.json`);
  assert.ok(shard[String(index[0].id)].annual.length > 0);
});

test("historical player metadata survives snapshot generation", () => {
  const historicalSeasons = new Set(catalog.catalog.seasons);
  const profileSeasons = new Set(Array.from({ length: 13 }, (_, index) => 2014 + index));
  let historicalRows = 0;
  let rowsWithTeam = 0;
  let eligibleProfileRows = 0;
  let profileRows = 0;
  for (const shard of new Set(read("players.json").map(({ shard }) => shard))) {
    for (const player of Object.values(read(`ratings-${String(shard).padStart(2, "0")}.json`))) {
      for (const row of player.annual) {
        if (!historicalSeasons.has(row.Season)) continue;
        historicalRows += 1;
        rowsWithTeam += Number(typeof row.TEAM_ABBREVIATION === "string" && row.TEAM_ABBREVIATION.length > 0);
        eligibleProfileRows += Number(profileSeasons.has(row.Season));
      }
      profileRows += player.profiles.filter(({ Season }) => profileSeasons.has(Season)).length;
    }
  }
  assert.ok(rowsWithTeam / historicalRows >= 0.98, "historical team labels disappeared");
  assert.ok(profileRows / eligibleProfileRows >= 0.98, "2014–26 player profiles disappeared");
});
