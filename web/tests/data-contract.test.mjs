import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (name) => JSON.parse(readFileSync(new URL(`../public/data/${name}`, import.meta.url)));
const catalog = read("catalog.json");
const client = readFileSync(new URL("../app/PlayerLab.tsx", import.meta.url), "utf8");

test("the catalog offers exactly the models the client can render", () => {
  assert.deepEqual(
    catalog.catalog.models.map(({ id, prefix }) => ({ id, prefix })),
    [
      { id: "aio", prefix: "aio_" },
      { id: "rapm", prefix: "normal_rapm_" },
      { id: "spm", prefix: "spm_" },
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

test("external correlations match the verified benchmark runs", () => {
  const rows = catalog.validation.external_benchmark.rows;
  const find = (scope, component) => rows.find((row) => row.scope === scope && row.component === component);
  assert.deepEqual(
    (({ players, bpm, xrapm }) => ({ players, bpm, xrapm }))(find("Three-season SPM windows", "net")),
    { players: 2295, bpm: 0.876, xrapm: 0.756 },
  );
  assert.equal(find("Three-season SPM windows", "defense").xrapm, 0.63);
  assert.deepEqual(
    (({ players, bpm, xrapm }) => ({ players, bpm, xrapm }))(find("Annual SPM baseline", "net")),
    { players: 2860, bpm: 0.897, xrapm: 0.762 },
  );
  assert.equal(find("Annual SPM plus tracking", "defense").xrapm, 0.701);
  for (const row of rows) assert.ok(row.run_id.length > 0, "every external row names its run");
});

test("every published season table is loadable and complete", () => {
  assert.deepEqual(catalog.catalog.seasons, [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]);
  for (const season of catalog.catalog.seasons) {
    const rows = read(`leaderboard-${season}.json`);
    assert.ok(rows.length > 100, `season ${season} is too small`);
    for (const row of rows) {
      assert.equal(row.Season, season);
      assert.equal(typeof row.PLAYER_NAME, "string");
      assert.equal(typeof row.aio_net, "number");
      assert.ok(Math.abs(row.aio_offense + row.aio_defense - row.aio_net) < 0.001, "offense plus defense must equal net");
      assert.ok(row.Poss_Off >= 0 && row.Poss_Def >= 0);
    }
  }
});

test("projection vintages are published only when both player and team rows exist", () => {
  const teams = read("projection-teams.json");
  const players = read("projection-players.json");
  const teamSeasons = [...new Set(teams.map((row) => row.projection_season))].sort();
  const playerSeasons = [...new Set(players.map((row) => row.projection_season))].sort();
  assert.deepEqual(teamSeasons, playerSeasons);
  assert.deepEqual(teamSeasons, [2027]);
  assert.ok(players.every((row) => row.projection_season === 2027));
});

test("the player index points at existing shards", () => {
  const index = read("players.json");
  assert.ok(index.length > 1000);
  for (const item of index) assert.ok(item.shard >= 0 && item.shard < catalog.shards);
  const shard = read(`ratings-${String(index[0].shard).padStart(2, "0")}.json`);
  assert.ok(shard[String(index[0].id)].annual.length > 0);
});
