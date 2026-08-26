import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";

const root = process.cwd();
const readJson = async (path) => JSON.parse(await readFile(path, "utf8"));

test("localhost skill shards expose the frozen 2026 research result", async () => {
  const index = await readJson(join(root, "local-data/skills/index.json"));
  assert.equal(index.schema, "courtsignal_local_player_skills_v1");
  assert.equal(index.scope, "localhost_only");
  assert.equal(index.definitions.length, 34);
  assert.equal(index.players.length, 558);
  assert.equal(index.seasons.at(-1), 2026);
  assert.ok(!index.seasons.includes(2027));
  const defaultPlayer = index.players.find((player) => player.id === index.defaultPlayerId);
  assert.ok(defaultPlayer?.complete2026);

  const player = await readJson(
    join(root, `local-data/skills/player-${index.defaultPlayerId}.json`),
  );
  assert.equal(Object.keys(player.skills).length, 34);
  assert.equal(player.profiles.at(-1).season, 2026);
  assert.equal(Object.keys(player.profiles.at(-1)).length, 13);
  assert.ok(player.games.free_throw_pct.length > 0);
  assert.ok(player.games.three_point_pct.length > 0);
  for (const skill of ["free_throw_pct", "three_point_pct"]) {
    const annual = player.skills[skill].rows.find((row) => row[0] === 2026);
    const lastGame = player.games[skill].filter((game) => game.played).at(-1);
    assert.ok(annual && lastGame);
    assert.ok(Math.abs(annual[1] - lastGame.estimate) <= 0.0002);
    assert.equal(annual[7], lastGame.date);
  }
  assert.ok(
    player.games.free_throw_pct
      .filter((game) => !game.played)
      .every((game) => game.raw === null && game.estimate === null),
  );
});

test("production build does not ship localhost skill shards", async () => {
  await assert.rejects(readFile(join(root, "dist/client/data/skills/index.json")));
});
