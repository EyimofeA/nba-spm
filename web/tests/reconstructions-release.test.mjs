import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

const catalog = JSON.parse(
  readFileSync(new URL("../public/data/reconstructions/catalog.json", import.meta.url), "utf8"),
);

test("public reconstructions contain only CourtSignal outputs", () => {
  assert.equal(catalog.schema, "courtsignal_reconstructions_v1");
  assert.deepEqual(
    catalog.replications.map((row) => row.metric),
    [
      "CourtSignal DARKO WOWY reconstruction",
      "CourtSignal RAPTOR reconstruction",
      "CourtSignal PIPM reconstruction",
    ],
  );
  assert.equal(catalog.boards.length, 113);
  assert.ok(catalog.boards.every((board) => board.source === "CourtSignal reconstruction"));
  assert.deepEqual(
    catalog.boards.filter((board) => board.metric.includes("RAPTOR")).map((board) => board.season).sort((a, b) => a - b),
    [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
  );
});

test("reconstruction shards match their catalog receipts", () => {
  for (const board of catalog.boards) {
    const filename = board.url.split("/").at(-1).split("?")[0];
    const bytes = readFileSync(new URL(`../public/data/reconstructions/${filename}`, import.meta.url));
    const receipt = catalog.files[filename];
    assert.equal(bytes.length, receipt.bytes, filename);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), receipt.sha256, filename);
    assert.equal(JSON.parse(bytes).length, board.rows, filename);
    assert.equal(new URL(board.url, "https://courtsignalnba.pages.dev").searchParams.get("v"), receipt.sha256.slice(0, 12));
  }
});

test("public reconstruction rows do not ship source ratings", () => {
  for (const board of catalog.boards) {
    const filename = board.url.split("/").at(-1).split("?")[0];
    const payload = readFileSync(new URL(`../public/data/reconstructions/${filename}`, import.meta.url), "utf8");
    assert.doesNotMatch(payload, /reference_|official|xrapm|pipm_net|bpm_net/i, filename);
  }
});
