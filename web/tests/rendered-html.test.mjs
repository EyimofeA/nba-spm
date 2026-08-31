import assert from "node:assert/strict";
import test from "node:test";

async function request(path = "/", fetchAsset = async () => new Response("Not found", { status: 404 })) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: fetchAsset } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the compact ratings product shell", async () => {
  const response = await request();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>CourtSignal<\/title>/i);
  assert.match(html, /Loading/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);

  for (const section of ["ratings", "roles", "research"]) {
    assert.match(html, new RegExp(`href="#${section}"`), `missing ${section} section link`);
  }
  assert.doesNotMatch(html, /href="#home"/i);
  assert.match(html, /NBA IMPACT/i);
  assert.match(html, /Methodology/i);
  assert.doesNotMatch(html, /href="#matchups"/i);
  assert.match(html, /points per 100 possessions/i);
  assert.doesNotMatch(html, /win probability|brier|stable role/i);
});

test("production JSON is cached without caching missing local research", async () => {
  const response = await request(
    "/data/catalog.json",
    async () => new Response("{}", { headers: { "content-type": "application/json" } }),
  );
  assert.equal(response.headers.get("cache-control"), "public, max-age=3600, stale-while-revalidate=86400");

  const missing = await request("/data/rapm-lab.json");
  assert.equal(missing.status, 404);
  assert.equal(missing.headers.get("cache-control"), null);
});
