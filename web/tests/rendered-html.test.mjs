import assert from "node:assert/strict";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the compact ratings product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>NBA Impact Lab<\/title>/i);
  assert.match(html, /Loading/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);

  for (const section of ["home", "ratings", "landscape", "player", "roles", "research"]) {
    assert.match(html, new RegExp(`href="#${section}"`), `missing ${section} section link`);
  }
  assert.match(html, /points per 100 possessions/i);
  assert.doesNotMatch(html, /win probability|brier|stable role/i);
});
