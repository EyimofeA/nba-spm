import { mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const output = resolve(root, "dist/pages");
const upstream = "https://nba-impact-lab.nba-impact-lab.workers.dev";

// Pages gives us the short free hostname. The production Worker remains the
// single tested application origin, so every HTML, script, and data request is
// forwarded as one coherent release rather than splitting a server-rendered
// build across two asset systems.
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await writeFile(
  resolve(output, "_worker.js"),
  `const upstream = ${JSON.stringify(upstream)};\n\nexport default {\n  async fetch(request) {\n    const target = new URL(request.url);\n    target.protocol = new URL(upstream).protocol;\n    target.hostname = new URL(upstream).hostname;\n    target.port = "";\n    const response = await fetch(new Request(target, request));\n    const headers = new Headers(response.headers);\n    if (target.pathname.startsWith("/_next/static/")) {\n      headers.set("Cache-Control", "public, max-age=31536000, immutable");\n    } else if (target.pathname.startsWith("/data/")) {\n      headers.set("Cache-Control", "public, max-age=3600, stale-while-revalidate=86400");\n    }\n    return new Response(response.body, {\n      status: response.status,\n      statusText: response.statusText,\n      headers,\n    });\n  },\n};\n`,
);
