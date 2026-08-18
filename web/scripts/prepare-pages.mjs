import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const output = resolve(root, "dist/pages");

// Pages needs one deployable directory. The existing server worker remains the
// renderer; Pages supplies its static assets through env.ASSETS.
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await cp(resolve(root, "dist/client"), output, { recursive: true });
await cp(resolve(root, "dist/server"), resolve(output, "server"), {
  recursive: true,
});
// This is Wrangler's generated Worker deployment config. It is not runtime
// code, and Pages otherwise mistakes it for the Pages project configuration.
await rm(resolve(output, "server/wrangler.json"));
await writeFile(
  resolve(output, "_worker.js"),
  'import worker from "./server/index.js";\n\nexport default worker;\n',
);
await writeFile(
  resolve(output, ".assetsignore"),
  "_worker.js\nserver/**\n",
);
