import { webcrypto } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

if (process.argv.length !== 4) {
  console.error("usage: node scripts/verify_cockpit_wasm.mjs <asset-dir> <trace.json>");
  process.exit(2);
}

globalThis.crypto ??= webcrypto;
const assetDirectory = path.resolve(process.argv[2]);
await import(pathToFileURL(path.join(assetDirectory, "wasm_exec.js")).href);

if (typeof globalThis.Go !== "function") {
  throw new Error("wasm_exec.js did not expose Go");
}
const go = new globalThis.Go();
const bytes = await readFile(path.join(assetDirectory, "verifier.wasm"));
const instance = await WebAssembly.instantiate(bytes, go.importObject);
void go.run(instance.instance);

for (let attempt = 0; attempt < 100; attempt += 1) {
  if (typeof globalThis.openManusVerifyTrace === "function") break;
  await new Promise((resolve) => setTimeout(resolve, 10));
}
if (typeof globalThis.openManusVerifyTrace !== "function") {
  throw new Error("WASM verifier did not initialize");
}

const trace = await readFile(path.resolve(process.argv[3]), "utf8");
const result = JSON.parse(globalThis.openManusVerifyTrace(trace));
console.log(JSON.stringify(result, null, 2));
process.exit(result.valid ? 0 : 1);
