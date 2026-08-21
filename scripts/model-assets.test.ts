import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { CWFORMER_V020_MANIFEST } from "../src/audio/offline/cwformer";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const modelDirectory = resolve(repository, "public/models/cwformer-v0.2.0");

function sha256(file: string) {
  return createHash("sha256").update(readFileSync(resolve(modelDirectory, file))).digest("hex");
}

describe("bundled CWformer assets", () => {
  it("matches every pinned release hash and the audited model size", () => {
    expect(statSync(resolve(modelDirectory, CWFORMER_V020_MANIFEST.modelFile)).size)
      .toBe(CWFORMER_V020_MANIFEST.modelBytes);
    expect(sha256(CWFORMER_V020_MANIFEST.modelFile)).toBe(CWFORMER_V020_MANIFEST.sha256.model);
    expect(sha256("mel_config.json")).toBe(CWFORMER_V020_MANIFEST.sha256.config);
    expect(sha256("mel_basis.npy")).toBe(CWFORMER_V020_MANIFEST.sha256.melBasis);
    expect(sha256("mel_window.npy")).toBe(CWFORMER_V020_MANIFEST.sha256.melWindow);
  });
});
