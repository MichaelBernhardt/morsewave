/// <reference lib="webworker" />

import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import {
  conditionCwformerLane,
  CWFORMER_V020_MANIFEST,
  CwformerStreamingAdapter,
  parseNpyFloat32,
  type CwformerConfig,
  type CwformerSession,
  type CwformerTensorFactory,
} from "./cwformer";
import {
  NEURAL_ADVISORY_MAX_LANES,
  resampleForCwformer,
  type NeuralDecodeAdvisory,
  type NeuralWorkerRequest,
  type NeuralWorkerResponse,
} from "./neuralAdvisory";

const worker = self as DedicatedWorkerGlobalScope;
const CWFORMER_ASSET_DIRECTORY = "/models/cwformer-v0.2.0";

function post(message: NeuralWorkerResponse) {
  worker.postMessage(message);
}

function assetUrl(file: string) {
  return new URL(`${CWFORMER_ASSET_DIRECTORY}/${file}`, worker.location.href).href;
}

function bytesToHex(bytes: Uint8Array) {
  return [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function verifySha256(buffer: ArrayBuffer, expected: string, label: string) {
  // Neural copy is optional, so fail closed if a future/unsupported WebView
  // cannot verify the exact bundled model and frontend tables.
  if (!worker.crypto?.subtle) {
    throw new Error("Bundled CWformer assets cannot be integrity-checked in this WebView.");
  }
  const digest = await worker.crypto.subtle.digest("SHA-256", buffer);
  if (bytesToHex(new Uint8Array(digest)) !== expected) {
    throw new Error(`Bundled CWformer ${label} failed its integrity check.`);
  }
}

async function fetchBundledAsset(file: string, expectedSha256: string) {
  const response = await fetch(assetUrl(file), { cache: "force-cache", credentials: "same-origin" });
  if (!response.ok) throw new Error(`Bundled CWformer asset ${file} is unavailable.`);
  const buffer = await response.arrayBuffer();
  await verifySha256(buffer, expectedSha256, file);
  return buffer;
}

async function decodeNeuralAdvisories(request: NeuralWorkerRequest) {
  const lanes = request.lanes.slice(0, NEURAL_ADVISORY_MAX_LANES);
  if (!lanes.length) return [];
  const channels = request.channels.map((buffer) => new Float32Array(buffer));

  post({
    type: "progress",
    fraction: 0,
    detail: `Loading local CWformer ${CWFORMER_V020_MANIFEST.version} advisory`,
  });
  const [modelBuffer, configBuffer, basisBuffer, windowBuffer, wasmResponse] = await Promise.all([
    fetchBundledAsset(
      CWFORMER_V020_MANIFEST.modelFile,
      CWFORMER_V020_MANIFEST.sha256.model,
    ),
    fetchBundledAsset("mel_config.json", CWFORMER_V020_MANIFEST.sha256.config),
    fetchBundledAsset("mel_basis.npy", CWFORMER_V020_MANIFEST.sha256.melBasis),
    fetchBundledAsset("mel_window.npy", CWFORMER_V020_MANIFEST.sha256.melWindow),
    fetch(new URL(ortWasmUrl, worker.location.href), {
      cache: "force-cache",
      credentials: "same-origin",
    }),
  ]);
  if (modelBuffer.byteLength !== CWFORMER_V020_MANIFEST.modelBytes) {
    throw new Error("Bundled CWformer model size does not match the pinned release.");
  }
  if (!wasmResponse.ok) throw new Error("Bundled ONNX Runtime WASM is unavailable.");

  const config = JSON.parse(new TextDecoder().decode(configBuffer)) as CwformerConfig;
  if (config.sample_rate !== 16_000) {
    throw new Error("Bundled CWformer configuration has an unexpected sample rate.");
  }
  const melBasis = parseNpyFloat32(basisBuffer);
  const melWindow = parseNpyFloat32(windowBuffer);
  const ort = await import("onnxruntime-web/wasm");
  ort.env.logLevel = "error";
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmBinary = await wasmResponse.arrayBuffer();

  const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
    executionProviders: ["wasm"],
    executionMode: "sequential",
    graphOptimizationLevel: "all",
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  });
  const tensor: CwformerTensorFactory = (type, data, dims) => type === "float32"
    ? new ort.Tensor("float32", data as Float32Array, [...dims])
    : new ort.Tensor("int64", data as BigInt64Array, [...dims]);
  const adapter = new CwformerStreamingAdapter(
    session as unknown as CwformerSession,
    tensor,
    { config, melBasis: melBasis.data, melWindow: melWindow.data },
  );
  const advisories: NeuralDecodeAdvisory[] = [];
  try {
    for (let laneIndex = 0; laneIndex < lanes.length; laneIndex += 1) {
      const lane = lanes[laneIndex];
      const source = channels[lane.channel];
      if (!source?.length) continue;
      post({
        type: "progress",
        fraction: 0.15 + 0.8 * laneIndex / lanes.length,
        detail: `Neural lane ${laneIndex + 1}/${lanes.length} · ${Math.round(lane.frequency)} Hz`,
      });
      const started = performance.now();
      const resampled = resampleForCwformer(source, request.sampleRate, config.sample_rate);
      const conditioned = conditionCwformerLane(resampled, config.sample_rate, lane.frequency, {
        targetHz: 700,
        basebandHz: 180,
      });
      try {
        const decoded = await adapter.decodeAudio16k(conditioned);
        const transcript = decoded.fullText.replace(/\s+/g, " ").trim();
        if (!transcript || !decoded.tokens.length) continue;
        const meanTokenConfidence = decoded.tokens.reduce(
          (sum, token) => sum + token.confidence,
          0,
        ) / decoded.tokens.length;
        advisories.push({
          engine: "CWformer",
          engineVersion: "0.2.0",
          runtime: "ONNX Runtime Web 1.27.0 · WASM/1T",
          authoritative: false,
          transcript,
          tokens: decoded.tokens,
          channel: lane.channel,
          sourceFrequency: lane.frequency,
          conditionedFrequency: 700,
          sourceSampleRate: request.sampleRate,
          modelSampleRate: 16_000,
          durationSeconds: source.length / request.sampleRate,
          meanTokenConfidence,
          inferenceMs: performance.now() - started,
        });
      } catch (cause) {
        console.warn(`CWformer lane ${Math.round(lane.frequency)} Hz was skipped.`, cause);
      }
    }
  } finally {
    await adapter.release();
  }
  return advisories;
}

worker.onmessage = async (event: MessageEvent<NeuralWorkerRequest>) => {
  if (event.data?.type !== "decode") return;
  try {
    const advisories = await decodeNeuralAdvisories(event.data);
    post({ type: "result", advisories });
  } catch (cause) {
    post({
      type: "error",
      message: cause instanceof Error ? cause.message : "CWformer advisory failed.",
    });
  }
};

export {};
