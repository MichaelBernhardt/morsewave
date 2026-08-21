import { describe, expect, it } from "vitest";
import {
  CWFORMER_LIMITS,
  CWFORMER_V020_MANIFEST,
  CWFORMER_VOCABULARY,
  CwformerMelFrontend,
  CwformerStreamingAdapter,
  conditionCwformerLane,
  parseNpyFloat32,
  type CwformerConfig,
  type CwformerFrontendAssets,
  type CwformerOutputTensor,
  type CwformerSession,
  type CwformerTensorData,
} from "./cwformer";

interface TestTensor extends CwformerOutputTensor {
  type: "float32" | "int64";
}

const tensor = (
  type: "float32" | "int64",
  data: CwformerTensorData,
  dims: readonly number[],
): TestTensor => ({ type, data, dims: [...dims] });

const tinyConfig: CwformerConfig = {
  sample_rate: 16,
  n_fft: 8,
  hop_length: 4,
  n_mels: 4,
  f_min: 1,
  f_max: 8,
  n_layers: 2,
  n_heads: 1,
  d_model: 4,
  d_k: 2,
  conv_kernel: 3,
  max_cache_len: 2,
  subsample_channels: 2,
};

function tinyAssets(): CwformerFrontendAssets {
  const bins = tinyConfig.n_fft / 2 + 1;
  const basis = new Float32Array(tinyConfig.n_mels * bins);
  for (let mel = 0; mel < tinyConfig.n_mels; mel += 1) basis[mel * bins + mel] = 1;
  return {
    config: tinyConfig,
    melBasis: basis,
    melWindow: new Float32Array(tinyConfig.n_fft).fill(1),
  };
}

function npyFloat32(values: number[], shape: number[]) {
  const dictionary = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}${
    shape.length === 1 ? "," : ""
  }), }`;
  const preamble = 10;
  const padding = (16 - (preamble + dictionary.length + 1) % 16) % 16;
  const header = `${dictionary}${" ".repeat(padding)}\n`;
  const buffer = new ArrayBuffer(preamble + header.length + values.length * 4);
  const bytes = new Uint8Array(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(buffer).setUint16(8, header.length, true);
  bytes.set(new TextEncoder().encode(header), preamble);
  const view = new DataView(buffer, preamble + header.length);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return buffer;
}

function logProbs(indices: number[]) {
  const data = new Float32Array(indices.length * CWFORMER_VOCABULARY.length).fill(-10);
  indices.forEach((winner, frame) => {
    data[frame * CWFORMER_VOCABULARY.length + winner] = Math.log(0.9);
    data[frame * CWFORMER_VOCABULARY.length + (winner + 1) % CWFORMER_VOCABULARY.length] =
      Math.log(0.05);
  });
  return tensor("float32", data, [indices.length, 1, CWFORMER_VOCABULARY.length]);
}

function toneAmplitude(audio: Float32Array, frequency: number, sampleRate: number, start = 0) {
  let real = 0;
  let imaginary = 0;
  for (let index = start; index < audio.length; index += 1) {
    const angle = 2 * Math.PI * frequency * index / sampleRate;
    real += audio[index] * Math.cos(angle);
    imaginary -= audio[index] * Math.sin(angle);
  }
  return 2 * Math.hypot(real, imaginary) / (audio.length - start);
}

class MockSession implements CwformerSession {
  readonly feeds: Array<Readonly<Record<string, unknown>>> = [];
  released = false;
  private runIndex = 0;

  async run(feeds: Readonly<Record<string, unknown>>) {
    this.feeds.push(feeds);
    const outputs: Record<string, TestTensor> = {
      log_probs: this.runIndex++ === 0
        ? logProbs([0, 2, 2, 0, 2])
        : logProbs([2, 0, 1, 3]),
      pos_offset_out: tensor("int64", new BigInt64Array([1n]), [1]),
      sub_buf1_out: feeds.sub_buf1 as TestTensor,
      sub_buf2_out: feeds.sub_buf2 as TestTensor,
    };
    for (let layer = 0; layer < (tinyConfig.n_layers ?? 0); layer += 1) {
      const cache = new Float32Array(1 * 1 * 3 * 2);
      for (let frame = 0; frame < 3; frame += 1) {
        cache[frame * 2] = frame * 10 + layer;
        cache[frame * 2 + 1] = frame * 10 + layer + 1;
      }
      outputs[`kv_k_layer${layer}_out`] = tensor("float32", cache, [1, 1, 3, 2]);
      outputs[`kv_v_layer${layer}_out`] = tensor("float32", cache.slice(), [1, 1, 3, 2]);
      outputs[`conv_buf_layer${layer}_out`] = feeds[`conv_buf_layer${layer}`] as TestTensor;
    }
    return outputs;
  }

  release() {
    this.released = true;
  }
}

describe("CWformer streaming adapter", () => {
  it("pins the audited v0.2.0 int8 release", () => {
    expect(CWFORMER_V020_MANIFEST).toMatchObject({
      version: "0.2.0",
      sourceCommit: "137b500b30efdcb2ea356979044fee42f44327f2",
      modelBytes: 24_953_338,
      sha256: {
        model: "3f45231d4064fab08d96f0e328833128339fb0df1f8e2a883e1ca3cc3c124b4c",
      },
    });
  });

  it("parses release-style little-endian float32 NPY tables", () => {
    const parsed = parseNpyFloat32(npyFloat32([1.25, -2.5, 3.75, 4], [2, 2]));
    expect(parsed.shape).toEqual([2, 2]);
    expect([...parsed.data]).toEqual([1.25, -2.5, 3.75, 4]);
    expect(() => parseNpyFloat32(new ArrayBuffer(12))).toThrow(/Invalid NPY/);
  });

  it("matches the reference streaming STFT frame grid and silence floor", () => {
    const frontend = new CwformerMelFrontend(tinyAssets());
    const first = frontend.computeStreaming(new Float32Array(8));
    const second = frontend.computeStreaming(new Float32Array(8));

    expect(first.frameCount).toBe(2);
    expect(second.frameCount).toBe(2);
    expect(first.data).toHaveLength(8);
    expect([...first.data]).toEqual(
      Array.from({ length: 8 }, () => expect.closeTo(Math.log(1e-6), 5)),
    );
  });

  it("computes the exact-length spectrum without zero-padding the release window", () => {
    const frontend = new CwformerMelFrontend(tinyAssets());
    const audio = Float32Array.from([0.25, -0.5, 0.75, 0.1, -0.2, 0.4, -0.6, 0.3]);
    const result = frontend.computeStreaming(audio);
    const padded = Float32Array.from([0, 0, 0, 0, ...audio]);
    const expected: number[] = [];
    for (const start of [0, 4]) {
      for (let bin = 0; bin < tinyConfig.n_mels; bin += 1) {
        let real = 0;
        let imaginary = 0;
        for (let sample = 0; sample < tinyConfig.n_fft; sample += 1) {
          const angle = 2 * Math.PI * bin * sample / tinyConfig.n_fft;
          real += padded[start + sample] * Math.cos(angle);
          imaginary -= padded[start + sample] * Math.sin(angle);
        }
        expected.push(Math.log(real * real + imaginary * imaginary + 1e-6));
      }
    }
    expect([...result.data]).toEqual(expected.map((value) => expect.closeTo(value, 5)));
  });

  it("isolates a physical carrier lane and translates it to the model's nominal pitch", () => {
    const sampleRate = 16_000;
    const audio = new Float32Array(sampleRate);
    for (let index = 0; index < audio.length; index += 1) {
      audio[index] = 0.2 * Math.cos(2 * Math.PI * 1_050 * index / sampleRate) +
        0.8 * Math.cos(2 * Math.PI * 1_850 * index / sampleRate);
    }
    const conditioned = conditionCwformerLane(audio, sampleRate, 1_050);
    const settled = Math.round(sampleRate * 0.1);
    expect(toneAmplitude(conditioned, 700, sampleRate, settled)).toBeCloseTo(0.2, 2);
    expect(toneAmplitude(conditioned, 1_500, sampleRate, settled)).toBeLessThan(0.005);
    expect(() => conditionCwformerLane(audio, sampleRate, 20, { basebandHz: 180 }))
      .toThrow(/frequencies/);
  });

  it("carries model state, bounds KV caches, and collapses CTC across chunks", async () => {
    const session = new MockSession();
    const adapter = new CwformerStreamingAdapter(session, tensor, tinyAssets(), 500);

    const first = await adapter.feedAudio16k(new Float32Array(8));
    const second = await adapter.feedAudio16k(new Float32Array(8));

    expect(first.newText).toBe("AA");
    expect(second.newText).toBe(" B");
    expect(second.fullText).toBe("AA B");
    expect(second.tokens.at(-1)).toMatchObject({
      value: "B",
      classIndex: 3,
      confidence: expect.closeTo(0.9, 5),
    });
    expect((session.feeds[0].mel_chunk as TestTensor).dims).toEqual([1, 2, 4]);
    expect((session.feeds[0].kv_k_layer0 as TestTensor).dims).toEqual([1, 1, 0, 2]);
    expect((session.feeds[1].kv_k_layer0 as TestTensor).dims).toEqual([1, 1, 2, 2]);
    expect([...(session.feeds[1].kv_k_layer0 as TestTensor).data as Float32Array])
      .toEqual([10, 11, 20, 21]);
  });

  it("peak-normalizes file input, enforces sequential ownership, and releases cleanly", async () => {
    const session = new MockSession();
    const adapter = new CwformerStreamingAdapter(session, tensor, tinyAssets(), 500);
    const result = await adapter.decodeAudio16k(new Float32Array(9).fill(0.1));
    expect(result.fullText.length).toBeGreaterThan(0);
    await adapter.release();
    expect(session.released).toBe(true);
    await expect(adapter.feedAudio16k(new Float32Array(8))).rejects.toThrow(/released/);
  });

  it("fails closed on invalid assets, model outputs, and overlong files", async () => {
    expect(() => new CwformerMelFrontend({
      ...tinyAssets(),
      melWindow: new Float32Array(7),
    })).toThrow(/mel tables/);

    const brokenSession: CwformerSession = { run: async () => ({}) };
    const adapter = new CwformerStreamingAdapter(brokenSession, tensor, tinyAssets(), 500);
    await expect(adapter.feedAudio16k(new Float32Array(8))).rejects.toThrow(/log_probs/);

    const bounded = new CwformerStreamingAdapter(new MockSession(), tensor, tinyAssets(), 500);
    const overlong = new Float32Array(
      tinyConfig.sample_rate * CWFORMER_LIMITS.maximumFileSeconds + 1,
    );
    await expect(bounded.decodeAudio16k(overlong)).rejects.toThrow(/30-minute/);
  });
});
