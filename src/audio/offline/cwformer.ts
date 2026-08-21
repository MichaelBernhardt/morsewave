/*
 * Streaming contract adapted from CWformer v0.2.0 inference_onnx.py.
 *
 * MIT License
 * Copyright (c) 2026 parsimo2010
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

export const CWFORMER_V020_MANIFEST = Object.freeze({
  version: "0.2.0",
  upstream: "https://github.com/parsimo2010/CWformer",
  release: "https://github.com/parsimo2010/CWformer/releases/tag/v0.2.0",
  sourceCommit: "137b500b30efdcb2ea356979044fee42f44327f2",
  license: "MIT",
  modelFile: "cwformer_streaming_int8.onnx",
  modelBytes: 24_953_338,
  sha256: Object.freeze({
    model: "3f45231d4064fab08d96f0e328833128339fb0df1f8e2a883e1ca3cc3c124b4c",
    config: "9d9688b9dc8fba976e97177116611df283b45e3b260cb5742accfa53dc13b66a",
    melBasis: "fb5389c9b11b9f559858362a311278922e6f8a95ca6864c0f21a2c01dbcc3183",
    melWindow: "138de18ad784bd866e7ebe9807f1713b69cebe99ebf43dd85dd85abab294eb88",
  }),
} as const);

export const CWFORMER_LIMITS = Object.freeze({
  chunkMs: 500,
  maximumCacheSeconds: 30,
  maximumFileSeconds: 30 * 60,
  maximumDecodedTokens: 100_000,
} as const);

export const CWFORMER_VOCABULARY = Object.freeze([
  "<blank>",
  " ",
  ...Array.from({ length: 26 }, (_, index) => String.fromCharCode(65 + index)),
  ...Array.from({ length: 10 }, (_, index) => String(index)),
  ...".,?/(&=+",
  "AR",
  "SK",
  "BT",
  "KN",
  "AS",
  "CT",
]);

const BLANK_INDEX = 0;
const LOG_MEL_EPSILON = 1e-6;

export interface CwformerConfig {
  sample_rate: number;
  n_fft: number;
  hop_length: number;
  n_mels: number;
  f_min: number;
  f_max: number;
  architecture?: string;
  n_layers?: number;
  n_heads?: number;
  d_model?: number;
  d_k?: number;
  conv_kernel?: number;
  max_cache_len?: number;
  subsample_channels?: number;
}

export interface CwformerFrontendAssets {
  config: CwformerConfig;
  melBasis: Float32Array;
  melWindow: Float32Array;
}

export type CwformerTensorData = Float32Array | BigInt64Array;

export interface CwformerOutputTensor {
  data: CwformerTensorData;
  dims: readonly number[];
}

export interface CwformerSession {
  run(feeds: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, CwformerOutputTensor>>>;
  release?(): Promise<void> | void;
}

export type CwformerTensorFactory = (
  type: "float32" | "int64",
  data: CwformerTensorData,
  dims: readonly number[],
) => unknown;

export interface CwformerToken {
  value: string;
  classIndex: number;
  frameIndex: number;
  timestampMs: number;
  confidence: number;
  logMargin: number;
}

export interface CwformerChunkResult {
  newText: string;
  fullText: string;
  tokens: CwformerToken[];
}

export interface CwformerDecodeOptions {
  signal?: AbortSignal;
}

export interface ParsedNpyFloat32 {
  data: Float32Array;
  shape: number[];
}

export interface CwformerLaneConditioningOptions {
  /** Center frequency presented to the neural model after translation. */
  targetHz?: number;
  /** One-sided baseband width retained around the classical decoder's lane. */
  basebandHz?: number;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function concatenate(left: Float32Array, right: Float32Array) {
  if (!left.length) return right.slice();
  if (!right.length) return left.slice();
  const joined = new Float32Array(left.length + right.length);
  joined.set(left);
  joined.set(right, left.length);
  return joined;
}

function abortError() {
  return new DOMException("CWformer decode cancelled.", "AbortError");
}

function validateAssets(assets: CwformerFrontendAssets) {
  const { config } = assets;
  const numeric = [
    config.sample_rate,
    config.n_fft,
    config.hop_length,
    config.n_mels,
    config.f_min,
    config.f_max,
  ];
  if (!numeric.every(Number.isFinite) || config.sample_rate <= 0 || config.n_fft <= 0 ||
    config.hop_length <= 0 || config.n_mels <= 0 || config.n_fft % 2 !== 0) {
    throw new Error("CWformer mel_config.json is invalid.");
  }
  const frequencyBins = config.n_fft / 2 + 1;
  if (assets.melWindow.length !== config.n_fft ||
    assets.melBasis.length !== config.n_mels * frequencyBins) {
    throw new Error("CWformer mel tables do not match mel_config.json.");
  }
}

interface BiquadState {
  z1: number;
  z2: number;
}

function lowpassCoefficients(sampleRate: number, cutoffHz: number, q: number) {
  const omega = 2 * Math.PI * cutoffHz / sampleRate;
  const cosine = Math.cos(omega);
  const alpha = Math.sin(omega) / (2 * q);
  const scale = 1 / (1 + alpha);
  return {
    b0: (1 - cosine) * 0.5 * scale,
    b1: (1 - cosine) * scale,
    b2: (1 - cosine) * 0.5 * scale,
    a1: -2 * cosine * scale,
    a2: (1 - alpha) * scale,
  };
}

function filterBiquad(
  input: number,
  state: BiquadState,
  coefficients: ReturnType<typeof lowpassCoefficients>,
) {
  const output = coefficients.b0 * input + state.z1;
  state.z1 = coefficients.b1 * input - coefficients.a1 * output + state.z2;
  state.z2 = coefficients.b2 * input - coefficients.a2 * output;
  return output;
}

/**
 * Isolates one frequency lane and translates it to the model's nominal 700 Hz.
 *
 * This is deliberately driven by a carrier already found by the physical
 * decoder. It is not a station detector and must not be used to merge text
 * across lanes. A fourth-order Butterworth low-pass is applied to complex I/Q
 * baseband before remodulation, avoiding the image produced by real mixing.
 */
export function conditionCwformerLane(
  audio: Float32Array,
  sampleRate: number,
  carrierHz: number,
  options: CwformerLaneConditioningOptions = {},
) {
  const targetHz = options.targetHz ?? 700;
  const basebandHz = options.basebandHz ?? 180;
  const nyquist = sampleRate / 2;
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 ||
    !Number.isFinite(carrierHz) || carrierHz <= 0 || carrierHz >= nyquist ||
    !Number.isFinite(targetHz) || targetHz <= 0 || targetHz >= nyquist ||
    !Number.isFinite(basebandHz) || basebandHz <= 0 ||
    basebandHz >= Math.min(carrierHz, nyquist - carrierHz, targetHz, nyquist - targetHz)) {
    throw new Error("CWformer lane conditioning frequencies are invalid.");
  }

  // Fourth-order Butterworth response: the two Q values are the conjugate
  // pole pairs for a normalized fourth-order low-pass.
  const stages = [
    lowpassCoefficients(sampleRate, basebandHz, 0.541196100146197),
    lowpassCoefficients(sampleRate, basebandHz, 1.306562964876377),
  ];
  const iState = stages.map(() => ({ z1: 0, z2: 0 }));
  const qState = stages.map(() => ({ z1: 0, z2: 0 }));
  const carrierStepCos = Math.cos(2 * Math.PI * carrierHz / sampleRate);
  const carrierStepSin = Math.sin(2 * Math.PI * carrierHz / sampleRate);
  const targetStepCos = Math.cos(2 * Math.PI * targetHz / sampleRate);
  const targetStepSin = Math.sin(2 * Math.PI * targetHz / sampleRate);
  let carrierCos = 1;
  let carrierSin = 0;
  let targetCos = 1;
  let targetSin = 0;
  const output = new Float32Array(audio.length);

  for (let index = 0; index < audio.length; index += 1) {
    const sample = Number.isFinite(audio[index]) ? audio[index] : 0;
    let inPhase = sample * carrierCos;
    let quadrature = -sample * carrierSin;
    for (let stage = 0; stage < stages.length; stage += 1) {
      inPhase = filterBiquad(inPhase, iState[stage], stages[stage]);
      quadrature = filterBiquad(quadrature, qState[stage], stages[stage]);
    }
    output[index] = 2 * (inPhase * targetCos - quadrature * targetSin);

    const nextCarrierCos = carrierCos * carrierStepCos - carrierSin * carrierStepSin;
    carrierSin = carrierSin * carrierStepCos + carrierCos * carrierStepSin;
    carrierCos = nextCarrierCos;
    const nextTargetCos = targetCos * targetStepCos - targetSin * targetStepSin;
    targetSin = targetSin * targetStepCos + targetCos * targetStepSin;
    targetCos = nextTargetCos;

    // Keep recursive oscillators on the unit circle during long recordings.
    if ((index & 4095) === 4095) {
      const carrierNorm = Math.hypot(carrierCos, carrierSin) || 1;
      carrierCos /= carrierNorm;
      carrierSin /= carrierNorm;
      const targetNorm = Math.hypot(targetCos, targetSin) || 1;
      targetCos /= targetNorm;
      targetSin /= targetNorm;
    }
  }
  return output;
}

/**
 * Exact-length two-factor Cooley-Tukey transform. The release's 400-point
 * window factors into 20 x 20, avoiding both an O(N²) transform and the
 * padding overhead of Bluestein. Prime lengths safely degenerate to a direct
 * transform, although production validates against the pinned 400-point table.
 */
class MixedRadixRealFft {
  private readonly firstFactor: number;
  private readonly secondFactor: number;
  private readonly firstCos: Float64Array;
  private readonly firstSin: Float64Array;
  private readonly secondCos: Float64Array;
  private readonly secondSin: Float64Array;
  private readonly crossCos: Float64Array;
  private readonly crossSin: Float64Array;
  private readonly stageReal: Float64Array;
  private readonly stageImaginary: Float64Array;

  constructor(private readonly length: number) {
    let divisor = Math.floor(Math.sqrt(length));
    while (divisor > 1 && length % divisor) divisor -= 1;
    this.firstFactor = divisor;
    this.secondFactor = length / divisor;
    this.firstCos = new Float64Array(this.secondFactor * this.secondFactor);
    this.firstSin = new Float64Array(this.secondFactor * this.secondFactor);
    this.secondCos = new Float64Array(this.firstFactor * this.firstFactor);
    this.secondSin = new Float64Array(this.firstFactor * this.firstFactor);
    this.crossCos = new Float64Array(length);
    this.crossSin = new Float64Array(length);
    this.stageReal = new Float64Array(length);
    this.stageImaginary = new Float64Array(length);
    for (let frequency = 0; frequency < this.secondFactor; frequency += 1) {
      for (let sample = 0; sample < this.secondFactor; sample += 1) {
        const index = frequency * this.secondFactor + sample;
        const angle = -2 * Math.PI * frequency * sample / this.secondFactor;
        this.firstCos[index] = Math.cos(angle);
        this.firstSin[index] = Math.sin(angle);
      }
    }
    for (let frequency = 0; frequency < this.firstFactor; frequency += 1) {
      for (let sample = 0; sample < this.firstFactor; sample += 1) {
        const index = frequency * this.firstFactor + sample;
        const angle = -2 * Math.PI * frequency * sample / this.firstFactor;
        this.secondCos[index] = Math.cos(angle);
        this.secondSin[index] = Math.sin(angle);
      }
    }
    for (let first = 0; first < this.firstFactor; first += 1) {
      for (let secondFrequency = 0; secondFrequency < this.secondFactor; secondFrequency += 1) {
        const index = first * this.secondFactor + secondFrequency;
        const angle = -2 * Math.PI * first * secondFrequency / length;
        this.crossCos[index] = Math.cos(angle);
        this.crossSin[index] = Math.sin(angle);
      }
    }
  }

  power(input: Float32Array, output: Float64Array) {
    // First transform: each decimated sequence x[n1 + N1*n2].
    for (let first = 0; first < this.firstFactor; first += 1) {
      for (let secondFrequency = 0; secondFrequency < this.secondFactor; secondFrequency += 1) {
        let real = 0;
        let imaginary = 0;
        const twiddleStart = secondFrequency * this.secondFactor;
        for (let second = 0; second < this.secondFactor; second += 1) {
          const sample = input[first + this.firstFactor * second];
          real += sample * this.firstCos[twiddleStart + second];
          imaginary += sample * this.firstSin[twiddleStart + second];
        }
        const stageIndex = first * this.secondFactor + secondFrequency;
        const crossCos = this.crossCos[stageIndex];
        const crossSin = this.crossSin[stageIndex];
        this.stageReal[stageIndex] = real * crossCos - imaginary * crossSin;
        this.stageImaginary[stageIndex] = real * crossSin + imaginary * crossCos;
      }
    }

    // Second transform; output index is k2 + N2*k1.
    for (let bin = 0; bin < output.length; bin += 1) {
      const firstFrequency = Math.floor(bin / this.secondFactor);
      const secondFrequency = bin % this.secondFactor;
      const twiddleStart = firstFrequency * this.firstFactor;
      let real = 0;
      let imaginary = 0;
      for (let first = 0; first < this.firstFactor; first += 1) {
        const stageIndex = first * this.secondFactor + secondFrequency;
        const stageReal = this.stageReal[stageIndex];
        const stageImaginary = this.stageImaginary[stageIndex];
        const cosine = this.secondCos[twiddleStart + first];
        const sine = this.secondSin[twiddleStart + first];
        real += stageReal * cosine - stageImaginary * sine;
        imaginary += stageReal * sine + stageImaginary * cosine;
      }
      output[bin] = real * real + imaginary * imaginary;
    }
  }
}

/** Parses the exact little-endian, C-order float32 NPY tables in the release. */
export function parseNpyFloat32(buffer: ArrayBuffer): ParsedNpyFloat32 {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12 || bytes[0] !== 0x93 ||
    String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error("Invalid NPY file.");
  }
  const major = bytes[6];
  const headerOffset = major === 1 ? 10 : 12;
  if (major < 1 || major > 3 || bytes.length < headerOffset) {
    throw new Error(`Unsupported NPY version ${major}.${bytes[7]}.`);
  }
  const headerLength = major === 1
    ? new DataView(buffer).getUint16(8, true)
    : new DataView(buffer).getUint32(8, true);
  const dataOffset = headerOffset + headerLength;
  if (dataOffset > bytes.length) throw new Error("Truncated NPY header.");
  const header = new TextDecoder(major === 3 ? "utf-8" : "latin1")
    .decode(bytes.slice(headerOffset, dataOffset));
  if (!/["']descr["']\s*:\s*["']<f4["']/.test(header) ||
    !/["']fortran_order["']\s*:\s*False/.test(header)) {
    throw new Error("CWformer requires little-endian C-order float32 NPY tables.");
  }
  const shapeMatch = header.match(/["']shape["']\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error("NPY shape is missing.");
  const shape = shapeMatch[1].split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  if (!shape.length || shape.some((dimension) => !Number.isInteger(dimension) || dimension < 0)) {
    throw new Error("NPY shape is invalid.");
  }
  const count = shape.reduce((product, dimension) => product * dimension, 1);
  if (dataOffset + count * 4 > bytes.length) throw new Error("Truncated NPY float data.");
  const view = new DataView(buffer, dataOffset, count * 4);
  const data = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    data[index] = view.getFloat32(index * 4, true);
  }
  return { data, shape };
}

export class CwformerMelFrontend {
  private readonly config: CwformerConfig;
  private readonly basis: Float32Array;
  private readonly window: Float32Array;
  private readonly fft: MixedRadixRealFft;
  private stftBuffer: Float32Array | null = null;

  constructor(assets: CwformerFrontendAssets) {
    validateAssets(assets);
    this.config = assets.config;
    this.basis = assets.melBasis;
    this.window = assets.melWindow;
    this.fft = new MixedRadixRealFft(this.config.n_fft);
  }

  reset() {
    this.stftBuffer = null;
  }

  computeStreaming(audioChunk: Float32Array) {
    let audio: Float32Array;
    if (this.stftBuffer) {
      audio = concatenate(this.stftBuffer, audioChunk);
    } else {
      audio = new Float32Array(this.config.n_fft / 2 + audioChunk.length);
      audio.set(audioChunk, this.config.n_fft / 2);
    }

    const frameCount = audio.length >= this.config.n_fft
      ? Math.floor((audio.length - this.config.n_fft) / this.config.hop_length) + 1
      : 0;
    const consumed = frameCount * this.config.hop_length;
    this.stftBuffer = audio.slice(consumed);
    if (!frameCount) {
      return { data: new Float32Array(), frameCount: 0 };
    }

    const bins = this.config.n_fft / 2 + 1;
    const output = new Float32Array(frameCount * this.config.n_mels);
    const windowed = new Float32Array(this.config.n_fft);
    const power = new Float64Array(bins);
    for (let frame = 0; frame < frameCount; frame += 1) {
      const frameStart = frame * this.config.hop_length;
      for (let sample = 0; sample < this.config.n_fft; sample += 1) {
        const value = Number.isFinite(audio[frameStart + sample]) ? audio[frameStart + sample] : 0;
        windowed[sample] = value * this.window[sample];
      }
      this.fft.power(windowed, power);
      for (let mel = 0; mel < this.config.n_mels; mel += 1) {
        let energy = 0;
        const basisStart = mel * bins;
        for (let bin = 0; bin < bins; bin += 1) {
          energy += power[bin] * this.basis[basisStart + bin];
        }
        output[frame * this.config.n_mels + mel] = Math.log(energy + LOG_MEL_EPSILON);
      }
    }
    return { data: output, frameCount };
  }
}

function requireOutput(
  outputs: Readonly<Record<string, CwformerOutputTensor>>,
  name: string,
) {
  const output = outputs[name];
  if (!output?.dims || !output.data) throw new Error(`CWformer model output ${name} is missing.`);
  return output;
}

function floatOutput(tensor: CwformerOutputTensor, name: string) {
  if (!(tensor.data instanceof Float32Array)) {
    throw new Error(`CWformer model output ${name} is not float32.`);
  }
  return tensor.data;
}

export class CwformerStreamingAdapter {
  readonly sampleRate: number;
  readonly chunkMs: number;

  private readonly session: CwformerSession;
  private readonly tensor: CwformerTensorFactory;
  private readonly config: Required<Pick<CwformerConfig,
    "n_layers" | "n_heads" | "d_model" | "d_k" | "conv_kernel" | "subsample_channels">> &
    CwformerConfig;
  private readonly frontend: CwformerMelFrontend;
  private readonly chunkSamples: number;
  private readonly maximumCacheFrames: number;
  private readonly ctcFrameMs: number;
  private state: Record<string, unknown> = {};
  private audioBuffer = new Float32Array();
  private decodedTokens: CwformerToken[] = [];
  private previousClass = -1;
  private outputFrame = 0;
  private running = false;
  private disposed = false;

  constructor(
    session: CwformerSession,
    tensorFactory: CwformerTensorFactory,
    assets: CwformerFrontendAssets,
    chunkMs = CWFORMER_LIMITS.chunkMs,
  ) {
    validateAssets(assets);
    if (!Number.isFinite(chunkMs) || chunkMs <= 0) throw new Error("CWformer chunk size must be positive.");
    this.session = session;
    this.tensor = tensorFactory;
    this.config = {
      ...assets.config,
      n_layers: assets.config.n_layers ?? 12,
      n_heads: assets.config.n_heads ?? 4,
      d_model: assets.config.d_model ?? 256,
      d_k: assets.config.d_k ?? 64,
      conv_kernel: assets.config.conv_kernel ?? 63,
      subsample_channels: assets.config.subsample_channels ?? 256,
    };
    this.frontend = new CwformerMelFrontend(assets);
    this.sampleRate = this.config.sample_rate;
    this.chunkMs = chunkMs;
    this.chunkSamples = Math.max(1, Math.round(chunkMs * this.sampleRate / 1_000));
    this.maximumCacheFrames = this.config.max_cache_len ??
      Math.round(CWFORMER_LIMITS.maximumCacheSeconds * 50);
    this.ctcFrameMs = this.config.hop_length / this.sampleRate * 2_000;
    this.reset();
  }

  reset() {
    if (this.running) throw new Error("Cannot reset CWformer during inference.");
    if (this.disposed) throw new Error("CWformer session has been released.");
    this.frontend.reset();
    this.audioBuffer = new Float32Array();
    this.decodedTokens = [];
    this.previousClass = -1;
    this.outputFrame = 0;
    this.state = this.initialState();
  }

  getFullText() {
    return this.decodedTokens.map((token) => token.value).join("").trim();
  }

  getTokens() {
    return this.decodedTokens.map((token) => ({ ...token }));
  }

  async feedAudio16k(audioChunk: Float32Array): Promise<CwformerChunkResult> {
    this.assertReady();
    if (this.running) throw new Error("CWformer audio feeds must be awaited sequentially.");
    this.running = true;
    try {
      this.audioBuffer = concatenate(this.audioBuffer, audioChunk);
      const emitted: CwformerToken[] = [];
      while (this.audioBuffer.length >= this.chunkSamples) {
        const chunk = this.audioBuffer.slice(0, this.chunkSamples);
        this.audioBuffer = this.audioBuffer.slice(this.chunkSamples);
        emitted.push(...await this.processChunk(chunk));
      }
      return this.chunkResult(emitted);
    } finally {
      this.running = false;
    }
  }

  async flush(): Promise<CwformerChunkResult> {
    this.assertReady();
    if (this.running) throw new Error("CWformer audio feeds must be awaited sequentially.");
    this.running = true;
    try {
      const emitted: CwformerToken[] = [];
      // This intentionally matches the v0.2.0 reference: right padding is
      // added only when a partial audio chunk remains.
      if (this.audioBuffer.length) {
        const tail = new Float32Array(this.audioBuffer.length + this.config.n_fft / 2);
        tail.set(this.audioBuffer);
        this.audioBuffer = new Float32Array();
        emitted.push(...await this.processChunk(tail));
      }
      return this.chunkResult(emitted);
    } finally {
      this.running = false;
    }
  }

  async decodeAudio16k(
    audio: Float32Array,
    options: CwformerDecodeOptions = {},
  ): Promise<CwformerChunkResult> {
    this.assertReady();
    if (audio.length > this.sampleRate * CWFORMER_LIMITS.maximumFileSeconds) {
      throw new Error("CWformer file exceeds the bounded 30-minute inference limit.");
    }
    this.reset();
    let peak = 0;
    for (const value of audio) {
      if (Number.isFinite(value)) peak = Math.max(peak, Math.abs(value));
    }
    const scale = peak < 1e-8 ? 1 : 0.7 / peak;
    for (let offset = 0; offset < audio.length; offset += this.chunkSamples) {
      if (options.signal?.aborted) throw abortError();
      const end = Math.min(audio.length, offset + this.chunkSamples);
      const chunk = new Float32Array(end - offset);
      for (let index = offset; index < end; index += 1) {
        chunk[index - offset] = (Number.isFinite(audio[index]) ? audio[index] : 0) * scale;
      }
      await this.feedAudio16k(chunk);
    }
    if (options.signal?.aborted) throw abortError();
    await this.flush();
    return this.chunkResult(this.getTokens());
  }

  async release() {
    if (this.running) throw new Error("Cannot release CWformer during inference.");
    if (this.disposed) return;
    this.disposed = true;
    this.state = {};
    this.audioBuffer = new Float32Array();
    this.decodedTokens = [];
    await this.session.release?.();
  }

  private assertReady() {
    if (this.disposed) throw new Error("CWformer session has been released.");
  }

  private initialState() {
    const state: Record<string, unknown> = {
      pos_offset: this.tensor("int64", new BigInt64Array([0n]), [1]),
    };
    const convolutionPadding = this.config.conv_kernel - 1;
    for (let layer = 0; layer < this.config.n_layers; layer += 1) {
      state[`kv_k_layer${layer}`] = this.tensor(
        "float32",
        new Float32Array(),
        [1, this.config.n_heads, 0, this.config.d_k],
      );
      state[`kv_v_layer${layer}`] = this.tensor(
        "float32",
        new Float32Array(),
        [1, this.config.n_heads, 0, this.config.d_k],
      );
      state[`conv_buf_layer${layer}`] = this.tensor(
        "float32",
        new Float32Array(this.config.d_model * convolutionPadding),
        [1, this.config.d_model, convolutionPadding],
      );
    }
    state.sub_buf1 = this.tensor(
      "float32",
      new Float32Array(2 * this.config.n_mels),
      [1, 1, 2, this.config.n_mels],
    );
    const frequencyAfterFirstSubsample = Math.ceil(this.config.n_mels / 2);
    state.sub_buf2 = this.tensor(
      "float32",
      new Float32Array(this.config.subsample_channels * 2 * frequencyAfterFirstSubsample),
      [1, this.config.subsample_channels, 2, frequencyAfterFirstSubsample],
    );
    return state;
  }

  private async processChunk(audioChunk: Float32Array) {
    const mel = this.frontend.computeStreaming(audioChunk);
    if (!mel.frameCount) return [];
    const feeds: Record<string, unknown> = {
      mel_chunk: this.tensor("float32", mel.data, [1, mel.frameCount, this.config.n_mels]),
      ...this.state,
    };
    const outputs = await this.session.run(feeds);
    const logProbs = requireOutput(outputs, "log_probs");
    this.state.pos_offset = requireOutput(outputs, "pos_offset_out");
    for (let layer = 0; layer < this.config.n_layers; layer += 1) {
      this.state[`kv_k_layer${layer}`] = this.boundedKvCache(
        requireOutput(outputs, `kv_k_layer${layer}_out`),
        `kv_k_layer${layer}_out`,
      );
      this.state[`kv_v_layer${layer}`] = this.boundedKvCache(
        requireOutput(outputs, `kv_v_layer${layer}_out`),
        `kv_v_layer${layer}_out`,
      );
      this.state[`conv_buf_layer${layer}`] = requireOutput(
        outputs,
        `conv_buf_layer${layer}_out`,
      );
    }
    this.state.sub_buf1 = requireOutput(outputs, "sub_buf1_out");
    this.state.sub_buf2 = requireOutput(outputs, "sub_buf2_out");
    return this.consumeLogProbs(logProbs);
  }

  private boundedKvCache(tensor: CwformerOutputTensor, name: string) {
    const data = floatOutput(tensor, name);
    const dims = tensor.dims.map(Number);
    if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== this.config.n_heads ||
      dims[3] !== this.config.d_k || data.length !== dims.reduce((a, b) => a * b, 1)) {
      throw new Error(`CWformer model output ${name} has an invalid shape.`);
    }
    const frames = dims[2];
    if (frames <= this.maximumCacheFrames) return tensor;
    const output = new Float32Array(this.config.n_heads * this.maximumCacheFrames * this.config.d_k);
    for (let head = 0; head < this.config.n_heads; head += 1) {
      const sourceStart = (head * frames + frames - this.maximumCacheFrames) * this.config.d_k;
      const sourceEnd = sourceStart + this.maximumCacheFrames * this.config.d_k;
      output.set(
        data.subarray(sourceStart, sourceEnd),
        head * this.maximumCacheFrames * this.config.d_k,
      );
    }
    return this.tensor(
      "float32",
      output,
      [1, this.config.n_heads, this.maximumCacheFrames, this.config.d_k],
    );
  }

  private consumeLogProbs(tensor: CwformerOutputTensor) {
    const data = floatOutput(tensor, "log_probs");
    const dims = tensor.dims.map(Number);
    if (dims.length !== 3 || dims[1] !== 1 || dims[2] !== CWFORMER_VOCABULARY.length ||
      data.length !== dims[0] * dims[1] * dims[2]) {
      throw new Error("CWformer log_probs has an invalid shape.");
    }
    const emitted: CwformerToken[] = [];
    for (let frame = 0; frame < dims[0]; frame += 1) {
      const start = frame * CWFORMER_VOCABULARY.length;
      let winner = 0;
      let best = -Infinity;
      let second = -Infinity;
      for (let classIndex = 0; classIndex < CWFORMER_VOCABULARY.length; classIndex += 1) {
        const value = data[start + classIndex];
        if (value > best) {
          second = best;
          best = value;
          winner = classIndex;
        } else if (value > second) second = value;
      }
      if (winner !== this.previousClass && winner !== BLANK_INDEX) {
        const token: CwformerToken = {
          value: CWFORMER_VOCABULARY[winner],
          classIndex: winner,
          frameIndex: this.outputFrame,
          timestampMs: this.outputFrame * this.ctcFrameMs,
          confidence: clamp(Math.exp(best), 0, 1),
          logMargin: best - second,
        };
        this.decodedTokens.push(token);
        emitted.push(token);
        if (this.decodedTokens.length > CWFORMER_LIMITS.maximumDecodedTokens) {
          throw new Error("CWformer output exceeded its bounded token limit.");
        }
      }
      this.previousClass = winner;
      this.outputFrame += 1;
    }
    return emitted;
  }

  private chunkResult(tokens: CwformerToken[]): CwformerChunkResult {
    return {
      newText: tokens.map((token) => token.value).join(""),
      fullText: this.getFullText(),
      tokens: tokens.map((token) => ({ ...token })),
    };
  }
}
