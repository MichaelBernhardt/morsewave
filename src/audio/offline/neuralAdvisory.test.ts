import { describe, expect, it } from "vitest";
import type { OfflineDecodeCandidate, OfflineDecodeResult } from "./optimizerCore";
import { resampleForCwformer, selectNeuralAdvisoryLanes } from "./neuralAdvisory";

const candidate = (
  frequency: number,
  channel = 0,
  overrides: Partial<OfflineDecodeCandidate> = {},
): OfflineDecodeCandidate => ({
  frequency,
  channel,
  score: 0.8,
  wpm: 18,
  contrastDb: 14,
  validRatio: 0.96,
  timingQuality: 0.8,
  languageQuality: 0.5,
  messageQuality: 0.4,
  hamEvidence: 0,
  decoderScore: 0.75,
  transcript: "CQ TEST",
  ...overrides,
});

const resultWith = (candidates: OfflineDecodeCandidate[]): OfflineDecodeResult => ({
  transcript: "CQ TEST",
  rawTranscript: "CQ TEST",
  tokens: [],
  frequency: 700,
  channel: 0,
  wpm: 18,
  thresholdDb: -40,
  score: 0.8,
  hamEvidence: 0,
  confidence: 0.8,
  converged: true,
  passCount: 100,
  candidates,
  variants: [],
});

describe("neural advisory lane bounds", () => {
  it("prioritizes the selected lane and keeps distinct credible carriers", () => {
    const channels = [new Float32Array(120_000), new Float32Array(120_000)];
    const lanes = selectNeuralAdvisoryLanes(resultWith([
      candidate(1_400, 1, { decoderScore: 0.9 }),
      candidate(701, 0),
      candidate(730, 1),
      candidate(2_100, 1, { timingQuality: 0.2 }),
    ]), channels, 12_000);
    expect(lanes.map((lane) => Math.round(lane.frequency))).toEqual([701, 1_400]);
  });

  it("reduces work for longer files and rejects files above the desktop inference bound", () => {
    const candidates = [200, 260, 320, 380].map((frequency) => candidate(frequency));
    const twoMinutes = [new Float32Array(2 * 60 * 1_000)];
    expect(selectNeuralAdvisoryLanes(resultWith(candidates), twoMinutes, 1_000)).toHaveLength(1);
    const overThreeMinutes = [new Float32Array(4 * 60 * 1_000)];
    expect(selectNeuralAdvisoryLanes(resultWith(candidates), overThreeMinutes, 1_000)).toEqual([]);
  });
});

describe("CWformer resampling", () => {
  it("preserves duration and the phase of a lane tone during 12 to 16 kHz conversion", () => {
    const sourceRate = 12_000;
    const source = Float32Array.from({ length: sourceRate }, (_, index) =>
      Math.sin(2 * Math.PI * 700 * index / sourceRate));
    const output = resampleForCwformer(source, sourceRate);
    expect(output).toHaveLength(16_000);
    expect(output[4_000]).toBeCloseTo(0, 3);
    expect(output[4_004]).toBeCloseTo(Math.sin(2 * Math.PI * 700 * 4_004 / 16_000), 2);
  });

  it("copies exact-rate input and fails closed on invalid rates", () => {
    const input = new Float32Array([0, 1, Number.NaN]);
    expect([...resampleForCwformer(input, 16_000)]).toEqual([0, 1, Number.NaN]);
    expect(() => resampleForCwformer(input, 0)).toThrow(/sample rate/);
  });
});
