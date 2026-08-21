import type { OfflineDecodeCandidate, OfflineDecodeResult } from "./optimizerCore";
import type { CwformerToken } from "./cwformer";

export const NEURAL_ADVISORY_MAX_LANES = 2;
export const NEURAL_ADVISORY_MAX_FILE_SECONDS = 3 * 60;
export const NEURAL_ADVISORY_MAX_TOTAL_AUDIO_SECONDS = 3 * 60;

export interface NeuralDecodeAdvisory {
  engine: "CWformer";
  engineVersion: "0.2.0";
  runtime: "ONNX Runtime Web 1.27.0 · WASM/1T";
  authoritative: false;
  transcript: string;
  tokens: CwformerToken[];
  channel: number;
  sourceFrequency: number;
  conditionedFrequency: 700;
  sourceSampleRate: number;
  modelSampleRate: 16_000;
  durationSeconds: number;
  meanTokenConfidence: number;
  inferenceMs: number;
}

export interface NeuralWorkerRequest {
  type: "decode";
  lanes: OfflineDecodeCandidate[];
  channels: ArrayBuffer[];
  sampleRate: number;
}

export type NeuralWorkerResponse =
  | { type: "progress"; fraction: number; detail: string }
  | { type: "result"; advisories: NeuralDecodeAdvisory[] }
  | { type: "error"; message: string };

/**
 * Selects only physically credible, distinct station tracks discovered by the
 * classical solver. Long recordings receive fewer advisory lanes so a bad
 * pile-up cannot turn one file analysis into unbounded neural work.
 */
export function selectNeuralAdvisoryLanes(
  result: OfflineDecodeResult,
  channels: readonly Float32Array[],
  sampleRate: number,
) {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !channels.length) return [];
  const durationSeconds = Math.max(...channels.map((channel) => channel.length), 0) / sampleRate;
  if (durationSeconds <= 0 || durationSeconds > NEURAL_ADVISORY_MAX_FILE_SECONDS) return [];
  const maximumLanes = Math.min(
    NEURAL_ADVISORY_MAX_LANES,
    Math.floor(NEURAL_ADVISORY_MAX_TOTAL_AUDIO_SECONDS / durationSeconds),
  );
  if (maximumLanes < 1) return [];

  return result.candidates
    .filter((candidate) => isCredibleLane(candidate, channels, sampleRate))
    .sort((left, right) => Number(isSelectedLane(right, result)) - Number(isSelectedLane(left, result)) ||
      right.decoderScore - left.decoderScore || right.score - left.score)
    .reduce<OfflineDecodeCandidate[]>((lanes, candidate) => {
      if (lanes.length >= maximumLanes || lanes.some((lane) =>
        Math.abs(lane.frequency - candidate.frequency) < 55)) return lanes;
      lanes.push(candidate);
      return lanes;
    }, []);
}

function isSelectedLane(candidate: OfflineDecodeCandidate, result: OfflineDecodeResult) {
  return candidate.channel === result.channel && Math.abs(candidate.frequency - result.frequency) < 5;
}

function isCredibleLane(
  candidate: OfflineDecodeCandidate,
  channels: readonly Float32Array[],
  sampleRate: number,
) {
  const channel = channels[candidate.channel];
  const copyLength = candidate.transcript.replace(/\s/g, "").length;
  return Boolean(channel?.length) &&
    Number.isFinite(candidate.frequency) &&
    candidate.frequency >= 180 &&
    candidate.frequency < sampleRate * 0.45 &&
    candidate.contrastDb >= 5 &&
    candidate.validRatio >= 0.72 &&
    candidate.timingQuality >= 0.4 &&
    candidate.decoderScore >= 0.25 &&
    copyLength >= 2;
}

/**
 * The optimizer supplies at most 12 kHz audio, while CWformer is fixed at
 * 16 kHz. Since this is an upsample and the subsequent lane filter retains
 * only a 360 Hz-wide band, interpolation is comfortably above its passband.
 */
export function resampleForCwformer(
  audio: Float32Array,
  sourceSampleRate: number,
  targetSampleRate = 16_000,
) {
  if (!audio.length) return new Float32Array();
  if (!Number.isFinite(sourceSampleRate) || sourceSampleRate <= 0 ||
    !Number.isFinite(targetSampleRate) || targetSampleRate <= 0) {
    throw new Error("Neural lane sample rate is invalid.");
  }
  if (sourceSampleRate === targetSampleRate) return audio.slice();
  const outputLength = Math.max(1, Math.round(audio.length * targetSampleRate / sourceSampleRate));
  const output = new Float32Array(outputLength);
  const step = sourceSampleRate / targetSampleRate;
  for (let index = 0; index < output.length; index += 1) {
    const sourcePosition = Math.min(audio.length - 1, index * step);
    const left = Math.floor(sourcePosition);
    const right = Math.min(audio.length - 1, left + 1);
    const fraction = sourcePosition - left;
    const leftValue = Number.isFinite(audio[left]) ? audio[left] : 0;
    const rightValue = Number.isFinite(audio[right]) ? audio[right] : 0;
    output[index] = leftValue + (rightValue - leftValue) * fraction;
  }
  return output;
}
