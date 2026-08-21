import { annotateMorseValue, decodeMorse, isMorseCharacter } from "../morse";
import { HAM_CW_TERMS, isHamCwScoringTerm } from "../hamVocabulary";
import { hamMessageEvidence } from "../hamInterpreter";
import {
  analyzeQsbEnvelope,
  buildQsbHmmRuns,
  qsbHmmEligibility,
  selectQsbHmmAlternate,
} from "./qsbHmm";

export interface OfflineDecodeOptions {
  wpmHint: number;
  languageAssist: boolean;
  minimumFrequency?: number;
  maximumFrequency?: number;
}

export interface OfflineDecodeProgress {
  phase: "survey" | "candidates" | "refine" | "complete";
  fraction: number;
  detail: string;
}

export interface OfflineDecodeToken {
  value: string;
  rawValue?: string;
  pattern: string;
  confidence: number;
  timestamp: number;
  space?: boolean;
}

export interface OfflineDecodeCandidate {
  /** Actual spectral line used for this copy. */
  frequency: number;
  /** Lowest credible coherent family member when this line is a keyed harmonic. */
  fundamentalHz?: number;
  channel: number;
  score: number;
  wpm: number;
  contrastDb: number;
  validRatio: number;
  timingQuality: number;
  languageQuality: number;
  messageQuality: number;
  hamEvidence: number;
  decoderScore: number;
  /** Literal decoder copy used for hypothesis scoring, before display aliases. */
  transcript: string;
  thresholdDb?: number;
}

export interface OfflineDecodeResult {
  /** User-facing copy, including enabled prosign notation. */
  transcript: string;
  /** Literal International Morse lookup used for scoring and convergence. */
  rawTranscript: string;
  tokens: OfflineDecodeToken[];
  frequency: number;
  /** Lowest credible coherent harmonic-family member; `frequency` remains the actual decode lane. */
  fundamentalHz?: number;
  channel: number;
  wpm: number;
  thresholdDb: number;
  score: number;
  /** Sequence-aware CW shorthand contribution used only for bounded ranking. */
  hamEvidence: number;
  confidence: number;
  converged: boolean;
  passCount: number;
  candidates: OfflineDecodeCandidate[];
  variants: OfflineDecodeCandidate[];
}

export interface OfflineWorkerRequest {
  type: "optimize";
  channels: ArrayBuffer[];
  sampleRate: number;
  options: OfflineDecodeOptions;
}

export type OfflineWorkerResponse =
  | { type: "progress"; progress: OfflineDecodeProgress }
  | { type: "result"; result: OfflineDecodeResult }
  | { type: "error"; message: string };

interface SpectralCandidate {
  frequency: number;
  channel: number;
  discoveryScore: number;
  previewStartSeconds?: number;
}

interface Envelope {
  levelsDb: Float32Array;
  hopMs: number;
  offsetMs: number;
  lowDb: number;
  highDb: number;
}

interface GateRun {
  keyed: boolean;
  startMs: number;
  endMs: number;
}

interface DecodeHypothesis {
  transcript: string;
  tokens: OfflineDecodeToken[];
  score: number;
  thresholdDb: number;
  thresholdFraction: number;
  hysteresisDb: number;
  debounceMs: number;
  characterUnits: number;
  wordUnits: number;
  unitMs: number;
  wpm: number;
  timingQuality: number;
  validRatio: number;
  languageQuality: number;
  messageQuality: number;
  hamEvidence: number;
  baseScore: number;
  decoderScore: number;
  contrastDb: number;
}

interface FinalDecodeResult {
  candidate: SpectralCandidate;
  hypothesis: DecodeHypothesis;
  envelope: Envelope;
  harmonicPenalty: number;
}

interface PreviewDecodeResult {
  candidate: SpectralCandidate;
  hypothesis: DecodeHypothesis;
  envelope: Envelope;
}

interface TimingHandoff {
  timestampMs: number;
  leftUnitMs: number;
  rightUnitMs: number;
  gapMs: number;
  gain: number;
}

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

const HAM_EVIDENCE_RANKING_WEIGHT = 0.25;

function hamRankingBonus(
  evidence: number,
  decoderScore: number,
  validRatio: number,
  timingQuality: number,
) {
  if (decoderScore < 0.62 || validRatio < 0.9 || timingQuality < 0.55) return 0;
  // The interpreter exposes up to 0.10 evidence; only a quarter of that may
  // affect ranking. It can break a genuinely close physical tie, never erase
  // a meaningful timing/signal lead.
  return evidence * HAM_EVIDENCE_RANKING_WEIGHT;
}

const median = (values: number[]) => {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
};

function maximumEnvelopeCorrelation(
  left: Float32Array,
  right: Float32Array,
  maximumLag = 25,
) {
  const sharedLength = Math.min(left.length, right.length);
  if (sharedLength < 16) return 0;
  let best = -1;
  for (let lag = -maximumLag; lag <= maximumLag; lag += 1) {
    const leftStart = Math.max(0, -lag);
    const rightStart = Math.max(0, lag);
    const length = sharedLength - Math.abs(lag);
    if (length < 16) continue;
    let leftSum = 0;
    let rightSum = 0;
    for (let index = 0; index < length; index += 1) {
      leftSum += left[leftStart + index];
      rightSum += right[rightStart + index];
    }
    const leftMean = leftSum / length;
    const rightMean = rightSum / length;
    let covariance = 0;
    let leftVariance = 0;
    let rightVariance = 0;
    for (let index = 0; index < length; index += 1) {
      const leftValue = left[leftStart + index] - leftMean;
      const rightValue = right[rightStart + index] - rightMean;
      covariance += leftValue * rightValue;
      leftVariance += leftValue * leftValue;
      rightVariance += rightValue * rightValue;
    }
    const denominator = Math.sqrt(leftVariance * rightVariance);
    if (denominator > 0) best = Math.max(best, covariance / denominator);
  }
  return best;
}

function areCorrelatedHarmonicTracks(
  left: FinalDecodeResult,
  right: FinalDecodeResult,
  allowKeyingSidebandOffset = false,
) {
  const [lower, upper] = [left, right].sort(
    (first, second) => first.candidate.frequency - second.candidate.frequency,
  );
  if (lower.candidate.channel !== upper.candidate.channel) return false;
  const ratio = upper.candidate.frequency / lower.candidate.frequency;
  const harmonic = Math.round(ratio);
  const maximumRatioError = allowKeyingSidebandOffset ? 0.05 : 0.01;
  if (harmonic < 2 || harmonic > 6 ||
    Math.abs(ratio - harmonic) / harmonic > maximumRatioError) return false;
  if (lower.hypothesis.validRatio < 0.9 || lower.hypothesis.timingQuality < 0.55) return false;
  if (Math.abs(lower.envelope.offsetMs - upper.envelope.offsetMs) > lower.envelope.hopMs * 2) {
    return false;
  }
  return maximumEnvelopeCorrelation(lower.envelope.levelsDb, upper.envelope.levelsDb) >=
    (allowKeyingSidebandOffset ? 0.82 : 0.68);
}

function isSameCompleteCopy(left: DecodeHypothesis, right: DecodeHypothesis) {
  const normalize = (transcript: string) => transcript
    .toUpperCase()
    .replace(/\s+/g, " ")
    .trim();
  const leftCopy = normalize(left.transcript);
  const rightCopy = normalize(right.transcript);
  // Short procedural fragments such as CQ or 5NN are routinely sent by
  // independent stations. Only remove confidence competition when two lanes
  // contain the same sustained copy; both lanes remain visible to the user.
  return leftCopy.replace(/\s/g, "").length >= 12 && leftCopy === rightCopy;
}

function reportedHarmonicFundamental(
  winner: FinalDecodeResult,
  results: FinalDecodeResult[],
) {
  const winnerWords = new Set(
    winner.hypothesis.transcript.toUpperCase().match(/[A-Z0-9]{3,}/g) ?? [],
  );
  const related = results.filter((lower) => {
    if (lower === winner || lower.candidate.channel !== winner.candidate.channel ||
      lower.candidate.frequency >= winner.candidate.frequency ||
      lower.hypothesis.validRatio < 0.9 || winner.hypothesis.validRatio < 0.9 ||
      lower.hypothesis.timingQuality < 0.5 || winner.hypothesis.timingQuality < 0.6) return false;
    const ratio = winner.candidate.frequency / lower.candidate.frequency;
    const harmonic = Math.round(ratio);
    if (harmonic < 2 || harmonic > 6 || Math.abs(ratio - harmonic) / harmonic > 0.003) return false;
    const wpmRatio = Math.max(lower.hypothesis.wpm, winner.hypothesis.wpm) /
      Math.max(1, Math.min(lower.hypothesis.wpm, winner.hypothesis.wpm));
    if (wpmRatio > 1.15 ||
      Math.abs(lower.envelope.offsetMs - winner.envelope.offsetMs) > lower.envelope.hopMs * 2 ||
      maximumEnvelopeCorrelation(lower.envelope.levelsDb, winner.envelope.levelsDb) < 0.6) return false;
    const lowerWords = lower.hypothesis.transcript.toUpperCase().match(/[A-Z0-9]{3,}/g) ?? [];
    return lowerWords.some((word) => winnerWords.has(word));
  });
  if (!related.length) return undefined;
  return Math.min(...related.map((result) => result.candidate.frequency));
}

function applyCorrelatedHarmonicPenalties(results: FinalDecodeResult[]) {
  for (let leftIndex = 0; leftIndex < results.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < results.length; rightIndex += 1) {
      const ordered = [results[leftIndex], results[rightIndex]].sort(
        (left, right) => left.candidate.frequency - right.candidate.frequency,
      );
      const [lower, upper] = ordered;
      if (!areCorrelatedHarmonicTracks(lower, upper)) continue;
      if (upper.hypothesis.baseScore - lower.hypothesis.baseScore > 0.07) continue;
      upper.harmonicPenalty = Math.max(upper.harmonicPenalty, 0.06);
    }
  }
}

function dedupeTransparentChannelCopies(results: PreviewDecodeResult[]) {
  // Dual-mono recordings are common, and evaluating the byte-identical track
  // from both channels can consume every finalist slot before a fundamental or
  // a genuinely independent station is refined. Keep receiver diversity when
  // the envelopes differ; collapse only effectively transparent channel copies.
  const kept: PreviewDecodeResult[] = [];
  for (const result of results) {
    const duplicateIndex = kept.findIndex((other) =>
      other.candidate.channel !== result.candidate.channel &&
      Math.abs(other.candidate.frequency - result.candidate.frequency) < 5 &&
      Math.abs(other.envelope.offsetMs - result.envelope.offsetMs) <= result.envelope.hopMs * 2 &&
      maximumEnvelopeCorrelation(other.envelope.levelsDb, result.envelope.levelsDb) >= 0.995
    );
    if (duplicateIndex < 0) {
      kept.push(result);
      continue;
    }
    const previous = kept[duplicateIndex];
    const resultQuality = [
      result.hypothesis.baseScore,
      result.hypothesis.timingQuality,
      result.hypothesis.contrastDb,
    ];
    const previousQuality = [
      previous.hypothesis.baseScore,
      previous.hypothesis.timingQuality,
      previous.hypothesis.contrastDb,
    ];
    const resultIsCleaner = resultQuality.some((value, index) =>
      value > previousQuality[index] &&
      resultQuality.slice(0, index).every((prior, priorIndex) => prior === previousQuality[priorIndex])
    );
    if (resultIsCleaner) kept[duplicateIndex] = result;
  }
  return kept;
}

const COMMON_BIGRAMS = new Set(
  "TH HE IN ER AN RE ON AT EN ND TI ES OR TE OF ED IS IT AL AR ST TO NT NG SE HA AS OU IO LE VE CO ME DE HI RI RO IC NE EA RA CE LI CH LL BE MA SI OM UR WI WA AV EG GI NN FI BA AD RS MO CW CQ QR QL QU BR OW FO OX LA AZ KE EY LP PY CA YE RD DA AY SO UN GR AP PP VY IM GL".split(" "),
);

const COMMON_WORDS = new Set(
  [
    ..."A I THE AND OF TO IN IS IT THIS THAT FOR FROM WITH ON AT BY AS BE OR ARE WAS WERE YOU YOUR MY ME WE OUR THEY HE SHE NAME CALL CALLED CALLING RADIO CW MORSE CODE SIGNAL COPY AGAIN PLEASE PSE THANKS TNX GOOD GREAT BAD FIST TEST NOW WPM TRANSITION FILE FOLLOWS WEATHER ANTENNA POWER RIG STATION OPERATOR MORNING AFTERNOON EVENING YESTERDAY SOUNDED HAPPY VERY HELLO HI ABOUT AFTER ALL ALSO BACK BEEN BEFORE BEST BOTH BUT CAN COME DAY DID DO DOWN EACH FIRST FUN GET GIVE GO HAD HAS HAVE HERE HOW INTO JUST KNOW LIKE LITTLE LIVE LOCAL LONG LOOK MADE MAKE MORE MOST MUCH NEW NO NOT ONLY OTHER OUT OVER PEOPLE PROBLEM RIGHT SAID SEE SOME SUM TAKE THAN THEM THEN THERE THESE TIME TWO UP USE VIDEO WAY WELL WHAT WHEN WHERE WHICH WHO WILL WORK WOULD YES ES VY ESVY IM GL CQ DE QRZ QSO QTH RST CLUB K KN AR SK R".split(" "),
    ...[...HAM_CW_TERMS].filter((term) => isHamCwScoringTerm(term)),
  ],
);

function makeHann(size: number) {
  return Float32Array.from({ length: size }, (_, index) =>
    0.5 - 0.5 * Math.cos(2 * Math.PI * index / Math.max(1, size - 1)),
  );
}

function magnitudeAtFrame(
  samples: Float32Array,
  start: number,
  size: number,
  frequency: number,
  sampleRate: number,
  window: Float32Array,
  windowSum: number,
) {
  const omega = 2 * Math.PI * frequency / sampleRate;
  const coefficient = 2 * Math.cos(omega);
  let previous = 0;
  let previousTwo = 0;
  for (let index = 0; index < size; index += 1) {
    const value = (samples[start + index] || 0) * window[index] + coefficient * previous - previousTwo;
    previousTwo = previous;
    previous = value;
  }
  const real = previous - previousTwo * Math.cos(omega);
  const imaginary = previousTwo * Math.sin(omega);
  return 2 * Math.hypot(real, imaginary) / Math.max(1, windowSum);
}

function discoverCandidates(
  channels: Float32Array[],
  sampleRate: number,
  options: OfflineDecodeOptions,
  requestedStartSeconds = 0,
  requestedLimit?: number,
) {
  const minimum = clamp(options.minimumFrequency ?? 200, 80, sampleRate * 0.45);
  const maximum = clamp(options.maximumFrequency ?? 4000, minimum + 50, sampleRate * 0.45);
  const step = 25;
  const frequencies: number[] = [];
  for (let frequency = minimum; frequency <= maximum; frequency += step) frequencies.push(frequency);
  const windowSize = Math.max(64, Math.round(sampleRate * 0.032));
  const hopSize = Math.max(16, Math.round(sampleRate * 0.016));
  const window = makeHann(windowSize);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  const recordingLength = channels[0]?.length ?? 0;
  const previewLength = Math.min(recordingLength, Math.round(sampleRate * 4.5));
  const previewStart = Math.round(clamp(
    requestedStartSeconds * sampleRate,
    0,
    Math.max(0, recordingLength - previewLength),
  ));
  const previewEnd = previewStart + previewLength;
  const all: SpectralCandidate[] = [];

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const samples = channels[channelIndex];
    const lows = new Float64Array(frequencies.length);
    const highs = new Float64Array(frequencies.length);
    const activity = new Float64Array(frequencies.length);
    const previousDb = new Float64Array(frequencies.length);
    let frames = 0;
    for (let start = previewStart; start + windowSize <= previewEnd; start += hopSize) {
      for (let index = 0; index < frequencies.length; index += 1) {
        const magnitude = magnitudeAtFrame(
          samples, start, windowSize, frequencies[index], sampleRate, window, windowSum,
        );
        const levelDb = 20 * Math.log10(magnitude + 1e-9);
        if (frames === 0) {
          lows[index] = levelDb;
          highs[index] = levelDb;
        } else {
          lows[index] += (levelDb - lows[index]) * (levelDb < lows[index] ? 0.28 : 0.018);
          highs[index] += (levelDb - highs[index]) * (levelDb > highs[index] ? 0.28 : 0.035);
          activity[index] = activity[index] * 0.9 + Math.min(24, Math.abs(levelDb - previousDb[index])) * 0.1;
        }
        previousDb[index] = levelDb;
      }
      frames += 1;
    }
    const spectralBaseline = median([...highs]);
    const scores = frequencies.map((frequency, index) => {
      const prominence = highs[index] - spectralBaseline;
      const contrast = highs[index] - lows[index];
      const keyedBonus = Math.min(9, Math.max(0, contrast - 4) * 0.2 + activity[index] * 0.45);
      return { frequency, score: prominence + keyedBonus };
    });
    for (let index = 1; index < scores.length - 1; index += 1) {
      if (scores[index].score < scores[index - 1].score || scores[index].score < scores[index + 1].score) continue;
      all.push({
        frequency: scores[index].frequency,
        channel: channelIndex,
        discoveryScore: scores[index].score,
        previewStartSeconds: previewStart / sampleRate,
      });
    }
  }

  const separated: SpectralCandidate[] = [];
  const candidateLimit = requestedLimit === undefined
    ? Math.min(8, channels.length * 4)
    : clamp(Math.round(requestedLimit), 1, 24);
  for (const candidate of all.sort((left, right) => right.discoveryScore - left.discoveryScore)) {
    if (separated.some((kept) => kept.channel === candidate.channel &&
      Math.abs(kept.frequency - candidate.frequency) < 55)) continue;
    separated.push(candidate);
    if (separated.length >= candidateLimit) break;
  }
  return separated;
}

function refineFrequency(
  samples: Float32Array,
  sampleRate: number,
  coarseFrequency: number,
  requestedStartSeconds = 0,
) {
  const windowSize = Math.max(128, Math.round(sampleRate * 0.064));
  const window = makeHann(windowSize);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  const previewLength = Math.min(samples.length, Math.round(sampleRate * 4.5));
  const previewStart = Math.round(clamp(
    requestedStartSeconds * sampleRate,
    0,
    Math.max(0, samples.length - previewLength),
  ));
  const previewEnd = previewStart + previewLength;
  const probeStarts: number[] = [];
  const probes = 18;
  for (let index = 0; index < probes; index += 1) {
    probeStarts.push(Math.max(previewStart, Math.min(previewEnd - windowSize,
      previewStart + Math.round(
        index / Math.max(1, probes - 1) * Math.max(0, previewLength - windowSize),
      ))));
  }
  let bestFrequency = coarseFrequency;
  let bestPower = -Infinity;
  for (let frequency = coarseFrequency - 16; frequency <= coarseFrequency + 16; frequency += 1) {
    let power = 0;
    for (const start of probeStarts) {
      const magnitude = magnitudeAtFrame(
        samples, start, windowSize, frequency, sampleRate, window, windowSum,
      );
      power += magnitude * magnitude;
    }
    if (power > bestPower) {
      bestPower = power;
      bestFrequency = frequency;
    }
  }
  return bestFrequency;
}

function clusterLevels(levels: Float32Array) {
  const finite = [...levels].filter(Number.isFinite).sort((left, right) => left - right);
  if (finite.length === 0) return { lowDb: -90, highDb: -30 };
  let low = finite[Math.floor(finite.length * 0.22)];
  let high = finite[Math.floor(finite.length * 0.78)];
  for (let iteration = 0; iteration < 12; iteration += 1) {
    let lowSum = 0;
    let lowCount = 0;
    let highSum = 0;
    let highCount = 0;
    const midpoint = (low + high) / 2;
    for (const level of finite) {
      if (level <= midpoint) {
        lowSum += level;
        lowCount += 1;
      } else {
        highSum += level;
        highCount += 1;
      }
    }
    if (lowCount) low = lowSum / lowCount;
    if (highCount) high = highSum / highCount;
  }
  return { lowDb: low, highDb: high };
}

function extractEnvelope(
  samples: Float32Array,
  sampleRate: number,
  frequency: number,
  maximumSeconds?: number,
  requestedStartSeconds = 0,
): Envelope {
  const windowSize = Math.max(64, Math.round(sampleRate * 0.024));
  const hopSize = Math.max(8, Math.round(sampleRate * 0.002));
  const requestedLength = maximumSeconds
    ? Math.min(samples.length, Math.round(maximumSeconds * sampleRate))
    : samples.length;
  const startSample = Math.round(clamp(
    requestedStartSeconds * sampleRate,
    0,
    Math.max(0, samples.length - requestedLength),
  ));
  const maximumLength = Math.min(requestedLength, samples.length - startSample);
  const frameCount = Math.max(0, Math.floor((maximumLength - windowSize) / hopSize) + 1);
  const levelsDb = new Float32Array(frameCount);
  const window = makeHann(windowSize);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const magnitude = magnitudeAtFrame(
      samples, startSample + frame * hopSize, windowSize, frequency, sampleRate, window, windowSum,
    );
    levelsDb[frame] = Math.max(-120, 20 * Math.log10(magnitude + 1e-9));
  }
  const clusters = clusterLevels(levelsDb);
  return {
    levelsDb,
    hopMs: hopSize / sampleRate * 1000,
    offsetMs: (startSample + windowSize / 2) / sampleRate * 1000,
    ...clusters,
  };
}

function buildRuns(
  envelope: Envelope,
  thresholdDb: number,
  hysteresisDb: number,
  debounceMs: number,
) {
  const states = new Uint8Array(envelope.levelsDb.length);
  let keyed = false;
  for (let index = 0; index < envelope.levelsDb.length; index += 1) {
    const level = envelope.levelsDb[index];
    if (!keyed && level >= thresholdDb + hysteresisDb) keyed = true;
    else if (keyed && level <= thresholdDb - hysteresisDb) keyed = false;
    states[index] = keyed ? 1 : 0;
  }

  // Debounce symmetrically in the state domain. Interior runs shorter than the
  // requested time are impulses/dropouts and inherit their matching neighbors.
  const minimumFrames = Math.max(1, Math.round(debounceMs / envelope.hopMs));
  for (let pass = 0; pass < 3; pass += 1) {
    let start = 0;
    while (start < states.length) {
      let end = start + 1;
      while (end < states.length && states[end] === states[start]) end += 1;
      if (end - start < minimumFrames) {
        const left = start > 0 ? states[start - 1] : states[end] ?? states[start];
        const right = end < states.length ? states[end] : left;
        if (left === right) states.fill(left, start, end);
      }
      start = end;
    }
  }

  const runs: GateRun[] = [];
  let start = 0;
  while (start < states.length) {
    let end = start + 1;
    while (end < states.length && states[end] === states[start]) end += 1;
    runs.push({
      keyed: states[start] === 1,
      startMs: envelope.offsetMs + start * envelope.hopMs,
      endMs: envelope.offsetMs + end * envelope.hopMs,
    });
    start = end;
  }
  return runs;
}

function fitUnit(runs: GateRun[], hintWpm: number, ignoreStuckCarrier = false) {
  const hintUnit = 1200 / clamp(hintWpm, 5, 60);
  const maximumPlausibleMark = Math.max(1200, hintUnit * 8);
  const marks = runs.filter((run) => run.keyed)
    .map((run) => run.endMs - run.startMs)
    .filter((mark) => !ignoreStuckCarrier || mark <= maximumPlausibleMark);
  const gaps = runs.filter((run, index) => !run.keyed && index > 0 && index < runs.length - 1)
    .map((run) => run.endMs - run.startMs)
    .filter((gap) => gap < 3000);
  let bestUnit = hintUnit;
  let bestLoss = Infinity;
  for (let unit = 20; unit <= 240; unit += 1) {
    let loss = 0;
    for (const mark of marks) {
      const dot = ((mark - unit) / (unit * 0.42)) ** 2;
      const dash = ((mark - unit * 3) / (unit * 0.9)) ** 2 + 0.04;
      loss += Math.min(9, dot, dash);
    }
    for (const gap of gaps) {
      const element = ((gap - unit) / (unit * 0.48)) ** 2;
      const character = ((gap - unit * 3) / (unit * 1.25)) ** 2 + 0.08;
      const word = ((gap - unit * 7) / (unit * 2.2)) ** 2 + 0.12;
      loss += Math.min(6, element, character, word);
    }
    loss += (marks.length + gaps.length) * 0.025 * Math.abs(Math.log(unit / hintUnit));
    if (loss < bestLoss) {
      bestLoss = loss;
      bestUnit = unit;
    }
  }
  const deviations = marks.map((mark) => Math.min(
    Math.abs(mark - bestUnit) / bestUnit,
    Math.abs(mark - bestUnit * 3) / (bestUnit * 3),
  ));
  return {
    unitMs: bestUnit,
    timingQuality: Math.exp(-median(deviations) * 2.4),
    marks: marks.length,
  };
}

function fitHandoffUnit(runs: GateRun[]) {
  const marks = runs.filter((run) => run.keyed).map((run) => run.endMs - run.startMs);
  const gaps = runs.filter((run, index) => !run.keyed && index > 0 && index < runs.length - 1)
    .map((run) => run.endMs - run.startMs)
    .filter((gap) => gap < 3000);
  let unitMs = 60;
  let loss = Infinity;
  for (let unit = 20; unit <= 240; unit += 1) {
    let nextLoss = 0;
    for (const mark of marks) {
      const dot = ((mark - unit) / (unit * 0.42)) ** 2;
      const dash = ((mark - unit * 3) / (unit * 0.9)) ** 2 + 0.04;
      nextLoss += Math.min(9, dot, dash);
    }
    for (const gap of gaps) {
      const element = ((gap - unit) / (unit * 0.48)) ** 2;
      const character = ((gap - unit * 3) / (unit * 1.25)) ** 2 + 0.08;
      const word = ((gap - unit * 7) / (unit * 2.2)) ** 2 + 0.12;
      nextLoss += Math.min(6, element, character, word);
    }
    if (nextLoss < loss) {
      loss = nextLoss;
      unitMs = unit;
    }
  }
  return { unitMs, loss, observations: marks.length + gaps.length };
}

function detectTimingHandoffs(envelope: Envelope, hypothesis: DecodeHypothesis) {
  const runs = buildRuns(
    envelope,
    hypothesis.thresholdDb,
    hypothesis.hysteresisDb,
    hypothesis.debounceMs,
  );
  const candidates: TimingHandoff[] = [];
  const sideRunCount = 48;
  for (let index = 1; index < runs.length - 1; index += 1) {
    const boundary = runs[index];
    if (boundary.keyed || boundary.endMs - boundary.startMs < 120) continue;
    const leftRuns = runs.slice(Math.max(0, index - sideRunCount), index);
    const rightRuns = runs.slice(index + 1, Math.min(runs.length, index + sideRunCount + 1));
    const left = fitHandoffUnit(leftRuns);
    const right = fitHandoffUnit(rightRuns);
    if (left.observations < 20 || right.observations < 20) continue;

    const unitRatio = Math.max(left.unitMs, right.unitMs) /
      Math.max(1, Math.min(left.unitMs, right.unitMs));
    const gapMs = boundary.endMs - boundary.startMs;
    const credibleSides = left.loss / left.observations <= 0.8 &&
      right.loss / right.observations <= 0.8;
    if (!credibleSides || unitRatio < 1.3 ||
      gapMs < Math.min(left.unitMs, right.unitMs) * 6) continue;

    const leftMiddle = Math.floor(leftRuns.length / 2);
    const rightMiddle = Math.floor(rightRuns.length / 2);
    const leftFar = fitHandoffUnit(leftRuns.slice(0, leftMiddle));
    const leftNear = fitHandoffUnit(leftRuns.slice(leftMiddle));
    const rightNear = fitHandoffUnit(rightRuns.slice(0, rightMiddle));
    const rightFar = fitHandoffUnit(rightRuns.slice(rightMiddle));
    const halfFits = [leftFar, leftNear, rightNear, rightFar];
    if (halfFits.some((fit) => fit.observations < 8)) continue;
    const leftStability = Math.max(leftFar.unitMs, leftNear.unitMs) /
      Math.max(1, Math.min(leftFar.unitMs, leftNear.unitMs));
    const rightStability = Math.max(rightFar.unitMs, rightNear.unitMs) /
      Math.max(1, Math.min(rightFar.unitMs, rightNear.unitMs));
    if (leftStability > 1.18 || rightStability > 1.18) continue;

    const pooled = fitHandoffUnit([...leftRuns, boundary, ...rightRuns]);
    const gain = (pooled.loss - left.loss - right.loss) / Math.max(1, pooled.observations);
    if (gain < 0.18) continue;
    candidates.push({
      timestampMs: (boundary.startMs + boundary.endMs) / 2,
      leftUnitMs: left.unitMs,
      rightUnitMs: right.unitMs,
      gapMs,
      gain,
    });
  }

  // A single handoff creates several nearby statistical candidates when a
  // window straddles it. Keep the longest physical separating gap in each
  // neighborhood; at most three actual operator changes are useful in one pass.
  const selected: TimingHandoff[] = [];
  for (const candidate of candidates.sort((left, right) => left.timestampMs - right.timestampMs)) {
    const nearbyIndex = selected.findIndex((kept) =>
      Math.abs(kept.timestampMs - candidate.timestampMs) < 1500
    );
    if (nearbyIndex < 0) selected.push(candidate);
    else if (candidate.gapMs > selected[nearbyIndex].gapMs) selected[nearbyIndex] = candidate;
  }
  return selected.sort((left, right) => left.timestampMs - right.timestampMs).slice(0, 3);
}

function sliceEnvelope(envelope: Envelope, startMs: number, endMs: number) {
  const startFrame = clamp(Math.round((startMs - envelope.offsetMs) / envelope.hopMs), 0, envelope.levelsDb.length);
  const endFrame = clamp(Math.round((endMs - envelope.offsetMs) / envelope.hopMs), startFrame, envelope.levelsDb.length);
  const levelsDb = envelope.levelsDb.slice(startFrame, endFrame);
  const clusters = clusterLevels(levelsDb);
  return {
    levelsDb,
    hopMs: envelope.hopMs,
    offsetMs: envelope.offsetMs + startFrame * envelope.hopMs,
    ...clusters,
  };
}

function languageQuality(transcript: string) {
  // Score letter pairs inside words only. Removing separators first used to
  // manufacture plausible pairs across spaces and around digits, which made a
  // short callsign interferer look more like prose than a complete message.
  const letterWords = transcript.match(/[A-Z]+/g) ?? [];
  let bigrams = 0;
  let acceptedBigrams = 0;
  for (const letters of letterWords) {
    for (let index = 1; index < letters.length; index += 1) {
      const pair = letters.slice(index - 1, index + 1);
      bigrams += 1;
      if (COMMON_BIGRAMS.has(pair)) acceptedBigrams += 1;
    }
  }
  const words = transcript.match(/[A-Z0-9]+(?:\/[A-Z0-9]+)?/g) ?? [];
  let wordPoints = 0;
  let implausibleDigits = 0;
  for (const word of words) {
    if (COMMON_WORDS.has(word)) wordPoints += 1;
    else if (isHamCwScoringTerm(word)) wordPoints += 1.05;
    else if (/^[A-Z]{1,2}\d[A-Z0-9]{1,5}(?:\/[A-Z0-9]+)?$/.test(word)) wordPoints += 0.8;
    else {
      implausibleDigits += (word.match(/\d/g) ?? []).length;
      if (word.length >= 3 && [...word].every((value) => /[A-Z]/.test(value))) wordPoints += 0.25;
    }
  }
  const bigramRatio = bigrams ? acceptedBigrams / bigrams : 0;
  const wordRatio = words.length ? wordPoints / words.length : 0;
  return clamp(bigramRatio * 0.68 + wordRatio * 0.32 - implausibleDigits * 0.09, 0, 1);
}

function messageEvidenceQuality(
  transcript: string,
  tokens: OfflineDecodeToken[],
  nextLanguageQuality: number,
) {
  const words = transcript.match(/[A-Z0-9]+(?:\/[A-Z0-9]+)?/g) ?? [];
  const characterCount = tokens.filter((token) => !token.space && token.value !== "�").length;
  if (!characterCount || !words.length) return 0;

  let shapedCharacters = 0;
  let totalCharacters = 0;
  for (const word of words) {
    totalCharacters += word.length;
    if (/^[A-Z]{2,}$/.test(word)) shapedCharacters += word.length;
    else if (COMMON_WORDS.has(word)) shapedCharacters += word.length * 0.9;
    else if (/^[A-Z]{1,2}\d[A-Z0-9]{1,5}(?:\/[A-Z0-9]+)?$/.test(word)) {
      // Callsigns remain first-class Morse content, but language assist should
      // prefer a sustained readable message when two stations overlap.
      shapedCharacters += word.length * 0.45;
    } else if (/^\d+$/.test(word)) shapedCharacters += word.length * 0.4;
    else shapedCharacters += word.length * 0.12;
  }

  const amount = 1 - Math.exp(-characterCount / 20);
  const shape = shapedCharacters / Math.max(1, totalCharacters);
  const linguisticSupport = 0.55 + nextLanguageQuality * 0.45;
  return clamp(amount * shape * linguisticSupport, 0, 1);
}

function structuredTrafficRankingAdjustment(hypothesis: DecodeHypothesis) {
  // Long contest/traffic messages are allowed a tiny carrier-selection repair
  // without changing their literal decode, threshold choice, confidence, or
  // convergence. The ordinary hypothesis scorer is intentionally conservative
  // about figures, and its absolute replacement penalty can otherwise let a
  // short clean key-click ridge beat a nearly complete QTC by one bad symbol.
  if (hypothesis.validRatio < 0.9 || hypothesis.timingQuality < 0.5) return 0;
  const words = hypothesis.transcript.match(/[A-Z0-9]+(?:\/[A-Z0-9]+)?/g) ?? [];
  const structured = words.filter((word) =>
    /^\d{1,6}(?:\/\d{1,3})?$/.test(word) ||
    /^(?=.*\d)[0-9ANT]{2,6}$/.test(word) ||
    /^(?=.{3,10}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+(?:\/[A-Z0-9]+)?$/.test(word)
  );
  if (structured.length < 3) return 0;

  const characters = hypothesis.tokens.filter((token) => !token.space);
  const replacements = characters.filter((token) => token.value === "�").length;
  const normalizedReplacementPenalty = Math.min(
    0.08,
    0.12 * replacements / Math.sqrt(Math.max(1, characters.length)),
  );
  const replacementRelief = Math.max(0, replacements * 0.08 - normalizedReplacementPenalty);
  const structuredRatio = structured.reduce((sum, word) => sum + word.length, 0) /
    Math.max(1, words.reduce((sum, word) => sum + word.length, 0));
  const qtcHeader = words[0] === "QTC" && /^\d+\/\d+$/.test(words[1] ?? "") ? 0.015 : 0;
  const timingSupport = clamp((hypothesis.timingQuality - 0.5) / 0.35, 0, 1) * 0.03;
  return clamp(
    replacementRelief + structuredRatio * 0.06 + qtcHeader + timingSupport,
    0,
    0.1,
  );
}

function wordTokens(tokens: OfflineDecodeToken[]) {
  const words: OfflineDecodeToken[][] = [];
  let current: OfflineDecodeToken[] = [];
  for (const token of tokens) {
    if (token.space) {
      if (current.length) words.push(current);
      current = [];
    } else current.push(token);
  }
  if (current.length) words.push(current);
  return words;
}

function wordPlausibility(word: string) {
  if (COMMON_WORDS.has(word)) return 2.4;
  if (isHamCwScoringTerm(word)) return 2.5;
  if (/^[A-Z]{1,2}\d[A-Z0-9]{1,5}(?:\/[A-Z0-9]+)?$/.test(word)) return 1.8;
  if (/[�]/.test(word)) return -2;
  if (/\d/.test(word)) return -0.8;
  let pairs = 0;
  let accepted = 0;
  for (let index = 1; index < word.length; index += 1) {
    pairs += 1;
    if (COMMON_BIGRAMS.has(word.slice(index - 1, index + 1))) accepted += 1;
  }
  return pairs ? accepted / pairs : 0;
}

function normalizeTokenSpacing(tokens: OfflineDecodeToken[]) {
  const normalized: OfflineDecodeToken[] = [];
  for (const token of tokens) {
    if (!token.space && /^[,.?!:;]$/.test(token.value) && normalized.at(-1)?.space) {
      normalized.pop();
    }
    normalized.push(token);
  }

  // A dash preceded by whitespace is punctuation, not an in-word hyphen. If
  // the sender ran the following word into it, restore only that unambiguous
  // display boundary while retaining both transmitted dash characters.
  for (let index = 1; index < normalized.length - 1; index += 1) {
    if (normalized[index].value !== "-" || !normalized[index - 1].space || normalized[index + 1].space) {
      continue;
    }
    normalized.splice(index + 1, 0, {
      value: " ",
      pattern: "/",
      confidence: 0.7,
      timestamp: normalized[index].timestamp,
      space: true,
    });
    index += 1;
  }
  return normalized;
}

function consensusHypothesis(variants: DecodeHypothesis[], languageAssist: boolean) {
  const best = variants[0];
  if (!best || !languageAssist) return best;
  const bestWords = wordTokens(best.tokens);
  if (bestWords.length < 2) return best;
  const credible = variants.filter((variant) =>
    variant.score >= best.score - 0.045 && wordTokens(variant.tokens).length === bestWords.length,
  );
  if (credible.length < 2) return best;
  const selectedWords: OfflineDecodeToken[][] = [];
  let changed = false;
  for (let wordIndex = 0; wordIndex < bestWords.length; wordIndex += 1) {
    const choices = new Map<string, { tokens: OfflineDecodeToken[]; support: number }>();
    for (const variant of credible) {
      const tokens = wordTokens(variant.tokens)[wordIndex];
      const word = tokens.map((token) => token.value).join("");
      const support = Math.exp((variant.score - best.score) * 18);
      const previous = choices.get(word);
      if (previous) previous.support += support;
      else choices.set(word, { tokens, support });
    }
    const ranked = [...choices.entries()].sort((left, right) =>
      right[1].support + wordPlausibility(right[0]) -
      (left[1].support + wordPlausibility(left[0])),
    );
    const selected = ranked[0]?.[1].tokens ?? bestWords[wordIndex];
    if (selected.map((token) => token.value).join("") !==
      bestWords[wordIndex].map((token) => token.value).join("")) changed = true;
    selectedWords.push(selected);
  }
  if (!changed) return best;
  const tokens: OfflineDecodeToken[] = [];
  for (let index = 0; index < selectedWords.length; index += 1) {
    if (index > 0) {
      const timestamp = selectedWords[index - 1].at(-1)?.timestamp ?? 0;
      tokens.push({ value: " ", pattern: "/", confidence: 0.72, timestamp, space: true });
    }
    tokens.push(...selectedWords[index]);
  }
  const transcript = tokens.map((token) => token.value).join("");
  const hamEvidence = hamMessageEvidence(transcript).score;
  return {
    ...best,
    tokens,
    transcript,
    hamEvidence,
    score: clamp(
      best.baseScore + hamRankingBonus(
        hamEvidence,
        best.decoderScore,
        best.validRatio,
        best.timingQuality,
      ),
      -2,
      1.2,
    ),
  };
}

function decodeRuns(
  runs: GateRun[],
  unitMs: number,
  characterUnits: number,
  wordUnits: number,
  languageAssist: boolean,
  ignoreStuckCarrier = false,
) {
  const tokens: OfflineDecodeToken[] = [];
  let pattern = "";
  let patternConfidence: number[] = [];
  let leadingGapMs = 0;
  const emit = (timestamp: number) => {
    if (!pattern) return;
    let emittedPattern = pattern;
    let repaired = false;
    if (languageAssist && leadingGapMs >= unitMs * 3.8 && tokens.length) {
      const previous = [...tokens].reverse().find((token) => !token.space);
      const priorCharacters: string[] = [];
      for (let index = tokens.length - 1; index >= 0; index -= 1) {
        const token = tokens[index];
        if (token.space) {
          if (priorCharacters.length) break;
          continue;
        }
        priorCharacters.unshift(token.value);
      }
      const priorWord = priorCharacters.join("");
      const originalValue = isMorseCharacter(pattern) ? decodeMorse(pattern) : "";
      const originalPair = `${previous?.value ?? ""}${originalValue}`;
      const repairs = [
        { symbol: ".", expectedUnits: 5, spreadUnits: 1.6 },
        { symbol: "-", expectedUnits: 7, spreadUnits: 2.1 },
      ];
      for (const candidate of repairs) {
        // This guard prevents a long ordinary character gap from duplicating
        // the first observed element. The useful case is a different leading
        // element that faded completely before the surviving pattern.
        if (candidate.symbol === pattern[0]) continue;
        const nextPattern = candidate.symbol + pattern;
        if (!isMorseCharacter(nextPattern)) continue;
        const timingError = Math.abs(leadingGapMs / unitMs - candidate.expectedUnits) /
          candidate.spreadUnits;
        const nextValue = decodeMorse(nextPattern);
        const nextPair = `${previous?.value ?? ""}${nextValue}`;
        const joinedWord = priorWord + nextValue;
        const plausibleWord = [...COMMON_WORDS].some((word) =>
          word === joinedWord || word.length > joinedWord.length && word.startsWith(joinedWord),
        );
        if (timingError <= 1.15 && plausibleWord &&
          COMMON_BIGRAMS.has(nextPair) && !COMMON_BIGRAMS.has(originalPair)) {
          emittedPattern = nextPattern;
          repaired = true;
          if (tokens.at(-1)?.space) tokens.pop();
          break;
        }
      }
    }
    const valid = isMorseCharacter(emittedPattern);
    tokens.push({
      value: valid ? decodeMorse(emittedPattern) : "�",
      pattern: emittedPattern,
      confidence: valid
        ? repaired
          ? 0.42
          : patternConfidence.reduce((sum, value) => sum + value, 0) / Math.max(1, patternConfidence.length)
        : 0.2,
      timestamp,
    });
    pattern = "";
    patternConfidence = [];
    leadingGapMs = 0;
  };

  for (let index = 0; index < runs.length; index += 1) {
    const run = runs[index];
    if (!run.keyed) continue;
    if (!pattern) {
      const gap = runs[index - 1];
      leadingGapMs = gap && !gap.keyed ? gap.endMs - gap.startMs : 0;
    }
    const duration = run.endMs - run.startMs;
    if (ignoreStuckCarrier && duration > Math.max(1200, unitMs * 8)) {
      // A long unmodulated carrier is a receiver/tuning state, not a Morse
      // dash. Treat it as a hard boundary so a beacon that starts later in a
      // recording cannot acquire a false leading or trailing T.
      emit(run.startMs);
      leadingGapMs = 0;
      continue;
    }
    const dotError = Math.abs(duration - unitMs) / Math.max(1, unitMs);
    const dashError = Math.abs(duration - unitMs * 3) / Math.max(1, unitMs * 3);
    const dash = duration >= unitMs * 1.95;
    pattern += dash ? "-" : ".";
    patternConfidence.push(clamp(1 - Math.min(dotError, dashError) * 0.72, 0.25, 0.99));
    const gap = runs[index + 1];
    const gapDuration = gap && !gap.keyed ? gap.endMs - gap.startMs : Infinity;
    // A continuous SOS is the sole nine-element signal. Only let an eight-mark
    // prefix absorb a ninth element when it starts a message or follows an
    // actual word boundary; this prevents clipped callsigns such as WII800
    // from collapsing into a false distress marker.
    const standaloneSosPrefix = pattern === "...---.." &&
      (tokens.length === 0 || Boolean(tokens.at(-1)?.space));
    if (gapDuration >= unitMs * characterUnits || pattern.length >= 8 && !standaloneSosPrefix) {
      emit(run.endMs);
      if (gapDuration >= unitMs * wordUnits && tokens.length && !tokens.at(-1)?.space) {
        tokens.push({ value: " ", pattern: "/", confidence: 0.96, timestamp: run.endMs, space: true });
      }
    }
  }
  emit(runs.at(-1)?.endMs ?? 0);
  while (tokens.at(-1)?.space) tokens.pop();
  const normalizedTokens = normalizeTokenSpacing(tokens);
  const transcript = normalizedTokens.map((token) => token.value).join("").trim();
  const characters = normalizedTokens.filter((token) => !token.space);
  const valid = characters.filter((token) => token.value !== "�").length;
  return {
    tokens: normalizedTokens,
    transcript,
    validRatio: characters.length ? valid / characters.length : 0,
  };
}

function optimizeEnvelope(
  envelope: Envelope,
  options: OfflineDecodeOptions,
  ignoreStuckCarrier = false,
  presetRuns?: GateRun[],
) {
  const contrastDb = envelope.highDb - envelope.lowDb;
  let best: DecodeHypothesis | null = null;
  const unique = new Map<string, DecodeHypothesis>();
  let passCount = 0;
  const standardThresholdProfiles = [
    { fraction: 0.4, recoveryPenalty: 0 },
    { fraction: 0.46, recoveryPenalty: 0 },
    { fraction: 0.52, recoveryPenalty: 0 },
    { fraction: 0.58, recoveryPenalty: 0 },
    { fraction: 0.64, recoveryPenalty: 0 },
  ];
  const recoveryThresholdProfiles = [
    // High gates can reject chatter and expose marks hidden by a low-level
    // keyed residue. Treat them as recovery passes: they may rescue a failed
    // ordinary sweep, but need a meaningful quality lead to displace one.
    { fraction: 0.72, recoveryPenalty: 0.05 },
    { fraction: 0.8, recoveryPenalty: 0.06 },
  ];
  const hysteresisValues = presetRuns ? [0] : [1, 1.5, 2];
  const debounceValues = presetRuns ? [0] : [4, 6, 10];
  const characterBoundaries = [1.75, 2.05, 2.4];
  // Real hand keys often collapse nominal 7-unit word spaces toward ordinary
  // character spacing. Search the ambiguous 3.5–5.5 unit region as well and
  // let timing plus language consensus decide instead of hard-wiring 7 units.
  const wordBoundaries = options.languageAssist
    ? [3.4, 3.8, 4.4, 5.2, 6.2, 7.4]
    : [5.8, 6.8, 7.8];

  const evaluateThresholdProfiles = (profiles: typeof standardThresholdProfiles) => {
    for (const { fraction: thresholdFraction, recoveryPenalty } of
      presetRuns ? profiles.slice(0, 1) : profiles) {
      const thresholdDb = envelope.lowDb + contrastDb * thresholdFraction;
      for (const hysteresisDb of hysteresisValues) {
        for (const debounceMs of debounceValues) {
          const runs = presetRuns ?? buildRuns(envelope, thresholdDb, hysteresisDb, debounceMs);
          const fitted = fitUnit(runs, options.wpmHint, ignoreStuckCarrier);
          if (fitted.marks < 3) continue;
          for (const characterUnits of characterBoundaries) {
            for (const wordUnits of wordBoundaries) {
              passCount += 1;
              const decoded = decodeRuns(
                runs,
                fitted.unitMs,
                characterUnits,
                wordUnits,
                options.languageAssist,
                ignoreStuckCarrier,
              );
              const nextLanguageQuality = options.languageAssist ? languageQuality(decoded.transcript) : 0.5;
              const speedQuality = Math.exp(-0.5 * (
                Math.log((1200 / fitted.unitMs) / clamp(options.wpmHint, 5, 60)) / Math.log(1.5)
              ) ** 2);
              const contrastQuality = clamp((contrastDb - 7) / 25, 0, 1);
              const densityQuality = clamp(decoded.tokens.filter((token) => !token.space).length / 4, 0, 1);
              const replacementCount = decoded.tokens.filter((token) => token.value === "�").length;
              const decoderScore = (0.35 * fitted.timingQuality +
                0.25 * nextLanguageQuality +
                0.2 * speedQuality +
                0.14 * decoded.validRatio +
                0.04 * contrastQuality +
                0.02 * densityQuality) * densityQuality - replacementCount * 0.08;
              const messageQuality = options.languageAssist
                ? messageEvidenceQuality(decoded.transcript, decoded.tokens, nextLanguageQuality)
                : 0;
              const hamEvidence = options.languageAssist
                ? hamMessageEvidence(decoded.transcript).score
                : 0;
              // Decoder quality says whether a track is Morse-like; message
              // evidence says whether there is enough coherent content to trust
              // it over a simultaneous, shorter station. Sequence-aware ham
              // shorthand is a final capped tie-breaker. Keep the pre-context
              // score separate so lexical evidence cannot declare convergence.
              const baseScore = clamp(
                decoderScore + messageQuality * 0.2 - recoveryPenalty,
                -2,
                1.2,
              );
              const score = clamp(
                baseScore + hamRankingBonus(
                  hamEvidence,
                  decoderScore,
                  decoded.validRatio,
                  fitted.timingQuality,
                ),
                -2,
                1.2,
              );
              const hypothesis: DecodeHypothesis = {
                ...decoded,
                score,
                thresholdDb,
                thresholdFraction,
                hysteresisDb,
                debounceMs,
                characterUnits,
                wordUnits,
                unitMs: fitted.unitMs,
                wpm: 1200 / fitted.unitMs,
                timingQuality: fitted.timingQuality,
                languageQuality: nextLanguageQuality,
                messageQuality,
                hamEvidence,
                baseScore,
                decoderScore,
                contrastDb,
              };
              if (!best || score > best.score) best = hypothesis;
              const previous = unique.get(decoded.transcript);
              if (!previous || score > previous.score) unique.set(decoded.transcript, hypothesis);
            }
          }
        }
      }
    }
  };

  evaluateThresholdProfiles(standardThresholdProfiles);
  const standardBaseScore = best ? (best as DecodeHypothesis).baseScore : -Infinity;
  const standardValidRatio = best ? (best as DecodeHypothesis).validRatio : 0;
  if (!presetRuns && (standardBaseScore < 0.84 || standardValidRatio < 0.96)) {
    evaluateThresholdProfiles(recoveryThresholdProfiles);
  }
  const alternatives = [...unique.values()].sort((left, right) => right.score - left.score).slice(0, 12);
  return {
    best: consensusHypothesis(alternatives, options.languageAssist) ?? best,
    passCount,
    alternatives,
  };
}

function optimizeTimingHandoffSegments(
  envelope: Envelope,
  global: DecodeHypothesis,
  options: OfflineDecodeOptions,
  ignoreStuckCarrier: boolean,
) {
  const handoffs = detectTimingHandoffs(envelope, global);
  if (!handoffs.length) return { hypothesis: global, passCount: 0 };
  const envelopeEndMs = envelope.offsetMs + envelope.levelsDb.length * envelope.hopMs;
  const boundaries = [envelope.offsetMs, ...handoffs.map((handoff) => handoff.timestampMs), envelopeEndMs];
  const unitHints = [handoffs[0].leftUnitMs, ...handoffs.map((handoff) => handoff.rightUnitMs)];
  const segments: DecodeHypothesis[] = [];
  let passCount = 0;
  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const section = sliceEnvelope(envelope, boundaries[index], boundaries[index + 1]);
    if (section.levelsDb.length < 64) return { hypothesis: global, passCount };
    const optimized = optimizeEnvelope(
      section,
      { ...options, wpmHint: 1200 / unitHints[index] },
      ignoreStuckCarrier,
    );
    passCount += optimized.passCount;
    const hypothesis = optimized.best;
    if (!hypothesis || hypothesis.baseScore < 0.62 || hypothesis.decoderScore < 0.62 ||
      hypothesis.validRatio < 0.9 || hypothesis.timingQuality < 0.55) {
      return { hypothesis: global, passCount };
    }
    segments.push(hypothesis);
  }

  const segmentTokens = segments.map((segment) => {
    const tokens = [...segment.tokens];
    while (tokens[0]?.space) tokens.shift();
    while (tokens.at(-1)?.space) tokens.pop();
    return tokens;
  });
  const weights = segmentTokens.map((tokens) =>
    Math.max(1, tokens.filter((token) => !token.space).length)
  );
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  const weighted = (select: (hypothesis: DecodeHypothesis) => number) =>
    segments.reduce((sum, segment, index) => sum + select(segment) * weights[index], 0) /
    Math.max(1, totalWeight);
  const tokens: OfflineDecodeToken[] = [];
  for (let index = 0; index < segmentTokens.length; index += 1) {
    if (tokens.length && !tokens.at(-1)?.space && segmentTokens[index].length) {
      tokens.push({
        value: " ",
        pattern: "/",
        confidence: 0.5,
        timestamp: handoffs[index - 1].timestampMs,
        space: true,
      });
    }
    tokens.push(...segmentTokens[index]);
  }
  const characters = tokens.filter((token) => !token.space);
  const validCharacters = characters.filter((token) => token.value !== "�").length;
  const globalCharacters = global.tokens.filter((token) => !token.space);
  const globalValidCharacters = globalCharacters.filter((token) => token.value !== "�").length;
  const replacements = characters.length - validCharacters;
  const globalReplacements = globalCharacters.length - globalValidCharacters;
  const timingQuality = weighted((segment) => segment.timingQuality);
  const weightedBaseScore = weighted((segment) => segment.baseScore);
  if (validCharacters < globalValidCharacters || replacements > globalReplacements ||
    timingQuality < global.timingQuality - 0.05 || weightedBaseScore < global.baseScore - 0.02) {
    return { hypothesis: global, passCount };
  }

  // Carrier selection has already happened. Cap the score lift so separate
  // per-operator language fits can repair timing without manufacturing a new
  // lexical advantage or an overconfident convergence result.
  const baseScore = Math.min(weightedBaseScore, global.baseScore + 0.12);
  const decoderScore = weighted((segment) => segment.decoderScore);
  const validRatio = characters.length ? validCharacters / characters.length : 0;
  const languageQuality = weighted((segment) => segment.languageQuality);
  const messageQuality = weighted((segment) => segment.messageQuality);
  const hamEvidence = weighted((segment) => segment.hamEvidence);
  const unitMs = weighted((segment) => segment.unitMs);
  const score = clamp(
    baseScore + hamRankingBonus(hamEvidence, decoderScore, validRatio, timingQuality),
    -2,
    1.2,
  );
  return {
    hypothesis: {
      ...global,
      transcript: tokens.map((token) => token.value).join("").trim(),
      tokens,
      score,
      thresholdDb: weighted((segment) => segment.thresholdDb),
      unitMs,
      wpm: 1200 / unitMs,
      timingQuality,
      validRatio,
      languageQuality,
      messageQuality,
      hamEvidence,
      baseScore,
      decoderScore,
    },
    passCount,
  };
}

export function optimizeMorseRecording(
  channels: Float32Array[],
  sampleRate: number,
  options: OfflineDecodeOptions,
  onProgress?: (progress: OfflineDecodeProgress) => void,
): OfflineDecodeResult {
  if (!channels.length || !channels[0]?.length) throw new Error("The recording contains no audio samples.");
  onProgress?.({ phase: "survey", fraction: 0.05, detail: "Surveying keyed spectral tracks" });
  const discovered = discoverCandidates(channels, sampleRate, options);
  if (!discovered.length) throw new Error("No narrow-band carrier candidates were found.");

  const candidates: SpectralCandidate[] = [];
  for (const candidate of discovered) {
    const frequency = refineFrequency(
      channels[candidate.channel],
      sampleRate,
      candidate.frequency,
      candidate.previewStartSeconds,
    );
    if (candidates.some((kept) => kept.channel === candidate.channel && Math.abs(kept.frequency - frequency) < 45)) continue;
    candidates.push({ ...candidate, frequency });
    if (candidates.length >= 6) break;
  }
  onProgress?.({ phase: "candidates", fraction: 0.22, detail: `Testing ${candidates.length} carrier tracks` });

  let passCount = 0;
  const previewResults: PreviewDecodeResult[] = [];
  for (let index = 0; index < candidates.length; index += 1) {
    const candidate = candidates[index];
    const envelope = extractEnvelope(
      channels[candidate.channel],
      sampleRate,
      candidate.frequency,
      4.5,
      candidate.previewStartSeconds,
    );
    const optimized = optimizeEnvelope(envelope, options);
    passCount += optimized.passCount;
    if (optimized.best) previewResults.push({ candidate, hypothesis: optimized.best, envelope });
    onProgress?.({
      phase: "candidates",
      fraction: 0.22 + 0.33 * (index + 1) / candidates.length,
      detail: `Scored carrier ${index + 1} of ${candidates.length}`,
    });
  }
  const durationSeconds = channels[0].length / sampleRate;
  const isPhysicallyCrediblePreview = (hypothesis: DecodeHypothesis) => {
    const characters = hypothesis.tokens.filter((token) => !token.space).length;
    return characters >= 6 && hypothesis.decoderScore >= 0.52 &&
      hypothesis.validRatio >= 0.9 && hypothesis.timingQuality >= 0.55 &&
      hypothesis.wpm >= 5 && hypothesis.wpm <= 60;
  };
  const isStrongDeepRecovery = (hypothesis: DecodeHypothesis) => {
    const characters = hypothesis.tokens.filter((token) => !token.space).length;
    return characters >= 4 && hypothesis.baseScore >= 0.66 &&
      hypothesis.decoderScore >= 0.65 && hypothesis.validRatio >= 0.98 &&
      hypothesis.timingQuality >= 0.58 && hypothesis.wpm >= 5 && hypothesis.wpm <= 50;
  };
  if (durationSeconds <= 180 &&
    !previewResults.some((result) => isPhysicallyCrediblePreview(result.hypothesis))) {
    // A strong heterodyne, voice peak, or keyed harmonic can consume the small
    // normal shortlist even when every resulting track has implausible timing.
    // Only in that failure state, inspect a bounded deeper discovery list using
    // short preview envelopes. Promote at most two tracks, and only when their
    // physical decoder evidence independently clears a stricter Morse gate.
    const deepDiscovered = discoverCandidates(channels, sampleRate, options, 0, 20);
    const recoveries: PreviewDecodeResult[] = [];
    for (const deep of deepDiscovered) {
      const frequency = refineFrequency(
        channels[deep.channel],
        sampleRate,
        deep.frequency,
        deep.previewStartSeconds,
      );
      if (candidates.some((kept) => kept.channel === deep.channel &&
        Math.abs(kept.frequency - frequency) < 45)) continue;
      const candidate = { ...deep, frequency };
      const envelope = extractEnvelope(
        channels[candidate.channel],
        sampleRate,
        candidate.frequency,
        4.5,
        candidate.previewStartSeconds,
      );
      const optimized = optimizeEnvelope(envelope, options);
      passCount += optimized.passCount;
      if (!optimized.best || !isStrongDeepRecovery(optimized.best)) continue;
      recoveries.push({ candidate, hypothesis: optimized.best, envelope });
    }
    recoveries
      .sort((left, right) => right.hypothesis.baseScore - left.hypothesis.baseScore)
      .slice(0, 2)
      .forEach((recovery) => {
        candidates.push(recovery.candidate);
        previewResults.push(recovery);
      });
  }
  if (!previewResults.length && durationSeconds > 4.5) {
    // Some beacons and recordings open with a long steady carrier, calibration
    // tone, or silence. Preserve the normal first-window ranking exactly; only
    // when it produces no Morse hypothesis at all, retry a few later windows.
    // In each window the strongest physical candidate gets first refusal, so a
    // short noise track cannot win merely because many extra passes were run.
    const starts = [
      durationSeconds * 0.55,
      durationSeconds * 0.5,
      durationSeconds * 0.33,
      durationSeconds - 4.5,
    ]
      .map((start) => clamp(start, 0, Math.max(0, durationSeconds - 4.5)))
      .filter((start, index, values) =>
        start > 0.25 && values.findIndex((value) => Math.abs(value - start) < 0.1) === index
      );
    let recovered = false;
    for (const start of starts) {
      const lateCandidates = discoverCandidates(channels, sampleRate, options, start);
      for (const late of lateCandidates) {
        const frequency = refineFrequency(
          channels[late.channel],
          sampleRate,
          late.frequency,
          late.previewStartSeconds,
        );
        const candidate = { ...late, frequency };
        onProgress?.({
          phase: "candidates",
          fraction: 0.55,
          detail: `Retrying late carrier ${Math.round(candidate.frequency)} Hz`,
        });
        const envelope = extractEnvelope(
          channels[candidate.channel],
          sampleRate,
          candidate.frequency,
          4.5,
          candidate.previewStartSeconds,
        );
        const optimized = optimizeEnvelope(envelope, options, true);
        passCount += optimized.passCount;
        if (!optimized.best) continue;
        candidates.push(candidate);
        previewResults.push({ candidate, hypothesis: optimized.best, envelope });
        recovered = true;
        break;
      }
      if (recovered) break;
    }
  }
  if (!previewResults.length) throw new Error("No Morse-like keyed track survived timing analysis.");
  previewResults.sort((left, right) => right.hypothesis.score - left.hypothesis.score);
  const independentPreviewResults = dedupeTransparentChannelCopies(previewResults)
    .sort((left, right) => right.hypothesis.score - left.hypothesis.score);

  const finalistCount = durationSeconds <= 120 ? Math.min(4, independentPreviewResults.length) : 1;
  const finalists = independentPreviewResults.slice(0, finalistCount);
  if (durationSeconds > 120 && finalists.length) {
    // Long recordings normally refine one track for bounded runtime. If its
    // preview is a keyed harmonic, reserve the credible lower family member
    // for a full-file pass as well. Correlation only grants finalist status;
    // the ordinary full decode and physical scoring still choose the winner.
    const upper = finalists[0];
    const lowerFamily = independentPreviewResults
      .filter((lower) => {
        if (lower.candidate.channel !== upper.candidate.channel ||
          lower.candidate.frequency >= upper.candidate.frequency ||
          lower.hypothesis.validRatio < 0.9 || lower.hypothesis.timingQuality < 0.38 ||
          upper.hypothesis.baseScore - lower.hypothesis.baseScore > 0.07 ||
          Math.abs(lower.envelope.offsetMs - upper.envelope.offsetMs) > lower.envelope.hopMs * 2) {
          return false;
        }
        const ratio = upper.candidate.frequency / lower.candidate.frequency;
        const harmonic = Math.round(ratio);
        return harmonic >= 2 && harmonic <= 6 &&
          Math.abs(ratio - harmonic) / harmonic <= 0.01 &&
          maximumEnvelopeCorrelation(lower.envelope.levelsDb, upper.envelope.levelsDb) >= 0.68;
      })
      .sort((left, right) => left.candidate.frequency - right.candidate.frequency)[0];
    if (lowerFamily) finalists.push(lowerFamily);
  }
  const finalResults: FinalDecodeResult[] = [];
  const variantsByTrack = new Map<string, DecodeHypothesis[]>();
  for (let index = 0; index < finalists.length; index += 1) {
    const finalist = finalists[index];
    onProgress?.({
      phase: "refine",
      fraction: 0.58 + 0.36 * index / finalists.length,
      detail: `Refining pass ${index + 1} of ${finalists.length}`,
    });
    const envelope = extractEnvelope(
      channels[finalist.candidate.channel], sampleRate, finalist.candidate.frequency,
    );
    const optimized = optimizeEnvelope(
      envelope,
      options,
      Boolean(finalist.candidate.previewStartSeconds),
    );
    passCount += optimized.passCount;
    if (optimized.best) {
      finalResults.push({
        candidate: finalist.candidate,
        hypothesis: optimized.best,
        envelope,
        harmonicPenalty: 0,
      });
      variantsByTrack.set(
        `${finalist.candidate.channel}:${Math.round(finalist.candidate.frequency)}`,
        optimized.alternatives,
      );
    }
  }
  // A short preview can prefer a busy noise lane simply because it yields more
  // characters than a slow wanted station. Long recordings normally refine a
  // single carrier for bounded runtime, so detect only a catastrophic
  // full-file collapse and spend one extra pass verifying a much stronger,
  // independently timing-stable physical track. This deliberately excludes
  // language/message evidence and cannot affect the normal finalist budget.
  if (durationSeconds > 120 && finalists.length === 1 && finalResults.length === 1) {
    const initial = finalResults[0].hypothesis;
    const catastrophicCollapse = initial.score < 0 && initial.decoderScore < 0.35 &&
      (initial.validRatio < 0.75 || initial.timingQuality < 0.45);
    if (catastrophicCollapse) {
      const rescue = independentPreviewResults.find((preview) => {
        const alreadyRefined = finalists.some((finalist) =>
          finalist.candidate.channel === preview.candidate.channel &&
          Math.abs(finalist.candidate.frequency - preview.candidate.frequency) < 5
        );
        const characters = preview.hypothesis.tokens.filter((token) => !token.space).length;
        return !alreadyRefined && characters >= 3 && preview.hypothesis.score >= 0.4 &&
          preview.candidate.discoveryScore >= finalists[0].candidate.discoveryScore * 2 &&
          preview.hypothesis.decoderScore >= 0.45 && preview.hypothesis.validRatio >= 0.98 &&
          preview.hypothesis.timingQuality >= 0.8 && preview.hypothesis.contrastDb >= 12 &&
          preview.hypothesis.wpm >= 5 && preview.hypothesis.wpm <= 50;
      });
      if (rescue) {
        const envelope = extractEnvelope(
          channels[rescue.candidate.channel], sampleRate, rescue.candidate.frequency,
        );
        const optimized = optimizeEnvelope(
          envelope,
          options,
          Boolean(rescue.candidate.previewStartSeconds),
        );
        passCount += optimized.passCount;
        if (optimized.best) {
          finalResults.push({
            candidate: rescue.candidate,
            hypothesis: optimized.best,
            envelope,
            harmonicPenalty: 0,
          });
          variantsByTrack.set(
            `${rescue.candidate.channel}:${Math.round(rescue.candidate.frequency)}`,
            optimized.alternatives,
          );
        }
      }
    }
  }
  applyCorrelatedHarmonicPenalties(finalResults);
  finalResults.sort((left, right) =>
    right.hypothesis.score + structuredTrafficRankingAdjustment(right.hypothesis) - right.harmonicPenalty -
    (left.hypothesis.score + structuredTrafficRankingAdjustment(left.hypothesis) - left.harmonicPenalty)
  );
  const winner = finalResults[0] ?? independentPreviewResults[0];
  const winningFinal = finalResults[0];
  const fundamentalHz = winningFinal
    ? reportedHarmonicFundamental(winningFinal, finalResults)
    : undefined;
  if (winningFinal) {
    const segmented = optimizeTimingHandoffSegments(
      winningFinal.envelope,
      winningFinal.hypothesis,
      options,
      Boolean(winningFinal.candidate.previewStartSeconds),
    );
    passCount += segmented.passCount;
    if (segmented.hypothesis !== winningFinal.hypothesis) {
      const trackKey = `${winningFinal.candidate.channel}:${Math.round(winningFinal.candidate.frequency)}`;
      variantsByTrack.set(trackKey, [
        segmented.hypothesis,
        ...(variantsByTrack.get(trackKey) ?? []),
      ]);
      winningFinal.hypothesis = segmented.hypothesis;
    }
  }
  // The hard/segmented lane remains authoritative for station selection,
  // confidence, and convergence. A duration-state alternate may refine only
  // the copy of that same lane when measured QSB and every corpus-proven guard
  // agree; it can never manufacture a lock.
  const authoritativeHypothesis = winner.hypothesis;
  if (winningFinal) {
    const diagnostics = analyzeQsbEnvelope(winningFinal.envelope);
    if (qsbHmmEligibility(diagnostics, authoritativeHypothesis.timingQuality).eligible) {
      const qsbRuns = buildQsbHmmRuns(
        winningFinal.envelope,
        authoritativeHypothesis.wpm,
      );
      const optimized = optimizeEnvelope(
        winningFinal.envelope,
        { ...options, wpmHint: authoritativeHypothesis.wpm },
        Boolean(winningFinal.candidate.previewStartSeconds),
        qsbRuns,
      );
      passCount += optimized.passCount;
      if (optimized.best) {
        const laneKey = `${winningFinal.candidate.channel}:${Math.round(winningFinal.candidate.frequency)}`;
        const decision = selectQsbHmmAlternate(diagnostics, {
          laneKey,
          transcript: authoritativeHypothesis.transcript,
          baseScore: authoritativeHypothesis.baseScore,
          validRatio: authoritativeHypothesis.validRatio,
          wpm: authoritativeHypothesis.wpm,
          timingQuality: authoritativeHypothesis.timingQuality,
          converged: false,
        }, {
          laneKey,
          transcript: optimized.best.transcript,
          baseScore: optimized.best.baseScore,
          validRatio: optimized.best.validRatio,
          wpm: optimized.best.wpm,
        });
        if (decision.useAlternate) {
          variantsByTrack.set(laneKey, [
            optimized.best,
            authoritativeHypothesis,
            ...(variantsByTrack.get(laneKey) ?? []),
          ]);
          winningFinal.hypothesis = optimized.best;
        }
      }
    }
  }
  const winnerVariants = variantsByTrack.get(
    `${winner.candidate.channel}:${Math.round(winner.candidate.frequency)}`,
  ) ?? [winner.hypothesis];
  const harmonicFundamentals = new Map(finalResults.flatMap((result) => {
    const reported = reportedHarmonicFundamental(result, finalResults);
    return reported === undefined
      ? []
      : [[`${result.candidate.channel}:${Math.round(result.candidate.frequency)}`, reported] as const];
  }));
  const harmonicDuplicateKeys = new Set(finalResults
    // A similar envelope alone is not proof of a duplicate: two operators can
    // key at the same cadence on different receiver channels. Only candidates
    // that passed the same-channel integer-harmonic test receive a penalty and
    // may be hidden from the station-lane list.
    .filter((result) => result !== winningFinal && (
      result.harmonicPenalty > 0 || Boolean(
        winningFinal && areCorrelatedHarmonicTracks(winningFinal, result, true)
      )
    ))
    .map((result) => `${result.candidate.channel}:${Math.round(result.candidate.frequency)}`));
  const alternatives = [...finalResults, ...independentPreviewResults]
    .filter((item) => !harmonicDuplicateKeys.has(
      `${item.candidate.channel}:${Math.round(item.candidate.frequency)}`,
    ))
    .filter((item, index, values) => values.findIndex((other) =>
      other.candidate.channel === item.candidate.channel &&
      Math.abs(other.candidate.frequency - item.candidate.frequency) < 5,
    ) === index)
    .sort((left, right) => {
      const leftSelected = left.candidate.channel === winner.candidate.channel &&
        Math.abs(left.candidate.frequency - winner.candidate.frequency) < 5;
      const rightSelected = right.candidate.channel === winner.candidate.channel &&
        Math.abs(right.candidate.frequency - winner.candidate.frequency) < 5;
      return Number(rightSelected) - Number(leftSelected) ||
        right.hypothesis.score - left.hypothesis.score;
    })
    .slice(0, 4)
    .map(({ candidate, hypothesis }) => ({
      frequency: candidate.frequency,
      fundamentalHz: harmonicFundamentals.get(
        `${candidate.channel}:${Math.round(candidate.frequency)}`,
      ),
      channel: candidate.channel,
      score: hypothesis.score,
      wpm: hypothesis.wpm,
      contrastDb: hypothesis.contrastDb,
      validRatio: hypothesis.validRatio,
      timingQuality: hypothesis.timingQuality,
      languageQuality: hypothesis.languageQuality,
      messageQuality: hypothesis.messageQuality,
      hamEvidence: hypothesis.hamEvidence,
      decoderScore: hypothesis.decoderScore,
      transcript: hypothesis.transcript,
      thresholdDb: hypothesis.thresholdDb,
    }));
  const variants = winnerVariants.slice(0, 8).map((hypothesis) => ({
    frequency: winner.candidate.frequency,
    channel: winner.candidate.channel,
    score: hypothesis.score,
    wpm: hypothesis.wpm,
    contrastDb: hypothesis.contrastDb,
    validRatio: hypothesis.validRatio,
    timingQuality: hypothesis.timingQuality,
    languageQuality: hypothesis.languageQuality,
    messageQuality: hypothesis.messageQuality,
    hamEvidence: hypothesis.hamEvidence,
    decoderScore: hypothesis.decoderScore,
    transcript: hypothesis.transcript,
    thresholdDb: hypothesis.thresholdDb,
  }));
  // Confidence compares like with like. A 4.5-second preview can be an
  // excellent fragment of a short interferer while collapsing over the full
  // recording; letting it challenge a full-file winner made exact long copy
  // look ambiguous. Use fully refined tracks whenever one exists, and treat
  // receiver channels at the same frequency as one physical station lane.
  const confidencePool: Array<FinalDecodeResult | PreviewDecodeResult> = winningFinal
    ? finalResults
    : independentPreviewResults;
  const physicalPeers = confidencePool
    .filter((item) => !harmonicDuplicateKeys.has(
      `${item.candidate.channel}:${Math.round(item.candidate.frequency)}`,
    ))
    .filter((item, index, values) => values.findIndex((other) =>
      Math.abs(other.candidate.frequency - item.candidate.frequency) < 5,
    ) === index)
    .filter((item) => Math.abs(item.candidate.frequency - winner.candidate.frequency) >= 5)
    .filter((item) => !isSameCompleteCopy(authoritativeHypothesis, item.hypothesis))
    .sort((left, right) => right.hypothesis.baseScore - left.hypothesis.baseScore);
  const absoluteQuality = clamp((authoritativeHypothesis.baseScore - 0.55) / 0.35, 0, 1);
  // With no second full-file track, the absolute decoder/validity gates still
  // carry convergence; there is no comparable station lane to subtract.
  const marginQuality = physicalPeers.length
    ? clamp((authoritativeHypothesis.baseScore - physicalPeers[0].hypothesis.baseScore) / 0.18, 0, 1)
    : 1;
  const confidence = clamp(0.25 + 0.74 * (absoluteQuality * 0.7 + marginQuality * 0.3), 0.25, 0.99);
  // The legacy streaming decoder remains the safer choice for recordings that
  // do not produce a strong, valid consensus (wideband voice/noise, multiple
  // overlapping bands, or extremely degraded historical audio). File search is
  // an enhancement, not permission to replace a better live result with a weak
  // offline guess.
  const converged = authoritativeHypothesis.baseScore >= 0.75 &&
    authoritativeHypothesis.decoderScore >= 0.7 &&
    authoritativeHypothesis.validRatio >= 0.96 && confidence >= 0.78;
  const rawTranscript = winner.hypothesis.transcript;
  const tokens = winner.hypothesis.tokens.map((token) => {
    if (token.space) return token;
    const value = annotateMorseValue(token.pattern, token.value, options.languageAssist);
    return value === token.value ? token : { ...token, value, rawValue: token.value };
  });
  const transcript = tokens.map((token) => token.value).join("");
  onProgress?.({ phase: "complete", fraction: 1, detail: "Selected the strongest Morse hypothesis" });
  return {
    transcript,
    rawTranscript,
    tokens,
    frequency: winner.candidate.frequency,
    fundamentalHz,
    channel: winner.candidate.channel,
    wpm: winner.hypothesis.wpm,
    thresholdDb: winner.hypothesis.thresholdDb,
    score: winner.hypothesis.score,
    hamEvidence: winner.hypothesis.hamEvidence,
    confidence,
    converged,
    passCount,
    candidates: alternatives,
    variants,
  };
}
