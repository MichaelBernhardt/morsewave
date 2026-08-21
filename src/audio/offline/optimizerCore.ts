import { annotateMorseValue, decodeMorse, isMorseCharacter } from "../morse";
import { HAM_CW_TERMS, isHamCwScoringTerm } from "../hamVocabulary";
import { hamMessageEvidence } from "../hamInterpreter";

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
  frequency: number;
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
  const previewLength = Math.min(channels[0]?.length ?? 0, Math.round(sampleRate * 4.5));
  const all: SpectralCandidate[] = [];

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const samples = channels[channelIndex];
    const lows = new Float64Array(frequencies.length);
    const highs = new Float64Array(frequencies.length);
    const activity = new Float64Array(frequencies.length);
    const previousDb = new Float64Array(frequencies.length);
    let frames = 0;
    for (let start = 0; start + windowSize <= previewLength; start += hopSize) {
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
      });
    }
  }

  const separated: SpectralCandidate[] = [];
  for (const candidate of all.sort((left, right) => right.discoveryScore - left.discoveryScore)) {
    if (separated.some((kept) => kept.channel === candidate.channel &&
      Math.abs(kept.frequency - candidate.frequency) < 55)) continue;
    separated.push(candidate);
    if (separated.length >= Math.min(8, channels.length * 4)) break;
  }
  return separated;
}

function refineFrequency(
  samples: Float32Array,
  sampleRate: number,
  coarseFrequency: number,
) {
  const windowSize = Math.max(128, Math.round(sampleRate * 0.064));
  const window = makeHann(windowSize);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  const previewLength = Math.min(samples.length, Math.round(sampleRate * 4.5));
  const probeStarts: number[] = [];
  const probes = 18;
  for (let index = 0; index < probes; index += 1) {
    probeStarts.push(Math.max(0, Math.min(previewLength - windowSize,
      Math.round(index / Math.max(1, probes - 1) * Math.max(0, previewLength - windowSize)))));
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
): Envelope {
  const windowSize = Math.max(64, Math.round(sampleRate * 0.024));
  const hopSize = Math.max(8, Math.round(sampleRate * 0.002));
  const maximumLength = maximumSeconds
    ? Math.min(samples.length, Math.round(maximumSeconds * sampleRate))
    : samples.length;
  const frameCount = Math.max(0, Math.floor((maximumLength - windowSize) / hopSize) + 1);
  const levelsDb = new Float32Array(frameCount);
  const window = makeHann(windowSize);
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const magnitude = magnitudeAtFrame(
      samples, frame * hopSize, windowSize, frequency, sampleRate, window, windowSum,
    );
    levelsDb[frame] = Math.max(-120, 20 * Math.log10(magnitude + 1e-9));
  }
  const clusters = clusterLevels(levelsDb);
  return {
    levelsDb,
    hopMs: hopSize / sampleRate * 1000,
    offsetMs: windowSize / sampleRate * 500,
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

function fitUnit(runs: GateRun[], hintWpm: number) {
  const marks = runs.filter((run) => run.keyed).map((run) => run.endMs - run.startMs);
  const gaps = runs.filter((run, index) => !run.keyed && index > 0 && index < runs.length - 1)
    .map((run) => run.endMs - run.startMs)
    .filter((gap) => gap < 3000);
  const hintUnit = 1200 / clamp(hintWpm, 5, 60);
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
  const hysteresisValues = [1, 1.5, 2];
  const debounceValues = [4, 6, 10];
  const characterBoundaries = [1.75, 2.05, 2.4];
  // Real hand keys often collapse nominal 7-unit word spaces toward ordinary
  // character spacing. Search the ambiguous 3.5–5.5 unit region as well and
  // let timing plus language consensus decide instead of hard-wiring 7 units.
  const wordBoundaries = options.languageAssist
    ? [3.4, 3.8, 4.4, 5.2, 6.2, 7.4]
    : [5.8, 6.8, 7.8];

  const evaluateThresholdProfiles = (profiles: typeof standardThresholdProfiles) => {
    for (const { fraction: thresholdFraction, recoveryPenalty } of profiles) {
      const thresholdDb = envelope.lowDb + contrastDb * thresholdFraction;
      for (const hysteresisDb of hysteresisValues) {
        for (const debounceMs of debounceValues) {
          const runs = buildRuns(envelope, thresholdDb, hysteresisDb, debounceMs);
          const fitted = fitUnit(runs, options.wpmHint);
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
  if (standardBaseScore < 0.84 || standardValidRatio < 0.96) {
    evaluateThresholdProfiles(recoveryThresholdProfiles);
  }
  const alternatives = [...unique.values()].sort((left, right) => right.score - left.score).slice(0, 12);
  return {
    best: consensusHypothesis(alternatives, options.languageAssist) ?? best,
    passCount,
    alternatives,
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
    const frequency = refineFrequency(channels[candidate.channel], sampleRate, candidate.frequency);
    if (candidates.some((kept) => kept.channel === candidate.channel && Math.abs(kept.frequency - frequency) < 45)) continue;
    candidates.push({ ...candidate, frequency });
    if (candidates.length >= 6) break;
  }
  onProgress?.({ phase: "candidates", fraction: 0.22, detail: `Testing ${candidates.length} carrier tracks` });

  let passCount = 0;
  const previewResults: Array<{ candidate: SpectralCandidate; hypothesis: DecodeHypothesis }> = [];
  for (let index = 0; index < candidates.length; index += 1) {
    const candidate = candidates[index];
    const envelope = extractEnvelope(
      channels[candidate.channel], sampleRate, candidate.frequency, 4.5,
    );
    const optimized = optimizeEnvelope(envelope, options);
    passCount += optimized.passCount;
    if (optimized.best) previewResults.push({ candidate, hypothesis: optimized.best });
    onProgress?.({
      phase: "candidates",
      fraction: 0.22 + 0.33 * (index + 1) / candidates.length,
      detail: `Scored carrier ${index + 1} of ${candidates.length}`,
    });
  }
  if (!previewResults.length) throw new Error("No Morse-like keyed track survived timing analysis.");
  previewResults.sort((left, right) => right.hypothesis.score - left.hypothesis.score);

  const durationSeconds = channels[0].length / sampleRate;
  const finalistCount = durationSeconds <= 120 ? Math.min(4, previewResults.length) : 1;
  const finalResults: Array<{ candidate: SpectralCandidate; hypothesis: DecodeHypothesis }> = [];
  const variantsByTrack = new Map<string, DecodeHypothesis[]>();
  for (let index = 0; index < finalistCount; index += 1) {
    const finalist = previewResults[index];
    onProgress?.({
      phase: "refine",
      fraction: 0.58 + 0.36 * index / finalistCount,
      detail: `Refining pass ${index + 1} of ${finalistCount}`,
    });
    const envelope = extractEnvelope(
      channels[finalist.candidate.channel], sampleRate, finalist.candidate.frequency,
    );
    const optimized = optimizeEnvelope(envelope, options);
    passCount += optimized.passCount;
    if (optimized.best) {
      finalResults.push({ candidate: finalist.candidate, hypothesis: optimized.best });
      variantsByTrack.set(
        `${finalist.candidate.channel}:${Math.round(finalist.candidate.frequency)}`,
        optimized.alternatives,
      );
    }
  }
  finalResults.sort((left, right) => right.hypothesis.score - left.hypothesis.score);
  const winner = finalResults[0] ?? previewResults[0];
  const winnerVariants = variantsByTrack.get(
    `${winner.candidate.channel}:${Math.round(winner.candidate.frequency)}`,
  ) ?? [winner.hypothesis];
  const alternatives = [...finalResults, ...previewResults]
    .filter((item, index, values) => values.findIndex((other) =>
      other.candidate.channel === item.candidate.channel &&
      Math.abs(other.candidate.frequency - item.candidate.frequency) < 5,
    ) === index)
    .sort((left, right) => right.hypothesis.score - left.hypothesis.score)
    .slice(0, 4)
    .map(({ candidate, hypothesis }) => ({
      frequency: candidate.frequency,
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
  const physicalPeers = [...finalResults, ...previewResults]
    .filter((item, index, values) => values.findIndex((other) =>
      other.candidate.channel === item.candidate.channel &&
      Math.abs(other.candidate.frequency - item.candidate.frequency) < 5,
    ) === index)
    .filter((item) => item.candidate.channel !== winner.candidate.channel ||
      Math.abs(item.candidate.frequency - winner.candidate.frequency) >= 5)
    .sort((left, right) => right.hypothesis.baseScore - left.hypothesis.baseScore);
  const physicalRunnerUp = physicalPeers[0]?.hypothesis.baseScore ?? 0;
  const absoluteQuality = clamp((winner.hypothesis.baseScore - 0.55) / 0.35, 0, 1);
  const marginQuality = clamp((winner.hypothesis.baseScore - physicalRunnerUp) / 0.18, 0, 1);
  const confidence = clamp(0.25 + 0.74 * (absoluteQuality * 0.7 + marginQuality * 0.3), 0.25, 0.99);
  // The legacy streaming decoder remains the safer choice for recordings that
  // do not produce a strong, valid consensus (wideband voice/noise, multiple
  // overlapping bands, or extremely degraded historical audio). File search is
  // an enhancement, not permission to replace a better live result with a weak
  // offline guess.
  const converged = winner.hypothesis.baseScore >= 0.75 &&
    winner.hypothesis.decoderScore >= 0.7 &&
    winner.hypothesis.validRatio >= 0.96 && confidence >= 0.78;
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
