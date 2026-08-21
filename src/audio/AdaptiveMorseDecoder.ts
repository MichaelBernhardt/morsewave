import { decodeMorse, isMorseCharacter, isMorsePrefix } from "./morse";

export interface DecoderConfig {
  sensitivity: number;
  wpmHint: number;
  adaptiveSpeed: boolean;
  handKeyAssist?: boolean;
  languageAssist?: boolean;
}

export interface SignalSample {
  timestamp: number;
  levelDb: number;
  purity: number;
  frequency: number;
}

export interface DecodedCharacter {
  value: string;
  pattern: string;
  confidence: number;
  timestamp: number;
}

export interface DecoderMetrics {
  gateOpen: boolean;
  noiseFloorDb: number;
  thresholdDb: number;
  snrDb: number;
  wpm: number;
  currentPattern: string;
  confidence: number;
  timingJitter: number;
  recoveredEdges: number;
}

export interface DecoderCallbacks {
  onCharacter?: (character: DecodedCharacter) => void;
  onWordGap?: (timestamp: number) => void;
  onMetrics?: (metrics: DecoderMetrics) => void;
}

interface BeamCharacter {
  value: string;
  pattern: string;
  confidence: number;
}

interface BeamState {
  characters: BeamCharacter[];
  pattern: string;
  symbolConfidence: number[];
  score: number;
}

const clamp = (value: number, minimum: number, maximum: number) =>
  Math.min(maximum, Math.max(minimum, value));

const median = (values: number[]) => {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1] + ordered[middle]) / 2
    : ordered[middle];
};

/**
 * Stateful edge, timing, and symbol decoder. Input timestamps are milliseconds.
 * It deliberately has no Web Audio dependencies, making the critical timing path
 * deterministic and testable with recordings or synthetic signals.
 */
export class AdaptiveMorseDecoder {
  private config: DecoderConfig;
  private readonly callbacks: DecoderCallbacks;
  private noiseFloorDb = -78;
  private signalFloorDb = -42;
  private signalCalibrated = false;
  private unitMs: number;
  private gateOpen = false;
  private candidateGate: boolean | null = null;
  private candidateSince = 0;
  private markStartedAt: number | null = null;
  private gapStartedAt: number | null = null;
  private currentPattern = "";
  private currentConfidence: number[] = [];
  private markUnits: number[] = [];
  private lastTimestamp = 0;
  private characterFlushed = false;
  private wordFlushed = false;
  private hasOutput = false;
  private fuzzyMarks: number[] = [];
  private fuzzyGaps: number[] = [];
  private pendingFuzzyGap: number | null = null;
  private fuzzyEmittedCount = 0;
  private fuzzyPreviewPattern = "";
  private gapBeforeMarkStartedAt: number | null = null;
  private mergeDropoutGap: number | null = null;
  private timingJitter = 0;
  private recoveredEdges = 0;
  private signalRearmed = true;
  private quietCandidateSince: number | null = null;

  constructor(config: DecoderConfig, callbacks: DecoderCallbacks = {}) {
    this.config = { ...config };
    this.callbacks = callbacks;
    this.unitMs = 1200 / clamp(config.wpmHint, 5, 60);
  }

  updateConfig(config: Partial<DecoderConfig>) {
    const previousHint = this.config.wpmHint;
    const previousHandKeyAssist = this.config.handKeyAssist;
    this.config = { ...this.config, ...config };
    if (config.handKeyAssist !== undefined && config.handKeyAssist !== previousHandKeyAssist) {
      this.reset(true);
    }
    if (!this.config.adaptiveSpeed || config.wpmHint !== undefined && config.wpmHint !== previousHint) {
      this.unitMs = 1200 / clamp(this.config.wpmHint, 5, 60);
      this.markUnits = [];
    }
  }

  reset(keepCalibration = false) {
    this.gateOpen = false;
    this.candidateGate = null;
    this.markStartedAt = null;
    this.gapStartedAt = null;
    this.currentPattern = "";
    this.currentConfidence = [];
    this.lastTimestamp = 0;
    this.characterFlushed = false;
    this.wordFlushed = false;
    this.hasOutput = false;
    this.fuzzyMarks = [];
    this.fuzzyGaps = [];
    this.pendingFuzzyGap = null;
    this.fuzzyEmittedCount = 0;
    this.fuzzyPreviewPattern = "";
    this.gapBeforeMarkStartedAt = null;
    this.mergeDropoutGap = null;
    this.timingJitter = 0;
    this.recoveredEdges = 0;
    this.signalRearmed = true;
    this.quietCandidateSince = null;
    if (!keepCalibration) {
      this.noiseFloorDb = -78;
      this.signalFloorDb = -42;
      this.signalCalibrated = false;
      this.unitMs = 1200 / clamp(this.config.wpmHint, 5, 60);
      this.markUnits = [];
    }
  }

  process(sample: SignalSample): DecoderMetrics {
    const timestamp = Math.max(this.lastTimestamp, sample.timestamp);
    this.lastTimestamp = timestamp;
    const levelDb = clamp(Number.isFinite(sample.levelDb) ? sample.levelDb : -120, -120, 6);
    const purity = clamp(Number.isFinite(sample.purity) ? sample.purity : 0, 0, 1);

    const sensitivity = clamp(this.config.sensitivity, 0, 100);
    const requiredSnr = 11 - sensitivity * 0.075;
    const requiredPurity = 0.32 - sensitivity * 0.0024;
    const separation = this.signalCalibrated ? Math.max(0, this.signalFloorDb - this.noiseFloorDb) : 0;
    // Use both sides of the observed level distribution. A noise-relative gate
    // alone becomes far too permissive on digitally quiet recordings: codec
    // ringing tens of dB below a full-scale CW mark is still a highly pure tone.
    // The signal-relative limb rejects that ringing, while weak signals (where
    // there is little separation) naturally fall back to the noise/SNR limb.
    const signalHeadroom = 6 + sensitivity * 0.055;
    const noiseThreshold = this.noiseFloorDb + requiredSnr;
    const signalThreshold = this.signalCalibrated && separation > requiredSnr + 2
      ? this.signalFloorDb - signalHeadroom
      : -120;
    const thresholdDb = Math.max(noiseThreshold, signalThreshold);
    const hysteresis = this.gateOpen ? -2.2 : 0.8;
    const purityFloor = this.gateOpen ? requiredPurity * 0.62 : requiredPurity;
    // A confirmed key-up interval arms a fresh signal-level acquisition. This
    // lets the decoder follow abrupt AGC changes or deep QSB without weakening
    // the signal-relative guard that rejects continuous tonal codec residue.
    // Pure low-level residue never supplies the quiet interval needed to arm.
    const quietEvidence = levelDb <= this.noiseFloorDb + Math.max(2.5, requiredSnr * 0.55) ||
      purity < requiredPurity * 0.5;
    if (!this.gateOpen && quietEvidence) {
      if (this.quietCandidateSince === null) this.quietCandidateSince = timestamp;
      const quietConfirmationMs = clamp(this.unitMs * 0.14, 10, 24);
      if (timestamp - this.quietCandidateSince >= quietConfirmationMs) this.signalRearmed = true;
    } else if (!this.gateOpen && !this.signalRearmed) {
      this.quietCandidateSince = null;
    }
    const normalGate = levelDb >= thresholdDb + hysteresis && purity >= purityFloor;
    const reacquiredGate = this.signalRearmed &&
      levelDb >= noiseThreshold + hysteresis && purity >= purityFloor;
    const rawGate = normalGate || reacquiredGate;

    if (!this.gateOpen) {
      const noiseAlpha = levelDb < this.noiseFloorDb ? 0.055 : 0.0018;
      // Highly tonal frames are probably signal candidates, not background noise.
      if (purity < requiredPurity * 1.15 || levelDb < thresholdDb) {
        this.noiseFloorDb += (levelDb - this.noiseFloorDb) * noiseAlpha;
      }
    } else {
      if (!this.signalCalibrated) {
        this.signalFloorDb = levelDb;
        this.signalCalibrated = true;
      } else {
        // Fast attack reaches the body of a shaped mark before its first falling
        // edge; slow release follows QSB without learning transition tails.
        const signalAlpha = levelDb > this.signalFloorDb ? 0.22 : 0.006;
        this.signalFloorDb += (levelDb - this.signalFloorDb) * signalAlpha;
      }
    }

    if (rawGate === this.gateOpen) {
      this.candidateGate = null;
    } else if (this.candidateGate !== rawGate) {
      this.candidateGate = rawGate;
      this.candidateSince = timestamp;
    } else {
      const debounceMs = rawGate
        ? (this.config.handKeyAssist ? 6 : 5)
        : (this.config.handKeyAssist ? clamp(this.unitMs * 0.2, 10, 28) : 8);
      if (timestamp - this.candidateSince >= debounceMs) {
        this.commitEdge(rawGate, this.candidateSince, rawGate && this.signalRearmed ? levelDb : undefined);
        this.candidateGate = null;
      }
    }

    if (!this.gateOpen) this.flushGap(timestamp);

    const metrics = this.makeMetrics(levelDb, thresholdDb);
    this.callbacks.onMetrics?.(metrics);
    return metrics;
  }

  flush(timestamp = this.lastTimestamp + this.unitMs * 7) {
    if (this.gateOpen && this.markStartedAt !== null) {
      this.commitEdge(false, timestamp);
    }
    if (this.config.handKeyAssist) {
      this.finalizeFuzzyWord(timestamp, false);
    } else {
      this.flushGap(timestamp + this.unitMs * 7);
    }
  }

  getMetrics(): DecoderMetrics {
    return this.makeMetrics(this.noiseFloorDb, this.noiseFloorDb + 6);
  }

  private commitEdge(open: boolean, timestamp: number, reacquiredLevelDb?: number) {
    if (open === this.gateOpen) return;
    if (open) {
      if (reacquiredLevelDb !== undefined) {
        this.signalFloorDb = reacquiredLevelDb;
        this.signalCalibrated = true;
      }
      this.signalRearmed = false;
      this.quietCandidateSince = null;
      if (this.config.handKeyAssist) {
        const gap = this.gapStartedAt === null ? null : Math.max(0, timestamp - this.gapStartedAt);
        if (gap !== null && this.fuzzyMarks.length > 0) {
          // Only bridge genuinely sub-element outages. Capping this in real
          // time prevents a stale slow speed hint from merging legitimate
          // element gaps when a 40–50 WPM sender first appears.
          if (gap < Math.min(this.unitMs * 0.36, 15)) {
            this.mergeDropoutGap = gap;
            this.pendingFuzzyGap = null;
          } else {
            this.pendingFuzzyGap = gap;
          }
        }
      } else {
        this.flushGap(timestamp);
      }
      this.gateOpen = true;
      this.gapBeforeMarkStartedAt = this.gapStartedAt;
      this.markStartedAt = timestamp;
      this.gapStartedAt = null;
      this.characterFlushed = false;
      this.wordFlushed = false;
      return;
    }

    this.gateOpen = false;
    if (this.markStartedAt === null) return;
    const duration = Math.max(0, timestamp - this.markStartedAt);
    this.markStartedAt = null;
    this.gapStartedAt = timestamp;
    this.characterFlushed = false;
    this.wordFlushed = false;
    if (this.config.handKeyAssist) {
      const accepted = this.consumeFuzzyMark(duration);
      if (!accepted && this.gapBeforeMarkStartedAt !== null) {
        this.gapStartedAt = this.gapBeforeMarkStartedAt;
        this.pendingFuzzyGap = null;
        this.mergeDropoutGap = null;
        this.recoveredEdges += 1;
      }
    } else this.consumeMark(duration);
    this.gapBeforeMarkStartedAt = null;
  }

  private consumeMark(duration: number) {
    if (duration < Math.max(7, this.unitMs * 0.24)) return;

    const ratio = duration / this.unitMs;
    // Hand-sent dits often run long and dahs often run short. The forgiving
    // boundary is the midpoint of those common novice timing envelopes.
    const markBoundary = this.config.handKeyAssist ? 1.68 : 2.05;
    const isDash = ratio >= markBoundary;
    const symbol = isDash ? "-" : ".";
    const expected = isDash ? 3 : 1;
    const timingError = Math.abs(ratio - expected) / (isDash ? 2.4 : 1.25);
    const confidence = clamp(1 - timingError, 0.2, 1);
    this.currentPattern += symbol;
    this.currentConfidence.push(confidence);

    if (this.config.adaptiveSpeed) {
      const candidateUnit = duration / expected;
      if (candidateUnit >= this.unitMs * 0.38 && candidateUnit <= this.unitMs * 2.4) {
        this.markUnits.push(candidateUnit);
        if (this.markUnits.length > 16) this.markUnits.shift();
        const target = median(this.markUnits);
        const alpha = this.markUnits.length < 4 ? 0.34 : 0.16;
        this.unitMs += (target - this.unitMs) * alpha;
        this.unitMs = clamp(this.unitMs, 20, 240);
      }
    }

    // More than eight elements is not a valid International Morse character.
    if (this.currentPattern.length > 8) {
      this.emitCharacter(this.lastTimestamp);
    }
  }

  private flushGap(timestamp: number) {
    if (this.gapStartedAt === null) return;
    const gap = timestamp - this.gapStartedAt;

    if (this.config.handKeyAssist) {
      // A very long pause is an unambiguous transmission break. Ordinary
      // character and word gaps are retained for beam-search segmentation.
      if (!this.wordFlushed && this.fuzzyMarks.length > 0 && gap >= this.unitMs * 11) {
        this.finalizeFuzzyWord(timestamp, true);
        this.wordFlushed = true;
      }
      return;
    }

    // Midpoints between the canonical 1/3/7-unit gaps preserve boundaries
    // while still tolerating hand-keyed spacing and abrupt speed changes.
    const characterBoundary = this.config.handKeyAssist ? 1.68 : 2.05;
    if (!this.characterFlushed && this.currentPattern && gap >= this.unitMs * characterBoundary) {
      this.emitCharacter(this.gapStartedAt + this.unitMs * 3);
      this.characterFlushed = true;
    }

    const wordBoundary = this.config.handKeyAssist ? 5.2 : 5.1;
    if (!this.wordFlushed && this.characterFlushed && this.hasOutput && gap >= this.unitMs * wordBoundary) {
      this.callbacks.onWordGap?.(this.gapStartedAt + this.unitMs * 7);
      this.wordFlushed = true;
    }
  }

  private emitCharacter(timestamp: number) {
    if (!this.currentPattern) return;
    const confidence = this.currentConfidence.length
      ? this.currentConfidence.reduce((sum, value) => sum + value, 0) / this.currentConfidence.length
      : 0;
    this.callbacks.onCharacter?.({
      value: decodeMorse(this.currentPattern),
      pattern: this.currentPattern,
      confidence,
      timestamp,
    });
    this.currentPattern = "";
    this.currentConfidence = [];
    this.hasOutput = true;
  }

  private consumeFuzzyMark(duration: number) {
    if (duration < Math.max(9, this.unitMs * 0.22)) return false;
    // A stuck carrier, tune-up tone, or heterodyne is not a Morse mark. Without
    // an upper bound it is eventually forced into a dash and emitted as a false
    // character when the carrier disappears.
    if (duration > Math.max(1200, this.unitMs * 8)) return false;
    if (this.mergeDropoutGap !== null && this.fuzzyMarks.length > 0) {
      this.fuzzyMarks[this.fuzzyMarks.length - 1] += this.mergeDropoutGap + duration;
      this.mergeDropoutGap = null;
      this.pendingFuzzyGap = null;
      this.recoveredEdges += 1;
      this.updateFuzzyUnitEstimate();
      const states = this.parseFuzzyWord(false);
      this.fuzzyPreviewPattern = states[0]?.pattern ?? "";
      this.emitStableFuzzyPrefix(states, this.lastTimestamp);
      return true;
    }
    if (this.pendingFuzzyGap !== null && this.fuzzyMarks.length > 0) {
      this.fuzzyGaps.push(this.pendingFuzzyGap);
    }
    this.pendingFuzzyGap = null;
    this.fuzzyMarks.push(duration);
    if (this.fuzzyMarks.length > 96) {
      // Pathological runs are finalized in bounded chunks so memory and beam
      // search remain deterministic even with a stuck carrier.
      this.finalizeFuzzyWord(this.lastTimestamp, false);
      return true;
    }
    this.updateFuzzyUnitEstimate();
    const states = this.parseFuzzyWord(false);
    this.fuzzyPreviewPattern = states[0]?.pattern ?? "";
    this.emitStableFuzzyPrefix(states, this.lastTimestamp);
    return true;
  }

  private updateFuzzyUnitEstimate() {
    if (this.fuzzyMarks.length < 2) return;
    const hintUnit = 1200 / clamp(this.config.wpmHint, 5, 60);
    let bestUnit = this.unitMs;
    let bestScore = Number.POSITIVE_INFINITY;
    const loss = (error: number, spread: number) => Math.min(9, (error / spread) ** 2);
    const observations = this.fuzzyMarks.length + this.fuzzyGaps.length;
    const coefficientOfVariation = (values: number[]) => {
      const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
      const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
        Math.max(1, values.length);
      return { mean, cv: Math.sqrt(variance) / Math.max(1, mean) };
    };
    let markClusterAnchor: number | null = null;
    if (this.fuzzyMarks.length >= 12) {
      const orderedMarks = [...this.fuzzyMarks].sort((left, right) => left - right);
      let bestSplit: { left: number[]; right: number[]; loss: number } | null = null;
      for (let split = 2; split <= orderedMarks.length - 2; split += 1) {
        const left = orderedMarks.slice(0, split);
        const right = orderedMarks.slice(split);
        const leftStats = coefficientOfVariation(left);
        const rightStats = coefficientOfVariation(right);
        const splitLoss = left.reduce((sum, value) => sum + (value - leftStats.mean) ** 2, 0) +
          right.reduce((sum, value) => sum + (value - rightStats.mean) ** 2, 0);
        if (!bestSplit || splitLoss < bestSplit.loss) bestSplit = { left, right, loss: splitLoss };
      }
      if (bestSplit) {
        const short = coefficientOfVariation(bestSplit.left);
        const long = coefficientOfVariation(bestSplit.right);
        if (long.mean / Math.max(1, short.mean) >= 1.25 && short.cv < 0.18 && long.cv < 0.18) {
          // Poor straight-key operators commonly stretch dots and crush dashes.
          // The two observed mark clusters still reveal the underlying dit much
          // more reliably than assuming either class is canonical by itself.
          markClusterAnchor = (short.mean + long.mean / 3) / 2;
        }
      }
    }
    const markStats = coefficientOfVariation(this.fuzzyMarks);
    const gapStats = coefficientOfVariation(this.fuzzyGaps);
    const singleClassHarmonic = this.fuzzyMarks.length >= 3 && this.fuzzyGaps.length >= 2 &&
      markStats.cv < 0.12 && gapStats.cv < 0.15 &&
      gapStats.mean / Math.max(1, markStats.mean) >= 0.65 &&
      gapStats.mean / Math.max(1, markStats.mean) <= 1.5;
    const hintWeight = singleClassHarmonic ? observations * 0.2 : 0.035;

    for (let candidate = 20; candidate <= 240; candidate += 1) {
      let score = Math.abs(Math.log(candidate / this.unitMs)) * 0.18;
      // With only dashes (or only dots), the observed timings are harmonically
      // ambiguous: three 200 ms dashes can also look like three 200 ms dots at
      // one-third the speed. Keep enough prior on the operator's speed hint to
      // resolve that ambiguity until mixed marks provide stronger evidence.
      score += Math.abs(Math.log(candidate / hintUnit)) * hintWeight;
      if (markClusterAnchor !== null) {
        score += observations * 2 * Math.abs(Math.log(candidate / markClusterAnchor));
      }
      for (const mark of this.fuzzyMarks) {
        const dot = loss(mark - candidate, candidate * 0.48);
        const dash = loss(mark - candidate * 3, candidate * 1.18) + 0.08;
        score += Math.min(dot, dash);
      }
      for (const gap of this.fuzzyGaps) {
        const element = loss(gap - candidate, candidate * 0.52);
        const character = loss(gap - candidate * 3, candidate * 1.22) + 0.22;
        const word = loss(gap - candidate * 7, candidate * 2.4) + 0.25;
        score += Math.min(element, character, word);
      }
      if (score < bestScore) {
        bestScore = score;
        bestUnit = candidate;
      }
    }

    const alpha = observations < 5 ? 0.3 : 0.58;
    this.unitMs += (bestUnit - this.unitMs) * alpha;
    this.unitMs = clamp(this.unitMs, 20, 240);

    const deviations = [
      ...this.fuzzyMarks.map((mark) => Math.min(
        Math.abs(mark - this.unitMs) / this.unitMs,
        Math.abs(mark - this.unitMs * 3) / (this.unitMs * 3),
      )),
      ...this.fuzzyGaps.map((gap) => Math.min(
        Math.abs(gap - this.unitMs) / this.unitMs,
        Math.abs(gap - this.unitMs * 3) / (this.unitMs * 3),
        Math.abs(gap - this.unitMs * 7) / (this.unitMs * 7),
      )),
    ];
    this.timingJitter = clamp(median(deviations) * 100, 0, 99);
  }

  private parseFuzzyWord(finalize: boolean): BeamState[] {
    if (this.fuzzyMarks.length === 0) return [];
    const timingLoss = (error: number, spread: number) => Math.min(10, (error / spread) ** 2);
    const symbolOptions = (duration: number) => {
      const dotCost = timingLoss(duration - this.unitMs, this.unitMs * 0.48);
      const dashCost = timingLoss(duration - this.unitMs * 3, this.unitMs * 1.2);
      const margin = Math.abs(dotCost - dashCost);
      return [
        { symbol: ".", cost: dotCost, confidence: clamp(0.5 + margin * 0.1, 0.5, 0.98) },
        { symbol: "-", cost: dashCost, confidence: clamp(0.5 + margin * 0.1, 0.5, 0.98) },
      ];
    };
    const characterPenalty = (value: string) => {
      if ("ETAOINSHRDLU".includes(value)) return 0;
      if (/^[A-Z0-9]$/.test(value)) return 0.09;
      if (value.startsWith("<")) return 0.2;
      return 0.14;
    };
    const commonBigrams = new Set(
      "TH HE IN ER AN RE ON AT EN ND TI ES OR TE OF ED IS IT AL AR ST TO NT NG SE HA AS OU IO LE VE CO ME DE HI RI RO IC NE EA RA CE LI CH LL BE MA SI OM UR WI WA AV EG GI NN FI BA AD RS MO CW CQ QR QL QU BR OW FO OX LA AZ KE EY LP PY".split(" "),
    );
    const commonWords = new Set(
      "A I THE AND OF TO IN IS IT THIS THAT FOR FROM WITH ON AT BY AS BE OR ARE WAS WERE YOU YOUR MY ME WE OUR THEY HE SHE NAME CALL CALLING RADIO CW MORSE CODE SIGNAL COPY AGAIN PLEASE PSE THANKS TNX GOOD BAD FIST TEST NOW WPM TRANSITION FILE FOLLOWS WEATHER ANTENNA POWER RIG STATION OPERATOR MORNING AFTERNOON EVENING HELLO HI".split(" "),
    );
    const currentWord = (characters: BeamCharacter[], trailingValue = "") => {
      const values = characters.map((character) => character.value);
      const boundary = values.lastIndexOf(" ");
      return values.slice(boundary + 1).join("") + trailingValue;
    };
    const previousWord = (characters: BeamCharacter[]) => {
      const words = characters.map((character) => character.value).join("").trim().split(/\s+/);
      return words.length > 1 ? words.at(-2) ?? "" : "";
    };
    const sequencePenalty = (value: string, characters: BeamCharacter[]) => {
      if (!this.config.languageAssist) return 0;
      const previous = characters.at(-1);
      if (!previous || previous.value === " ") return 0;
      if (!/^[A-Z]$/.test(previous.value) || !/^[A-Z]$/.test(value)) return 0;
      // Callsigns are intentionally not forced through an English model.
      if (/\d/.test(currentWord(characters))) return 0;
      return commonBigrams.has(previous.value + value) ? 0 : 0.85;
    };
    const wordBoundaryPenalty = (value: string, characters: BeamCharacter[]) => {
      if (!this.config.languageAssist) return 0;
      const word = currentWord(characters, value);
      if (/^(CQ|DE|QRZ|QSO|QTH|RST|TEST|K|KN|AR|SK|R)$/.test(word)) return -1.35;
      if (/^[A-Z]{1,2}\d[A-Z0-9]{1,4}(?:\/[A-Z0-9]+)?$/.test(word)) return -0.35;
      if (commonWords.has(word)) return -0.85;
      if (word.length === 1) return /^[AIKR]$/.test(word) ? 0.15 : 0.85;
      if (/\d/.test(word)) return 0.1;
      return word.length === 2 ? 0.12 : 0;
    };
    let beam: BeamState[] = [];

    for (let index = 0; index < this.fuzzyMarks.length; index += 1) {
      const options = symbolOptions(this.fuzzyMarks[index]);
      if (index === 0) {
        beam = options.map((option) => ({
          characters: [],
          pattern: option.symbol,
          symbolConfidence: [option.confidence],
          score: option.cost,
        }));
        continue;
      }

      const gap = this.fuzzyGaps[index - 1] ?? this.unitMs;
      const elementCost = timingLoss(gap - this.unitMs, this.unitMs * 0.55);
      const characterCost = timingLoss(gap - this.unitMs * 3, this.unitMs * 1.25) + 0.12;
      const wordCost = timingLoss(gap - this.unitMs * 7, this.unitMs * 2.4) + 0.15;
      const next: BeamState[] = [];

      for (const state of beam) {
        for (const option of options) {
          const joinedPattern = state.pattern + option.symbol;
          if (joinedPattern.length <= 8 && isMorsePrefix(joinedPattern)) {
            next.push({
              characters: state.characters,
              pattern: joinedPattern,
              symbolConfidence: [...state.symbolConfidence, option.confidence],
              score: state.score + option.cost + elementCost,
            });
          }
          // A mark that fades fully below the gate leaves an off interval of
          // roughly 1 + mark + 1 units. Keep dit/dah insertion hypotheses in
          // the beam, with an explicit penalty and visibly low confidence.
          // Normal character boundaries still win unless later Morse/context
          // evidence makes the dropout interpretation materially better.
          if (gap >= this.unitMs * 1.8) {
            const missingOptions = [
              { symbol: ".", cost: timingLoss(gap - this.unitMs * 3, this.unitMs * 0.95) + 2.1 },
              { symbol: "-", cost: timingLoss(gap - this.unitMs * 5, this.unitMs * 1.45) + 2.75 },
            ];
            for (const missing of missingOptions) {
              const repairedPattern = state.pattern + missing.symbol + option.symbol;
              if (repairedPattern.length <= 8 && isMorsePrefix(repairedPattern)) {
                const repairedValue = isMorseCharacter(repairedPattern) ? decodeMorse(repairedPattern) : "";
                const expectedCallsignDigit = previousWord(state.characters) === "DE" &&
                  /^[A-Z]{1,2}$/.test(currentWord(state.characters)) && /^\d$/.test(repairedValue);
                next.push({
                  characters: state.characters,
                  pattern: repairedPattern,
                  symbolConfidence: [...state.symbolConfidence, 0.28, option.confidence],
                  score: state.score + option.cost + missing.cost - (expectedCallsignDigit ? 1.65 : 0),
                });
              }
            }
          }
          if (isMorseCharacter(state.pattern)) {
            const value = decodeMorse(state.pattern);
            const transitionCost = sequencePenalty(value, state.characters);
            const confidence = state.symbolConfidence.reduce((sum, item) => sum + item, 0) /
              Math.max(1, state.symbolConfidence.length);
            next.push({
              characters: [...state.characters, { value, pattern: state.pattern, confidence }],
              pattern: option.symbol,
              symbolConfidence: [option.confidence],
              score: state.score + option.cost + characterCost + characterPenalty(value) + transitionCost,
            });
            next.push({
              characters: [
                ...state.characters,
                { value, pattern: state.pattern, confidence },
                { value: " ", pattern: "/", confidence: clamp(1 - wordCost / 8, 0.45, 0.98) },
              ],
              pattern: option.symbol,
              symbolConfidence: [option.confidence],
              score: state.score + option.cost + wordCost + characterPenalty(value) + transitionCost +
                wordBoundaryPenalty(value, state.characters),
            });
          }
        }
      }

      const unique = new Map<string, BeamState>();
      for (const state of next.sort((left, right) => left.score - right.score)) {
        const key = `${state.characters.map((item) => item.pattern).join("/")}|${state.pattern}`;
        if (!unique.has(key)) unique.set(key, state);
        if (unique.size >= 64) break;
      }
      beam = [...unique.values()];
    }

    if (finalize) {
      beam = beam.map((state) => {
        if (!isMorseCharacter(state.pattern)) {
          return {
            ...state,
            characters: [...state.characters, { value: "�", pattern: state.pattern, confidence: 0.25 }],
            pattern: "",
            symbolConfidence: [],
            score: state.score + 4,
          };
        }
        const value = decodeMorse(state.pattern);
        const transitionCost = sequencePenalty(value, state.characters);
        const confidence = state.symbolConfidence.reduce((sum, item) => sum + item, 0) /
          Math.max(1, state.symbolConfidence.length);
        return {
          ...state,
          characters: [...state.characters, { value, pattern: state.pattern, confidence }],
          pattern: "",
          symbolConfidence: [],
          score: state.score + characterPenalty(value) + transitionCost,
        };
      });
    }
    return beam.sort((left, right) => left.score - right.score).slice(0, 16);
  }

  private emitStableFuzzyPrefix(states: BeamState[], timestamp: number) {
    if (states.length === 0) return;
    // Keep a deliberately wide ambiguity band for poor fists. Emitting a
    // character slightly later is preferable to committing an irreversible
    // split before the following timing evidence arrives.
    const credible = states.filter((state) => state.score <= states[0].score + 6).slice(0, 16);
    let common = credible[0]?.characters.length ?? 0;
    for (let index = 0; index < common; index += 1) {
      const pattern = credible[0].characters[index]?.pattern;
      if (!credible.every((state) => state.characters[index]?.pattern === pattern)) {
        common = index;
        break;
      }
    }
    const best = states[0];
    while (this.fuzzyEmittedCount < common) {
      const character = best.characters[this.fuzzyEmittedCount];
      if (character.value === " ") this.callbacks.onWordGap?.(timestamp);
      else this.callbacks.onCharacter?.({ ...character, timestamp });
      this.fuzzyEmittedCount += 1;
      this.hasOutput = true;
    }
  }

  private finalizeFuzzyWord(timestamp: number, emitWordGap: boolean) {
    if (this.fuzzyMarks.length === 0) return;
    this.updateFuzzyUnitEstimate();
    const states = this.parseFuzzyWord(true);
    const best = states[0];
    if (best) {
      while (this.fuzzyEmittedCount < best.characters.length) {
        const character = best.characters[this.fuzzyEmittedCount];
        if (character.value === " ") this.callbacks.onWordGap?.(timestamp);
        else this.callbacks.onCharacter?.({ ...character, timestamp });
        this.fuzzyEmittedCount += 1;
        this.hasOutput = true;
      }
    }
    if (emitWordGap && this.hasOutput && best?.characters.at(-1)?.value !== " ") {
      this.callbacks.onWordGap?.(timestamp);
    }
    this.fuzzyMarks = [];
    this.fuzzyGaps = [];
    this.pendingFuzzyGap = null;
    this.fuzzyEmittedCount = 0;
    this.fuzzyPreviewPattern = "";
    this.mergeDropoutGap = null;
  }

  private makeMetrics(levelDb: number, thresholdDb: number): DecoderMetrics {
    const confidence = this.gateOpen
      ? clamp((levelDb - this.noiseFloorDb - 2) / 18, 0, 1)
      : this.config.handKeyAssist && this.fuzzyMarks.length
        ? clamp(1 - this.timingJitter / 75, 0.2, 1)
        : this.currentConfidence.length
        ? this.currentConfidence.reduce((sum, value) => sum + value, 0) / this.currentConfidence.length
        : 0;
    return {
      gateOpen: this.gateOpen,
      noiseFloorDb: this.noiseFloorDb,
      thresholdDb,
      snrDb: Math.max(0, levelDb - this.noiseFloorDb),
      wpm: 1200 / this.unitMs,
      currentPattern: this.config.handKeyAssist ? this.fuzzyPreviewPattern : this.currentPattern,
      confidence,
      timingJitter: this.timingJitter,
      recoveredEdges: this.recoveredEdges,
    };
  }
}
