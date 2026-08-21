export const QSB_HMM_POLICY = Object.freeze({
  diagnosticStepMs: 250,
  diagnosticRadiusMs: 1_800,
  minimumTimingQuality: 0.7,
  minimumKeyedClusterStdDb: 3,
  frameMs: 8,
  emissionCenter: 0.7,
  emissionGain: 6,
  minimumBaseScoreGain: 0.008,
  maximumValidRatioDrop: 0.01,
  maximumWpmDriftRatio: 0.15,
  minimumWpm: 5,
  maximumWpm: 60,
} as const);

const COMPARISON_EPSILON = 1e-12;

export interface QsbEnvelope {
  levelsDb: Float32Array;
  hopMs: number;
  offsetMs: number;
  lowDb: number;
  highDb: number;
}

export interface QsbGateRun {
  keyed: boolean;
  startMs: number;
  endMs: number;
}

export interface QsbEnvelopeDiagnostics {
  keyedClusterStdDb: number;
  windowCount: number;
}

export type QsbEligibilityBlocker = "weak-timing" | "insufficient-qsb" | "invalid-metrics";

export interface QsbHmmEligibility {
  eligible: boolean;
  blockers: QsbEligibilityBlocker[];
}

export interface QsbHardCandidateMetrics {
  laneKey: string;
  transcript: string;
  baseScore: number;
  validRatio: number;
  wpm: number;
  timingQuality: number;
  converged: boolean;
}

export interface QsbAlternateCandidateMetrics {
  laneKey: string;
  transcript: string;
  baseScore: number;
  validRatio: number;
  wpm: number;
}

export type QsbSelectionBlocker =
  | QsbEligibilityBlocker
  | "different-lane"
  | "same-copy"
  | "insufficient-score-gain"
  | "validity-drop"
  | "wpm-drift";

export interface QsbHmmSelection {
  useAlternate: boolean;
  converged: boolean;
  blockers: QsbSelectionBlocker[];
  baseScoreGain: number;
  validRatioDrop: number;
  wpmDriftRatio: number;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function assertEnvelope(envelope: QsbEnvelope) {
  if (!Number.isFinite(envelope.hopMs) || envelope.hopMs <= 0 ||
    !Number.isFinite(envelope.offsetMs) || !Number.isFinite(envelope.lowDb) ||
    !Number.isFinite(envelope.highDb) || envelope.highDb < envelope.lowDb) {
    throw new Error("QSB HMM requires a finite envelope and a positive hop size.");
  }
}

function clusterLevels(levels: Float32Array) {
  const finite: number[] = [];
  for (const level of levels) {
    if (Number.isFinite(level)) finite.push(level);
  }
  finite.sort((left, right) => left - right);
  if (!finite.length) return { lowDb: -90, highDb: -30 };

  let lowDb = finite[Math.floor(finite.length * 0.22)];
  let highDb = finite[Math.floor(finite.length * 0.78)];
  for (let iteration = 0; iteration < 12; iteration += 1) {
    let lowSum = 0;
    let lowCount = 0;
    let highSum = 0;
    let highCount = 0;
    const midpoint = (lowDb + highDb) / 2;
    for (const level of finite) {
      if (level <= midpoint) {
        lowSum += level;
        lowCount += 1;
      } else {
        highSum += level;
        highCount += 1;
      }
    }
    if (lowCount) lowDb = lowSum / lowCount;
    if (highCount) highDb = highSum / highCount;
  }
  return { lowDb, highDb };
}

/**
 * Measures slow movement of the keyed cluster while retaining the global
 * two-cluster estimate as a prior. This is a detector for actual QSB, not a
 * replacement gate: quiet recordings should remain on the hard-gate path.
 */
export function analyzeQsbEnvelope(envelope: QsbEnvelope): QsbEnvelopeDiagnostics {
  assertEnvelope(envelope);
  if (!envelope.levelsDb.length) return { keyedClusterStdDb: 0, windowCount: 0 };

  const step = Math.max(1, Math.round(QSB_HMM_POLICY.diagnosticStepMs / envelope.hopMs));
  const radius = Math.max(step, Math.round(QSB_HMM_POLICY.diagnosticRadiusMs / envelope.hopMs));
  const keyedClusters: number[] = [];

  for (let index = 0; index < envelope.levelsDb.length; index += step) {
    const local = clusterLevels(envelope.levelsDb.slice(
      Math.max(0, index - radius),
      Math.min(envelope.levelsDb.length, index + radius + 1),
    ));
    const contrastDb = local.highDb - local.lowDb;
    const localTrust = clamp((contrastDb - 4) / 6, 0, 1);
    keyedClusters.push(local.highDb * localTrust + envelope.highDb * (1 - localTrust));
  }

  const mean = keyedClusters.reduce((sum, value) => sum + value, 0) /
    Math.max(1, keyedClusters.length);
  const variance = keyedClusters.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(1, keyedClusters.length);
  return {
    keyedClusterStdDb: Math.sqrt(variance),
    windowCount: keyedClusters.length,
  };
}

export function qsbHmmEligibility(
  diagnostics: QsbEnvelopeDiagnostics,
  timingQuality: number,
): QsbHmmEligibility {
  const blockers: QsbEligibilityBlocker[] = [];
  if (!Number.isFinite(timingQuality) || !Number.isFinite(diagnostics.keyedClusterStdDb)) {
    blockers.push("invalid-metrics");
  } else {
    if (timingQuality < QSB_HMM_POLICY.minimumTimingQuality) blockers.push("weak-timing");
    if (diagnostics.keyedClusterStdDb < QSB_HMM_POLICY.minimumKeyedClusterStdDb) {
      blockers.push("insufficient-qsb");
    }
  }
  return { eligible: blockers.length === 0, blockers };
}

function durationPrior(state: 0 | 1, durationUnits: number) {
  const gaussian = (center: number, spread: number, penalty = 0) =>
    penalty - 0.5 * ((durationUnits - center) / spread) ** 2;
  if (state) {
    return Math.max(
      gaussian(1, 0.38),
      gaussian(3, 0.68, -0.12),
    );
  }
  return Math.max(
    gaussian(1, 0.42),
    gaussian(3, 0.9, -0.08),
    gaussian(7, 2.1, -0.18),
    -2.2 - Math.max(0, durationUnits - 7) * 0.025,
  );
}

function statesToRuns(envelope: QsbEnvelope, states: Uint8Array): QsbGateRun[] {
  const runs: QsbGateRun[] = [];
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

/**
 * Produces a same-carrier alternate gate using a two-state hidden semi-Markov
 * Viterbi pass. The caller must seed `hardWpm` from the ordinary hard-gate
 * decode; the duration prior is deliberately not allowed to estimate speed.
 */
export function buildQsbHmmRuns(envelope: QsbEnvelope, hardWpm: number): QsbGateRun[] {
  assertEnvelope(envelope);
  if (!envelope.levelsDb.length) return [];
  if (!Number.isFinite(hardWpm) || hardWpm <= 0) {
    throw new Error("QSB HMM requires the positive WPM fitted by the hard gate.");
  }

  const factor = Math.max(1, Math.round(QSB_HMM_POLICY.frameMs / envelope.hopMs));
  const frameMs = factor * envelope.hopMs;
  const frameCount = Math.ceil(envelope.levelsDb.length / factor);
  const onPrefix = new Float64Array(frameCount + 1);
  const offPrefix = new Float64Array(frameCount + 1);
  const contrastDb = Math.max(4, envelope.highDb - envelope.lowDb);

  for (let frame = 0; frame < frameCount; frame += 1) {
    const start = frame * factor;
    const end = Math.min(envelope.levelsDb.length, start + factor);
    let normalized = 0;
    for (let index = start; index < end; index += 1) {
      const level = Number.isFinite(envelope.levelsDb[index])
        ? envelope.levelsDb[index]
        : envelope.lowDb;
      normalized += (level - envelope.lowDb) / contrastDb;
    }
    normalized = clamp(normalized / Math.max(1, end - start), -0.5, 1.5);
    const logOdds = clamp(
      (normalized - QSB_HMM_POLICY.emissionCenter) * QSB_HMM_POLICY.emissionGain,
      -7,
      7,
    );
    onPrefix[frame + 1] = onPrefix[frame] - Math.log1p(Math.exp(-logOdds));
    offPrefix[frame + 1] = offPrefix[frame] - Math.log1p(Math.exp(logOdds));
  }

  const boundedWpm = clamp(hardWpm, QSB_HMM_POLICY.minimumWpm, QSB_HMM_POLICY.maximumWpm);
  const unitFrames = Math.max(1.5, (1_200 / boundedWpm) / frameMs);
  const onMinimum = Math.max(1, Math.floor(unitFrames * 0.28));
  const onMaximum = Math.max(onMinimum, Math.ceil(unitFrames * 4.8));
  const offMinimum = Math.max(1, Math.floor(unitFrames * 0.25));
  const offMaximum = Math.min(
    frameCount,
    Math.max(Math.ceil(unitFrames * 15), Math.ceil(2_200 / frameMs)),
  );
  const onPriors = Float64Array.from({ length: onMaximum + 1 }, (_, duration) =>
    duration < onMinimum ? -Infinity : durationPrior(1, duration / unitFrames)
  );
  const offPriors = Float64Array.from({ length: offMaximum + 1 }, (_, duration) =>
    duration < offMinimum ? -Infinity : durationPrior(0, duration / unitFrames)
  );
  const onScore = new Float64Array(frameCount + 1).fill(-Infinity);
  const offScore = new Float64Array(frameCount + 1).fill(-Infinity);
  const onPrevious = new Int32Array(frameCount + 1).fill(-1);
  const offPrevious = new Int32Array(frameCount + 1).fill(-1);
  onScore[0] = 0;
  offScore[0] = 0;

  for (let end = 1; end <= frameCount; end += 1) {
    for (let duration = onMinimum; duration <= Math.min(onMaximum, end); duration += 1) {
      const start = end - duration;
      const score = offScore[start] + onPrefix[end] - onPrefix[start] + onPriors[duration];
      if (score > onScore[end]) {
        onScore[end] = score;
        onPrevious[end] = start;
      }
    }
    for (let duration = offMinimum; duration <= Math.min(offMaximum, end); duration += 1) {
      const start = end - duration;
      const score = onScore[start] + offPrefix[end] - offPrefix[start] + offPriors[duration];
      if (score > offScore[end]) {
        offScore[end] = score;
        offPrevious[end] = start;
      }
    }

    // Extend an established long silence with a bounded geometric tail. This
    // avoids O(n²) duration searches and prevents synthetic keyed blips in
    // multi-second pauses.
    const continuingStart = offPrevious[end - 1];
    if (continuingStart >= 0 && end - 1 - continuingStart >= offMaximum) {
      const score = offScore[end - 1] + offPrefix[end] - offPrefix[end - 1] - 0.002;
      if (score > offScore[end]) {
        offScore[end] = score;
        offPrevious[end] = continuingStart;
      }
    }
  }

  const compactStates = new Uint8Array(frameCount);
  let state: 0 | 1 = onScore[frameCount] > offScore[frameCount] ? 1 : 0;
  let end = frameCount;
  while (end > 0) {
    const start = state ? onPrevious[end] : offPrevious[end];
    if (start < 0 || start >= end) {
      compactStates.fill(0, 0, end);
      break;
    }
    if (state) compactStates.fill(1, start, end);
    end = start;
    state = state ? 0 : 1;
  }

  const states = new Uint8Array(envelope.levelsDb.length);
  for (let index = 0; index < states.length; index += 1) {
    states[index] = compactStates[Math.min(frameCount - 1, Math.floor(index / factor))];
  }
  return statesToRuns(envelope, states);
}

/**
 * Applies the full safety policy after the alternate runs have passed through
 * the ordinary timing/character decoder. Convergence is inherited solely from
 * the hard-gate result, so the HMM can improve copy but never manufacture a
 * convergence claim.
 */
export function selectQsbHmmAlternate(
  diagnostics: QsbEnvelopeDiagnostics,
  hard: QsbHardCandidateMetrics,
  alternate: QsbAlternateCandidateMetrics,
): QsbHmmSelection {
  const eligibility = qsbHmmEligibility(diagnostics, hard.timingQuality);
  const blockers: QsbSelectionBlocker[] = [...eligibility.blockers];
  const finiteMetrics = [
    hard.baseScore,
    hard.validRatio,
    hard.wpm,
    alternate.baseScore,
    alternate.validRatio,
    alternate.wpm,
  ].every(Number.isFinite) && hard.wpm > 0 && alternate.wpm > 0 &&
    hard.validRatio >= 0 && hard.validRatio <= 1 &&
    alternate.validRatio >= 0 && alternate.validRatio <= 1;

  const baseScoreGain = alternate.baseScore - hard.baseScore;
  const validRatioDrop = hard.validRatio - alternate.validRatio;
  const wpmDriftRatio = Math.abs(alternate.wpm - hard.wpm) / Math.max(1e-9, Math.abs(hard.wpm));

  if (!finiteMetrics && !blockers.includes("invalid-metrics")) blockers.push("invalid-metrics");
  if (hard.laneKey !== alternate.laneKey) blockers.push("different-lane");
  if (hard.transcript === alternate.transcript) blockers.push("same-copy");
  if (!Number.isFinite(baseScoreGain) ||
    baseScoreGain < QSB_HMM_POLICY.minimumBaseScoreGain - COMPARISON_EPSILON) {
    blockers.push("insufficient-score-gain");
  }
  if (!Number.isFinite(validRatioDrop) ||
    validRatioDrop > QSB_HMM_POLICY.maximumValidRatioDrop + COMPARISON_EPSILON) {
    blockers.push("validity-drop");
  }
  if (!Number.isFinite(wpmDriftRatio) ||
    wpmDriftRatio > QSB_HMM_POLICY.maximumWpmDriftRatio + COMPARISON_EPSILON) {
    blockers.push("wpm-drift");
  }

  return {
    useAlternate: blockers.length === 0,
    converged: hard.converged,
    blockers,
    baseScoreGain,
    validRatioDrop,
    wpmDriftRatio,
  };
}
