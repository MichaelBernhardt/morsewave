import { describe, expect, it } from "vitest";
import {
  QSB_HMM_POLICY,
  analyzeQsbEnvelope,
  buildQsbHmmRuns,
  qsbHmmEligibility,
  selectQsbHmmAlternate,
  type QsbEnvelope,
  type QsbEnvelopeDiagnostics,
  type QsbHardCandidateMetrics,
} from "./qsbHmm";

function envelopeWithKeyedVariation(keyedVariationDb: number): QsbEnvelope {
  const hopMs = 2;
  const levelsDb = new Float32Array(10_000);
  for (let index = 0; index < levelsDb.length; index += 1) {
    const timeMs = index * hopMs;
    const keyed = timeMs % 160 < 80;
    const fadeDb = keyedVariationDb * Math.sin(2 * Math.PI * timeMs / 8_000);
    levelsDb[index] = keyed ? -24 + fadeDb : -70;
  }
  return { levelsDb, hopMs, offsetMs: 12, lowDb: -70, highDb: -24 };
}

function envelopeFromSegments(segments: Array<{ keyed: boolean; units: number }>, wpm = 20) {
  const hopMs = 2;
  const unitMs = 1_200 / wpm;
  const levels: number[] = [];
  for (const segment of segments) {
    const frames = Math.round(segment.units * unitMs / hopMs);
    for (let frame = 0; frame < frames; frame += 1) {
      levels.push(segment.keyed ? -20 : -70);
    }
  }
  return {
    levelsDb: Float32Array.from(levels),
    hopMs,
    offsetMs: 12,
    lowDb: -70,
    highDb: -20,
  } satisfies QsbEnvelope;
}

const qsbDiagnostics: QsbEnvelopeDiagnostics = {
  keyedClusterStdDb: 4.2,
  windowCount: 80,
};

const hardCandidate: QsbHardCandidateMetrics = {
  laneKey: "0:595",
  transcript: "QTC 48/10 DL5ALI",
  baseScore: 0.699,
  validRatio: 1,
  wpm: 27.27,
  timingQuality: 0.721,
  converged: false,
};

describe("bounded QSB HMM", () => {
  it("locks the corpus-proven policy constants", () => {
    expect(QSB_HMM_POLICY).toMatchObject({
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
    });
    expect(Object.isFrozen(QSB_HMM_POLICY)).toBe(true);
  });

  it("detects slow keyed-level movement but ignores a stationary keyed cluster", () => {
    const stable = analyzeQsbEnvelope(envelopeWithKeyedVariation(0));
    const fading = analyzeQsbEnvelope(envelopeWithKeyedVariation(9));

    expect(stable.keyedClusterStdDb).toBeLessThan(0.1);
    expect(fading.keyedClusterStdDb).toBeGreaterThan(QSB_HMM_POLICY.minimumKeyedClusterStdDb);
    expect(fading.windowCount).toBeGreaterThan(20);
  });

  it("requires both reliable hard-gate timing and physical QSB evidence", () => {
    expect(qsbHmmEligibility(qsbDiagnostics, 0.7)).toEqual({ eligible: true, blockers: [] });
    expect(qsbHmmEligibility({ ...qsbDiagnostics, keyedClusterStdDb: 2.999 }, 0.7)).toEqual({
      eligible: false,
      blockers: ["insufficient-qsb"],
    });
    expect(qsbHmmEligibility(qsbDiagnostics, 0.699)).toEqual({
      eligible: false,
      blockers: ["weak-timing"],
    });
  });

  it("produces bounded Morse-duration runs and bridges a short in-mark fade", () => {
    const envelope = envelopeFromSegments([
      { keyed: false, units: 3 },
      { keyed: true, units: 1 },
      { keyed: false, units: 1 },
      { keyed: true, units: 3 },
      { keyed: false, units: 7 },
    ]);
    // A 6 ms dropout is shorter than the HMM observation frame.
    const dashStart = Math.round((3 + 1 + 1) * 60 / envelope.hopMs);
    envelope.levelsDb.fill(-70, dashStart + 30, dashStart + 33);

    const runs = buildQsbHmmRuns(envelope, 20);
    const keyed = runs.filter((run) => run.keyed);

    expect(keyed).toHaveLength(2);
    expect(keyed[0].endMs - keyed[0].startMs).toBeCloseTo(60, -1);
    expect(keyed[1].endMs - keyed[1].startMs).toBeGreaterThanOrEqual(160);
    expect(runs.every((run) => run.endMs > run.startMs)).toBe(true);
  });

  it("does not hallucinate marks in silence and handles an empty envelope", () => {
    const silence = envelopeWithKeyedVariation(0);
    silence.levelsDb.fill(-70);
    expect(buildQsbHmmRuns(silence, 20).some((run) => run.keyed)).toBe(false);
    expect(buildQsbHmmRuns({ ...silence, levelsDb: new Float32Array() }, 20)).toEqual([]);
    expect(() => buildQsbHmmRuns(silence, Number.NaN)).toThrow(/positive WPM/);
  });

  it("selects a same-lane alternate only inside every measured safety bound", () => {
    const selected = selectQsbHmmAlternate(qsbDiagnostics, hardCandidate, {
      laneKey: hardCandidate.laneKey,
      transcript: "QTC 48/10 DL5AWI",
      baseScore: hardCandidate.baseScore + QSB_HMM_POLICY.minimumBaseScoreGain,
      validRatio: 0.99,
      wpm: hardCandidate.wpm * 1.15,
    });

    expect(selected.useAlternate).toBe(true);
    expect(selected.blockers).toEqual([]);
    // The hard decode was non-converged; the HMM is never allowed to promote it.
    expect(selected.converged).toBe(false);
  });

  it.each([
    ["different lane", { laneKey: "1:595" }, "different-lane"],
    ["same copy", { transcript: hardCandidate.transcript }, "same-copy"],
    ["small score gain", { baseScore: hardCandidate.baseScore + 0.0079 }, "insufficient-score-gain"],
    ["validity loss", { validRatio: 0.989 }, "validity-drop"],
    ["speed drift", { wpm: hardCandidate.wpm * 1.151 }, "wpm-drift"],
  ])("rejects %s", (_name, change, blocker) => {
    const decision = selectQsbHmmAlternate(qsbDiagnostics, hardCandidate, {
      laneKey: hardCandidate.laneKey,
      transcript: "QTC 48/10 DL5AWI",
      baseScore: hardCandidate.baseScore + 0.009,
      validRatio: 0.99,
      wpm: hardCandidate.wpm,
      ...change,
    });
    expect(decision.useAlternate).toBe(false);
    expect(decision.blockers).toContain(blocker);
  });

  it("keeps the Priyom-style failure modes out of the alternate path", () => {
    const weakTiming = selectQsbHmmAlternate(
      { keyedClusterStdDb: 6.46, windowCount: 240 },
      { ...hardCandidate, timingQuality: 0.581 },
      {
        laneKey: hardCandidate.laneKey,
        transcript: "851 851 E E 488 59",
        baseScore: hardCandidate.baseScore + 0.071,
        validRatio: 1,
        wpm: hardCandidate.wpm,
      },
    );
    const stationarySignal = selectQsbHmmAlternate(
      { keyedClusterStdDb: 0.59, windowCount: 270 },
      { ...hardCandidate, timingQuality: 0.8 },
      {
        laneKey: hardCandidate.laneKey,
        transcript: "T58 T58 297",
        baseScore: hardCandidate.baseScore + 0.031,
        validRatio: 1,
        wpm: hardCandidate.wpm,
      },
    );

    expect(weakTiming.blockers).toContain("weak-timing");
    expect(stationarySignal.blockers).toContain("insufficient-qsb");
    expect(weakTiming.useAlternate).toBe(false);
    expect(stationarySignal.useAlternate).toBe(false);
  });

  it.each([
    ["ARRL", 4.29, 0.663, false],
    ["OAP f70", 1.89, 0.921, false],
    ["OAP 6d", 1.94, 0.733, false],
    ["CJ3A", 5.21, 0.83, true],
    ["WAE JH3AIU", 4.2, 0.721, true],
    ["WAE KL7HF", 8.27, 0.839, true],
    ["Priyom M12", 6.46, 0.581, false],
    ["Priyom M14", 0.59, 0.693, false],
    ["UCSB cylinder", 1.02, 0.36, false],
  ])("matches the measured %s corpus eligibility", (_sample, keyedStdDb, timing, eligible) => {
    expect(qsbHmmEligibility({ keyedClusterStdDb: keyedStdDb, windowCount: 1 }, timing).eligible)
      .toBe(eligible);
  });

  it("preserves an existing hard-gate convergence claim without creating one", () => {
    const decision = selectQsbHmmAlternate(qsbDiagnostics, {
      ...hardCandidate,
      converged: true,
    }, {
      laneKey: hardCandidate.laneKey,
      transcript: "QTC 48/10 DL5AWI",
      baseScore: hardCandidate.baseScore + 0.009,
      validRatio: 1,
      wpm: hardCandidate.wpm,
    });
    expect(decision.useAlternate).toBe(true);
    expect(decision.converged).toBe(true);
  });
});
