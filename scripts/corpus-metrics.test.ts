import { describe, expect, it } from "vitest";
import { assessCorpusCopy, copyMetrics, editDistance, normalizeRawCopy } from "./corpus-metrics";

describe("corpus copy metrics", () => {
  it("normalizes case and whitespace but keeps literal Morse punctuation", () => {
    expect(normalizeRawCopy("  =  Now\n18 wpm  = ")).toBe("= NOW 18 WPM =");
    expect(normalizeRawCopy("<BT> + &")).toBe("<BT> + &");
  });

  it("computes standard Levenshtein distance", () => {
    expect(editDistance([..."KITTEN"], [..."SITTING"])).toBe(3);
    expect(editDistance([], ["A", "B"])).toBe(2);
  });

  it("separates character, word, and boundary errors", () => {
    const onlySpacing = copyMetrics("CQ CQ DE W1AW", "CQCQ DE W1AW");
    expect(onlySpacing.characters.errorRate).toBe(0);
    expect(onlySpacing.words.distance).toBe(2);
    expect(onlySpacing.spacing.f1).toBeLessThan(1);

    const substitution = copyMetrics("MORSE CODE", "MORSE COPE");
    expect(substitution.characters.distance).toBe(1);
    expect(substitution.words.distance).toBe(1);
  });

  it("aligns word boundaries after an inserted character", () => {
    const inserted = copyMetrics(
      "QTC 48/10 1451 DL5AWI 38",
      "QTC 48/10 E1451 DL5AWI 38",
    );

    expect(inserted.characters.distance).toBe(1);
    expect(inserted.spacing.f1).toBe(1);
  });

  it("scores exact references, partial anchors, keywords, and receiver expectations", () => {
    const assessment = assessCorpusCopy({
      reference: "CQ CQ DE W1AW",
      anchors: ["CQ CQ", "DE W1AW"],
      keywords: ["CQ", "W1AW"],
      carrierHz: { target: 750, tolerance: 20 },
      wpm: { min: 17, max: 19 },
      converged: true,
      limits: { maxCer: 0, maxWer: 0, minBoundaryF1: 1 },
    }, {
      rawTranscript: "cq   cq de w1aw",
      frequency: 748,
      wpm: 18.2,
      converged: true,
    });

    expect(assessment.passed).toBe(true);
    expect(assessment.anchors).toMatchObject({ matched: 2, total: 2, recall: 1 });
    expect(assessment.keywords).toMatchObject({ matched: 2, total: 2, recall: 1 });
  });

  it("does not let substring-only copies satisfy exact keyword assertions", () => {
    const assessment = assessCorpusCopy({
      anchors: ["WII800"],
      keywords: ["WII800"],
    }, {
      rawTranscript: "I WII800E",
      frequency: 646,
      wpm: 26.7,
      converged: false,
    });

    expect(assessment.anchors.recall).toBe(1);
    expect(assessment.keywords.recall).toBe(0);
    expect(assessment.passed).toBe(false);
  });

  it("does not treat an empty normalized keyword as a match", () => {
    const assessment = assessCorpusCopy({ keywords: ["???"] }, {
      rawTranscript: "CQ DE W1AW",
      frequency: 750,
      wpm: 18,
      converged: true,
    });

    expect(assessment.keywords.recall).toBe(0);
    expect(assessment.passed).toBe(false);
  });

  it("accepts either of two narrow physical carrier bands", () => {
    const expectation = {
      carrierHz: {
        anyOf: [
          { target: 1_000, tolerance: 50 },
          { target: 2_000, tolerance: 50 },
        ],
      },
    };

    expect(assessCorpusCopy(expectation, {
      rawTranscript: "T25",
      frequency: 2_018,
      wpm: 18,
      converged: false,
    }).passed).toBe(true);
    expect(assessCorpusCopy(expectation, {
      rawTranscript: "T25",
      frequency: 1_497,
      wpm: 18,
      converged: false,
    }).passed).toBe(false);
  });

  it("scores a reported harmonic fundamental without disguising the decode lane", () => {
    const assessment = assessCorpusCopy({
      carrierHz: { target: 700, tolerance: 20 },
    }, {
      rawTranscript: "WB7B 5NN",
      frequency: 2_098,
      fundamentalHz: 699,
      wpm: 28,
      converged: false,
    });

    expect(assessment.passed).toBe(true);
    expect(assessment.checks[0]).toMatchObject({ actual: 699 });
  });
});
