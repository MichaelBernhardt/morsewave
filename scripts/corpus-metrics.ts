export interface ErrorMetric {
  readonly distance: number;
  readonly referenceLength: number;
  readonly hypothesisLength: number;
  readonly errorRate: number;
  readonly accuracy: number;
}

export interface SpacingMetric {
  readonly referenceBoundaries: number;
  readonly hypothesisBoundaries: number;
  readonly matchedBoundaries: number;
  readonly precision: number;
  readonly recall: number;
  readonly f1: number;
  readonly exact: boolean;
}

export interface CopyMetrics {
  readonly normalizedReference: string;
  readonly normalizedHypothesis: string;
  /** CER excludes whitespace; word-boundary quality is reported separately. */
  readonly characters: ErrorMetric;
  readonly words: ErrorMetric;
  readonly spacing: SpacingMetric;
}

export interface NumericRange {
  readonly target?: number;
  readonly tolerance?: number;
  readonly min?: number;
  readonly max?: number;
}

export interface NumericExpectation extends NumericRange {
  /** Accept any one of several narrow, physically equivalent ranges. */
  readonly anyOf?: readonly NumericRange[];
}

export interface CorpusLimits {
  readonly maxCer?: number;
  readonly maxWer?: number;
  readonly minBoundaryF1?: number;
  readonly minAnchorRecall?: number;
  readonly minKeywordRecall?: number;
}

export interface CorpusExpectations {
  readonly reference?: string;
  /** Normalized phrases that may occur inside a partial/off-air copy. */
  readonly anchors?: readonly string[];
  /** Exact normalized word tokens expected somewhere in the copy. */
  readonly keywords?: readonly string[];
  readonly carrierHz?: NumericExpectation;
  readonly wpm?: NumericExpectation;
  readonly converged?: boolean;
  readonly limits?: CorpusLimits;
}

export interface CorpusObservation {
  readonly rawTranscript: string;
  readonly frequency: number;
  readonly fundamentalHz?: number;
  readonly wpm: number;
  readonly converged: boolean;
}

export interface CheckResult {
  readonly name: string;
  readonly passed: boolean;
  readonly actual: string | number | boolean;
  readonly expected: string | number | boolean;
}

export interface MatchMetric {
  readonly matched: number;
  readonly total: number;
  readonly recall: number;
  readonly results: readonly { value: string; matched: boolean }[];
}

export interface CorpusAssessment {
  readonly copy?: CopyMetrics;
  readonly anchors: MatchMetric;
  readonly keywords: MatchMetric;
  readonly checks: readonly CheckResult[];
  readonly passed: boolean;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

export function normalizeRawCopy(value: string) {
  return value
    .normalize("NFKC")
    .toUpperCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, "\"")
    .replace(/[\u2013\u2014]/g, "-")
    .replace(/\s+/g, " ")
    .trim();
}

export function editDistance<T>(reference: readonly T[], hypothesis: readonly T[]) {
  if (!reference.length) return hypothesis.length;
  if (!hypothesis.length) return reference.length;
  let previous = Array.from({ length: hypothesis.length + 1 }, (_, index) => index);
  for (let left = 1; left <= reference.length; left += 1) {
    const current = new Array<number>(hypothesis.length + 1);
    current[0] = left;
    for (let right = 1; right <= hypothesis.length; right += 1) {
      current[right] = Math.min(
        current[right - 1] + 1,
        previous[right] + 1,
        previous[right - 1] + (reference[left - 1] === hypothesis[right - 1] ? 0 : 1),
      );
    }
    previous = current;
  }
  return previous[hypothesis.length];
}

function errorMetric<T>(reference: readonly T[], hypothesis: readonly T[]): ErrorMetric {
  const distance = editDistance(reference, hypothesis);
  const errorRate = reference.length ? distance / reference.length : hypothesis.length ? 1 : 0;
  return {
    distance,
    referenceLength: reference.length,
    hypothesisLength: hypothesis.length,
    errorRate,
    accuracy: clamp01(1 - errorRate),
  };
}

function copyWords(value: string) {
  const normalized = normalizeRawCopy(value);
  return normalized ? normalized.split(" ") : [];
}

function boundaryOffsets(words: readonly string[]) {
  const offsets = new Set<number>();
  let offset = 0;
  for (let index = 0; index < words.length - 1; index += 1) {
    offset += [...words[index]].length;
    offsets.add(offset);
  }
  return offsets;
}

type AlignmentStep = "match" | "delete-reference" | "insert-hypothesis";

function alignedHypothesisOffsets(reference: readonly string[], hypothesis: readonly string[]) {
  const width = hypothesis.length + 1;
  const costs = new Uint32Array((reference.length + 1) * width);
  for (let left = 0; left <= reference.length; left += 1) costs[left * width] = left;
  for (let right = 0; right <= hypothesis.length; right += 1) costs[right] = right;
  for (let left = 1; left <= reference.length; left += 1) {
    for (let right = 1; right <= hypothesis.length; right += 1) {
      costs[left * width + right] = Math.min(
        costs[left * width + right - 1] + 1,
        costs[(left - 1) * width + right] + 1,
        costs[(left - 1) * width + right - 1] +
          (reference[left - 1] === hypothesis[right - 1] ? 0 : 1),
      );
    }
  }

  const reversed: AlignmentStep[] = [];
  let left = reference.length;
  let right = hypothesis.length;
  while (left > 0 || right > 0) {
    const current = costs[left * width + right];
    if (left > 0 && right > 0 && current === costs[(left - 1) * width + right - 1] +
      (reference[left - 1] === hypothesis[right - 1] ? 0 : 1)) {
      reversed.push("match");
      left -= 1;
      right -= 1;
    } else if (left > 0 && current === costs[(left - 1) * width + right] + 1) {
      reversed.push("delete-reference");
      left -= 1;
    } else {
      reversed.push("insert-hypothesis");
      right -= 1;
    }
  }

  const mapping = new Uint32Array(hypothesis.length + 1);
  left = 0;
  right = 0;
  mapping[0] = 0;
  for (const step of reversed.reverse()) {
    if (step === "match") {
      left += 1;
      right += 1;
      mapping[right] = left;
    } else if (step === "delete-reference") {
      left += 1;
      mapping[right] = left;
    } else {
      right += 1;
      mapping[right] = left;
    }
  }
  return mapping;
}

export function copyMetrics(reference: string, hypothesis: string): CopyMetrics {
  const normalizedReference = normalizeRawCopy(reference);
  const normalizedHypothesis = normalizeRawCopy(hypothesis);
  const referenceWords = copyWords(normalizedReference);
  const hypothesisWords = copyWords(normalizedHypothesis);
  const referenceCharacters = [...normalizedReference.replace(/\s/g, "")];
  const hypothesisCharacters = [...normalizedHypothesis.replace(/\s/g, "")];
  const expectedBoundaries = boundaryOffsets(referenceWords);
  const actualBoundaries = boundaryOffsets(hypothesisWords);
  // Compare word breaks on the optimal character alignment. Otherwise one
  // early inserted dit shifts every later absolute offset and makes correct
  // spacing look catastrophically wrong.
  const alignedOffsets = alignedHypothesisOffsets(referenceCharacters, hypothesisCharacters);
  const consumedReferenceBoundaries = new Set<number>();
  let matchedBoundaries = 0;
  for (const offset of actualBoundaries) {
    const aligned = alignedOffsets[Math.min(offset, hypothesisCharacters.length)];
    if (expectedBoundaries.has(aligned) && !consumedReferenceBoundaries.has(aligned)) {
      consumedReferenceBoundaries.add(aligned);
      matchedBoundaries += 1;
    }
  }
  const precision = actualBoundaries.size ? matchedBoundaries / actualBoundaries.size : expectedBoundaries.size ? 0 : 1;
  const recall = expectedBoundaries.size ? matchedBoundaries / expectedBoundaries.size : actualBoundaries.size ? 0 : 1;
  const f1 = precision + recall ? 2 * precision * recall / (precision + recall) : 0;

  return {
    normalizedReference,
    normalizedHypothesis,
    characters: errorMetric(referenceCharacters, hypothesisCharacters),
    words: errorMetric(referenceWords, hypothesisWords),
    spacing: {
      referenceBoundaries: expectedBoundaries.size,
      hypothesisBoundaries: actualBoundaries.size,
      matchedBoundaries,
      precision,
      recall,
      f1,
      exact: normalizedReference === normalizedHypothesis,
    },
  };
}

function normalizedWordTokens(value: string) {
  return normalizeRawCopy(value).match(/<[A-Z]+>|[A-Z0-9]+(?:\/[A-Z0-9]+)*|[=+&]/g) ?? [];
}

function phraseMatches(expected: readonly string[] | undefined, transcript: string): MatchMetric {
  const normalizedTranscript = normalizeRawCopy(transcript);
  const results = (expected ?? []).map((value) => ({
    value,
    matched: normalizedTranscript.includes(normalizeRawCopy(value)),
  }));
  const matched = results.filter((result) => result.matched).length;
  return { matched, total: results.length, recall: results.length ? matched / results.length : 1, results };
}

function keywordMatches(expected: readonly string[] | undefined, transcript: string): MatchMetric {
  const actual = new Set(normalizedWordTokens(transcript));
  const results = (expected ?? []).map((value) => {
    const tokens = normalizedWordTokens(value);
    return { value, matched: tokens.length > 0 && tokens.every((token) => actual.has(token)) };
  });
  const matched = results.filter((result) => result.matched).length;
  return { matched, total: results.length, recall: results.length ? matched / results.length : 1, results };
}

function numericBounds(expectation: NumericRange) {
  const minimum = expectation.min ?? (expectation.target !== undefined
    ? expectation.target - (expectation.tolerance ?? 0)
    : -Infinity);
  const maximum = expectation.max ?? (expectation.target !== undefined
    ? expectation.target + (expectation.tolerance ?? 0)
    : Infinity);
  const label = expectation.target !== undefined
    ? `${expectation.target} ± ${expectation.tolerance ?? 0}`
    : `${Number.isFinite(minimum) ? minimum : "-∞"}…${Number.isFinite(maximum) ? maximum : "∞"}`;
  return { minimum, maximum, label };
}

function numericCheck(name: string, actual: number, expectation: NumericExpectation): CheckResult {
  const ranges = (expectation.anyOf?.length ? expectation.anyOf : [expectation]).map(numericBounds);
  return {
    name,
    passed: ranges.some(({ minimum, maximum }) => actual >= minimum && actual <= maximum),
    actual,
    expected: ranges.map(({ label }) => label).join(" or "),
  };
}

export function assessCorpusCopy(
  expectation: CorpusExpectations,
  observation: CorpusObservation,
): CorpusAssessment {
  const copy = expectation.reference === undefined
    ? undefined
    : copyMetrics(expectation.reference, observation.rawTranscript);
  const anchors = phraseMatches(expectation.anchors, observation.rawTranscript);
  const keywords = keywordMatches(expectation.keywords, observation.rawTranscript);
  const checks: CheckResult[] = [];
  const limits = expectation.limits ?? {};

  if (expectation.carrierHz) checks.push(numericCheck(
    "carrierHz",
    observation.fundamentalHz ?? observation.frequency,
    expectation.carrierHz,
  ));
  if (expectation.wpm) checks.push(numericCheck("wpm", observation.wpm, expectation.wpm));
  if (expectation.converged !== undefined) {
    checks.push({
      name: "converged",
      passed: observation.converged === expectation.converged,
      actual: observation.converged,
      expected: expectation.converged,
    });
  }
  if (limits.maxCer !== undefined && copy) {
    checks.push({ name: "cer", passed: copy.characters.errorRate <= limits.maxCer,
      actual: copy.characters.errorRate, expected: `<= ${limits.maxCer}` });
  }
  if (limits.maxWer !== undefined && copy) {
    checks.push({ name: "wer", passed: copy.words.errorRate <= limits.maxWer,
      actual: copy.words.errorRate, expected: `<= ${limits.maxWer}` });
  }
  if (limits.minBoundaryF1 !== undefined && copy) {
    checks.push({ name: "boundaryF1", passed: copy.spacing.f1 >= limits.minBoundaryF1,
      actual: copy.spacing.f1, expected: `>= ${limits.minBoundaryF1}` });
  }
  if (anchors.total) {
    const minimum = limits.minAnchorRecall ?? 1;
    checks.push({ name: "anchorRecall", passed: anchors.recall >= minimum,
      actual: anchors.recall, expected: `>= ${minimum}` });
  }
  if (keywords.total) {
    const minimum = limits.minKeywordRecall ?? 1;
    checks.push({ name: "keywordRecall", passed: keywords.recall >= minimum,
      actual: keywords.recall, expected: `>= ${minimum}` });
  }

  return Object.freeze({ copy, anchors, keywords, checks, passed: checks.every((check) => check.passed) });
}
