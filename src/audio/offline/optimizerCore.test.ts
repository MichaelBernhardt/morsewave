import { describe, expect, it } from "vitest";
import { encodeText } from "../morse";
import { optimizeMorseRecording } from "./optimizerCore";

interface ToneSegment {
  keyed: boolean;
  seconds: number;
}

function keyedSegments(text: string, wpm: number, wordGapUnits = 7) {
  const unit = 1.2 / wpm;
  const segments: ToneSegment[] = [{ keyed: false, seconds: unit * 3 }];
  const words = encodeText(text);
  words.forEach((word, wordIndex) => {
    word.forEach((pattern, characterIndex) => {
      [...pattern].forEach((symbol, symbolIndex) => {
        segments.push({ keyed: true, seconds: unit * (symbol === "." ? 1 : 3) });
        if (symbolIndex < pattern.length - 1) segments.push({ keyed: false, seconds: unit });
      });
      if (characterIndex < word.length - 1) segments.push({ keyed: false, seconds: unit * 3 });
    });
    if (wordIndex < words.length - 1) segments.push({ keyed: false, seconds: unit * wordGapUnits });
  });
  segments.push({ keyed: false, seconds: unit * 8 });
  return segments;
}

function renderKeyedTone(
  text: string,
  wpm: number,
  frequency: number,
  amplitude: number,
  sampleRate: number,
  wordGapUnits = 7,
) {
  const segments = keyedSegments(text, wpm, wordGapUnits);
  const length = Math.ceil(segments.reduce((sum, segment) => sum + segment.seconds, 0) * sampleRate);
  const samples = new Float32Array(length);
  let offset = 0;
  for (const segment of segments) {
    const end = Math.min(length, offset + Math.round(segment.seconds * sampleRate));
    if (segment.keyed) {
      for (let index = offset; index < end; index += 1) {
        samples[index] = Math.sin(2 * Math.PI * frequency * index / sampleRate) * amplitude;
      }
    }
    offset = end;
  }
  return samples;
}

function smoothKeyedTone(
  source: Float32Array,
  frequency: number,
  amplitude: number,
  sampleRate: number,
) {
  const samples = new Float32Array(source.length);
  const alpha = 1 - Math.exp(-1 / (sampleRate * 0.004));
  let envelope = 0;
  for (let index = 0; index < source.length; index += 1) {
    const target = Math.abs(source[index]) > 1e-8 ? 1 : 0;
    envelope += (target - envelope) * alpha;
    samples[index] = Math.sin(2 * Math.PI * frequency * index / sampleRate) *
      envelope * amplitude;
  }
  return samples;
}

describe("offline multipass optimizer", () => {
  it("keeps raw punctuation for scoring and annotates prosigns only in returned copy", () => {
    const sampleRate = 12_000;
    const samples = renderKeyedTone("CQ <BT> DE W1AW <AR> <SOS>", 20, 700, 0.42, sampleRate);
    const result = optimizeMorseRecording([samples], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 650,
      maximumFrequency: 750,
    });

    expect(result.rawTranscript).toBe("CQ = DE W1AW + <SOS>");
    expect(result.transcript).toBe("CQ <BT> DE W1AW <AR> <SOS>");
    expect(result.tokens.filter((token) => token.rawValue)).toEqual([
      expect.objectContaining({ pattern: "-...-", rawValue: "=", value: "<BT>" }),
      expect.objectContaining({ pattern: ".-.-.", rawValue: "+", value: "<AR>" }),
    ]);
  }, 30_000);

  it("uses coherent ham procedure only as bounded multipass evidence", () => {
    const sampleRate = 12_000;
    const samples = renderKeyedTone(
      "CQ CQ DE W1AW UR RST 5NN 5NN OP IS ED QTH",
      20,
      700,
      0.42,
      sampleRate,
    );
    const result = optimizeMorseRecording([samples], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 650,
      maximumFrequency: 750,
    });

    expect(result.rawTranscript).toBe("CQ CQ DE W1AW UR RST 5NN 5NN OP IS ED QTH");
    expect(result.hamEvidence).toBeGreaterThanOrEqual(0.08);
    expect(result.hamEvidence).toBeLessThanOrEqual(0.1);
    expect(result.candidates[0].hamEvidence).toBe(result.hamEvidence);
    expect(result.converged).toBe(true);
  }, 30_000);

  it("selects the intended Morse track instead of a stronger simultaneous station", () => {
    const sampleRate = 12_000;
    const wanted = renderKeyedTone("CALLED TEST", 20, 300, 0.42, sampleRate);
    const interferer = renderKeyedTone("TNX QSO", 13, 1800, 0.58, sampleRate);
    const length = Math.max(wanted.length, interferer.length);
    const mixed = new Float32Array(length);
    let seed = 17;
    for (let index = 0; index < length; index += 1) {
      seed = seed * 16807 % 2147483647;
      const noise = ((seed - 1) / 2147483646 * 2 - 1) * 0.008;
      mixed[index] = (wanted[index] || 0) + (interferer[index] || 0) + noise;
    }

    const result = optimizeMorseRecording([mixed], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 200,
      maximumFrequency: 2400,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(296);
    expect(result.frequency).toBeLessThanOrEqual(304);
    expect(result.transcript).toBe("CALLED TEST");
    expect(result.candidates.some((candidate) => candidate.frequency > 1750)).toBe(true);
    expect(result.passCount).toBeGreaterThan(500);
  }, 30_000);

  it("does not let an identical long spectral copy reduce decode confidence", () => {
    const sampleRate = 12_000;
    const text = "NOW 18 WPM TRANSITION FILE FOLLOWS";
    const primary = renderKeyedTone(text, 18, 700, 0.42, sampleRate);
    const spectralCopy = renderKeyedTone(text, 18, 1289, 0.34, sampleRate);
    const mixed = new Float32Array(Math.max(primary.length, spectralCopy.length));
    for (let index = 0; index < mixed.length; index += 1) {
      mixed[index] = (primary[index] || 0) + (spectralCopy[index] || 0);
    }

    const result = optimizeMorseRecording([mixed], sampleRate, {
      wpmHint: 18,
      languageAssist: true,
      minimumFrequency: 500,
      maximumFrequency: 1500,
    });

    expect(result.rawTranscript).toBe(text);
    expect(result.candidates.some((candidate) => candidate.frequency > 1200)).toBe(true);
    expect(result.confidence).toBeGreaterThanOrEqual(0.78);
    expect(result.converged).toBe(true);
  }, 30_000);

  it("opens a bounded deeper carrier search when the normal shortlist has no keyed timing", () => {
    const sampleRate = 12_000;
    const expected = "T25 T25";
    const wanted = renderKeyedTone(expected, 25, 2250, 0.25, sampleRate);
    let keyedEnvelope = 0;
    const envelopeAlpha = 1 - Math.exp(-1 / (sampleRate * 0.003));
    for (let index = 0; index < wanted.length; index += 1) {
      const target = Math.abs(wanted[index]) > 1e-8 ? 0.25 : 0;
      keyedEnvelope += (target - keyedEnvelope) * envelopeAlpha;
      wanted[index] = Math.sin(2 * Math.PI * 2250 * index / sampleRate) * keyedEnvelope;
    }
    // Exact hop-bin tones remain steady in the detector and deliberately fill
    // the four ordinary discovery slots without imitating keyed timing.
    const blockers = [375, 750, 1125, 1500];
    const mixed = wanted.slice();
    for (let index = 0; index < mixed.length; index += 1) {
      for (const frequency of blockers) {
        mixed[index] += Math.sin(2 * Math.PI * frequency * index / sampleRate) * 0.32;
      }
    }
    expect(mixed.length / sampleRate).toBeLessThanOrEqual(4.5);

    const result = optimizeMorseRecording([mixed], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 200,
      maximumFrequency: 2500,
    });

    expect(result.rawTranscript).toBe(expected);
  }, 30_000);

  it("prefers a correlated transmitter fundamental over its keyed harmonic", () => {
    const sampleRate = 12_000;
    const fundamental = renderKeyedTone("CWT KC4D DE KE5FRF", 27, 700, 0.34, sampleRate);
    const fifthHarmonic = renderKeyedTone("CWT KC4D DE KE5FRF", 27, 3500, 0.5, sampleRate);
    const mixed = new Float32Array(Math.max(fundamental.length, fifthHarmonic.length));
    for (let index = 0; index < mixed.length; index += 1) {
      mixed[index] = (fundamental[index] || 0) + (fifthHarmonic[index] || 0);
    }

    // Real archives frequently encode the same receiver as dual mono. A
    // transparent second channel must not consume the finalist slot needed by
    // the correlated fundamental.
    const result = optimizeMorseRecording([mixed, mixed.slice()], sampleRate, {
      wpmHint: 25,
      languageAssist: true,
      minimumFrequency: 200,
      maximumFrequency: 3800,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(696);
    expect(result.frequency).toBeLessThanOrEqual(704);
    expect(result.rawTranscript).toBe("CWT KC4D DE KE5FRF");
    // A correlated harmonic is the same keyed station, not an independent
    // station lane. Keep it out of the user-facing alternatives entirely.
    expect(result.candidates.some((candidate) => candidate.frequency > 3000)).toBe(false);
  }, 30_000);

  it("retries later windows after a long unmodulated opening carrier", () => {
    const sampleRate = 12_000;
    const frequency = 410;
    const openingFrames = sampleRate * 6;
    const closingFrames = Math.round(sampleRate * 1.5);
    const message = renderKeyedTone("WG", 9, frequency, 0.42, sampleRate);
    const samples = new Float32Array(openingFrames + message.length + closingFrames);
    for (let index = 0; index < openingFrames; index += 1) {
      samples[index] = Math.sin(2 * Math.PI * frequency * index / sampleRate) * 0.42;
    }
    samples.set(message, openingFrames);
    for (let index = openingFrames + message.length; index < samples.length; index += 1) {
      samples[index] = Math.sin(2 * Math.PI * frequency * index / sampleRate) * 0.42;
    }

    const result = optimizeMorseRecording([samples], sampleRate, {
      wpmHint: 10,
      languageAssist: true,
      minimumFrequency: 300,
      maximumFrequency: 500,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(406);
    expect(result.frequency).toBeLessThanOrEqual(414);
    expect(result.rawTranscript).toBe("WG");
    expect(result.converged).toBe(false);
  }, 30_000);

  it("verifies a timing-stable carrier after a sole long-file finalist collapses", () => {
    const sampleRate = 12_000;
    const durationSeconds = 121;
    const wantedFrequency = 1000;
    const falseFrequency = 1800;
    const wantedText = `8 7 M ${Array(18)
      .fill("879 36 62588 13583 6257T 5342T")
      .join(" ")}`;
    const wantedSource = renderKeyedTone(wantedText, 13, wantedFrequency, 1, sampleRate);
    const wanted = smoothKeyedTone(wantedSource, wantedFrequency, 0.7, sampleRate);
    const samples = new Float32Array(durationSeconds * sampleRate);
    samples.set(wanted.subarray(0, samples.length));

    // A convincing high-speed preview wins the one-carrier long-file budget.
    // After that preview, seven-mark garbage keeps the same spectral lane busy
    // while making its full-file Morse validity collapse.
    const falseSource = new Float32Array(samples.length);
    const falsePreview = renderKeyedTone(
      "TEST TEST TEST TEST TEST",
      60,
      falseFrequency,
      1,
      sampleRate,
    );
    falseSource.set(falsePreview.subarray(0, falseSource.length));
    const unitFrames = Math.round(sampleRate * 0.02);
    let offset = Math.round(sampleRate * 4.6);
    while (offset < falseSource.length) {
      for (let mark = 0; mark < 7 && offset < falseSource.length; mark += 1) {
        for (let frame = 0; frame < unitFrames && offset < falseSource.length; frame += 1) {
          falseSource[offset] = Math.sin(
            2 * Math.PI * falseFrequency * offset / sampleRate,
          );
          offset += 1;
        }
        offset += unitFrames;
      }
      offset += unitFrames * 3;
    }
    const falseTone = smoothKeyedTone(falseSource, falseFrequency, 0.35, sampleRate);
    for (let index = 0; index < samples.length; index += 1) samples[index] += falseTone[index];

    const result = optimizeMorseRecording([samples], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 950,
      maximumFrequency: 1850,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(996);
    expect(result.frequency).toBeLessThanOrEqual(1004);
    expect(result.wpm).toBeGreaterThan(12);
    expect(result.wpm).toBeLessThan(14);
    expect(result.rawTranscript).toContain("8 7 M 879 36 62588 13583 6257T 5342T");
    expect(result.candidates.some((candidate) =>
      candidate.frequency > 1400 && candidate.score < 0
    )).toBe(true);
  }, 30_000);

  it("recalibrates timing when a new operator takes over the same carrier", () => {
    const sampleRate = 12_000;
    const first = renderKeyedTone("CQ CQ DE K1ABC BK", 12, 700, 0.42, sampleRate);
    const second = renderKeyedTone("RR TU DE W9XYZ SK", 28, 700, 0.42, sampleRate);
    const samples = new Float32Array(first.length + second.length);
    samples.set(first);
    samples.set(second, first.length);

    const result = optimizeMorseRecording([samples], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 650,
      maximumFrequency: 750,
    });

    expect(result.rawTranscript).toBe("CQ CQ DE K1ABC BK RR TU DE W9XYZ SK");
  }, 30_000);

  it("uses structured traffic only to break a close physical carrier tie", () => {
    const sampleRate = 12_000;
    const wanted = renderKeyedTone("QTC 17/10 1618 UY5ZZ 4T8", 29, 790, 0.34, sampleRate);
    const keyClicks = renderKeyedTone("T H ES S S S D S N S T H S R S", 60, 560, 0.5, sampleRate);
    const mixed = new Float32Array(Math.max(wanted.length, keyClicks.length));
    for (let index = 0; index < mixed.length; index += 1) {
      mixed[index] = (wanted[index] || 0) + (keyClicks[index] || 0);
    }

    const result = optimizeMorseRecording([mixed], sampleRate, {
      wpmHint: 25,
      languageAssist: true,
      minimumFrequency: 450,
      maximumFrequency: 900,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(786);
    expect(result.frequency).toBeLessThanOrEqual(794);
    // The deliberately square 560 Hz key clicks corrupt the first few 790 Hz
    // edges, but the selected lane must retain the sustained structured copy.
    expect(result.rawTranscript).toContain("1618 UY5ZZ 4T8");
  }, 30_000);

  it("prefers a complete hand-spaced message over a short callsign interferer", () => {
    const sampleRate = 12_000;
    const expected = "WEATHER GREAT FUN WITH LOCAL RADIO CLUB VIDEO ON MY RIG";
    const wanted = renderKeyedTone(expected, 22, 795, 0.34, sampleRate, 4.1);
    const interferer = renderKeyedTone("G0TUE G0TUE G0XEPSE K", 20, 2300, 0.52, sampleRate);
    const length = Math.max(wanted.length, interferer.length);
    const mixed = new Float32Array(length);
    let seed = 29;
    for (let index = 0; index < length; index += 1) {
      seed = seed * 16807 % 2147483647;
      const noise = ((seed - 1) / 2147483646 * 2 - 1) * 0.009;
      mixed[index] = (wanted[index] || 0) + (interferer[index] || 0) + noise;
    }

    const result = optimizeMorseRecording([mixed], sampleRate, {
      wpmHint: 20,
      languageAssist: true,
      minimumFrequency: 200,
      maximumFrequency: 2600,
    });

    expect(result.frequency).toBeGreaterThanOrEqual(790);
    expect(result.frequency).toBeLessThanOrEqual(800);
    expect(result.transcript).toBe(expected);
    expect(result.converged).toBe(true);
    expect(result.candidates.some((candidate) => candidate.frequency > 2250)).toBe(true);
  }, 30_000);
});
