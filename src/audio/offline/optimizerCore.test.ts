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
  }, 10_000);

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
  }, 10_000);

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
  }, 10_000);
});
