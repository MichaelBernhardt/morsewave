import { describe, expect, it } from "vitest";
import { AdaptiveMorseDecoder } from "./AdaptiveMorseDecoder";
import { encodeText } from "./morse";

interface Segment {
  tone: boolean;
  duration: number;
}

function signalFor(text: string, unit: number): Segment[] {
  const words = encodeText(text);
  const segments: Segment[] = [{ tone: false, duration: unit * 5 }];
  words.forEach((word, wordIndex) => {
    word.forEach((pattern, characterIndex) => {
      [...pattern].forEach((symbol, symbolIndex) => {
        segments.push({ tone: true, duration: unit * (symbol === "." ? 1 : 3) });
        if (symbolIndex < pattern.length - 1) segments.push({ tone: false, duration: unit });
      });
      if (characterIndex < word.length - 1) segments.push({ tone: false, duration: unit * 3 });
    });
    if (wordIndex < words.length - 1) segments.push({ tone: false, duration: unit * 7 });
  });
  segments.push({ tone: false, duration: unit * 8 });
  return segments;
}

function decode(text: string, actualWpm: number, hintWpm = actualWpm, jitter = 0, randomSeed = 42) {
  let output = "";
  const decoded: string[] = [];
  const decoder = new AdaptiveMorseDecoder(
    { sensitivity: 68, wpmHint: hintWpm, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
    {
      onCharacter: ({ value, pattern }) => { output += value; decoded.push(`${pattern}:${value}`); },
      onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
    },
  );
  const unit = 1200 / actualWpm;
  let time = 0;
  let seed = randomSeed;
  const random = () => {
    seed = seed * 16807 % 2147483647;
    return (seed - 1) / 2147483646;
  };
  for (const segment of signalFor(text, unit)) {
    const variedDuration = segment.duration * (1 + (random() * 2 - 1) * jitter);
    const end = time + variedDuration;
    while (time <= end) {
      const noise = (random() * 2 - 1) * 1.5;
      decoder.process({
        timestamp: time,
        levelDb: (segment.tone ? -27 : -72) + noise,
        purity: segment.tone ? 0.88 : 0.035,
        frequency: 700,
      });
      time += 5;
    }
  }
  decoder.flush(time);
  return { output: output.trim(), metrics: decoder.getMetrics(), decoded };
}

function decodeTimingProfile(text: string, actualWpm: number, profile: {
  dot: number;
  dash: number;
  elementGap: number;
  characterGap: number;
  wordGap: number;
}, hintWpm = actualWpm) {
  let output = "";
  const unit = 1200 / actualWpm;
  const decoder = new AdaptiveMorseDecoder(
    { sensitivity: 82, wpmHint: hintWpm, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
    {
      onCharacter: ({ value }) => { output += value; },
      onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
    },
  );
  let time = 0;
  for (const segment of signalFor(text, unit)) {
    let duration = segment.duration;
    if (segment.tone) duration = unit * (segment.duration < unit * 2 ? profile.dot : profile.dash);
    else if (Math.abs(segment.duration - unit) < 1) duration = unit * profile.elementGap;
    else if (Math.abs(segment.duration - unit * 3) < 1) duration = unit * profile.characterGap;
    else if (Math.abs(segment.duration - unit * 7) < 1) duration = unit * profile.wordGap;
    const end = time + duration;
    while (time <= end) {
      decoder.process({
        timestamp: time,
        levelDb: segment.tone ? -30 : -78,
        purity: segment.tone ? 0.9 : 0.025,
        frequency: 700,
      });
      time += 5;
    }
  }
  decoder.flush(time);
  return { output: output.trim(), metrics: decoder.getMetrics() };
}

describe("AdaptiveMorseDecoder", () => {
  it("decodes canonical International Morse", () => {
    expect(decode("SOS TEST 123", 20).output).toBe("SOS TEST 123");
    expect(decode("<SOS>", 20).output).toBe("<SOS>");
    expect(decode("WII800", 20).output).toBe("WII800");
  });

  it("tolerates substantial keying jitter", () => {
    expect(decode("CQ DE W1AW", 18, 18, 0.16).output).toBe("CQ DE W1AW");
  });

  it("recovers highly uneven novice timing", () => {
    expect(decode("CQ DE BEGINNER", 13, 18, 0.42).output).toBe("CQ DE BEGINNER");
  });

  it("stays stable across randomized hand-key timing profiles", () => {
    for (let seed = 1; seed <= 30; seed += 1) {
      expect(decode("BAD FIST TEST", 15, 20, 0.42, seed).output, `seed ${seed}`).toBe("BAD FIST TEST");
    }
  }, 20_000);

  it("bridges key bounce, brief fades, and false edge clicks", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 92, wpmHint: 16, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      {
        onCharacter: ({ value }) => { output += value; },
        onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
      },
    );
    const unit = 1200 / 16;
    let time = 0;
    for (const segment of signalFor("BOUNCE TEST", unit)) {
      const started = time;
      const end = time + segment.duration;
      while (time <= end) {
        const within = time - started;
        const briefFade = segment.tone && segment.duration > unit * 2 &&
          within > segment.duration * 0.5 && within < segment.duration * 0.5 + 12;
        const contactClick = !segment.tone && segment.duration > unit * 2 && within >= 35 && within <= 45;
        const keyed = segment.tone ? !briefFade : contactClick;
        decoder.process({
          timestamp: time,
          levelDb: keyed ? -48 : -65,
          purity: keyed ? 0.72 : 0.035,
          frequency: 704,
        });
        time += 5;
      }
    }
    decoder.flush(time);
    expect(output.trim()).toBe("BOUNCE TEST");
    expect(decoder.getMetrics().recoveredEdges).toBeGreaterThan(0);
  });

  it("can infer one fully faded element inside a callsign", () => {
    let output = "";
    const unit = 1200 / 20;
    const original = signalFor("DE F5LPY", unit);
    let markIndex = 0;
    const missingMark = original.findIndex((segment) => segment.tone && markIndex++ === 11);
    const damaged = [
      ...original.slice(0, missingMark - 1),
      {
        tone: false,
        duration: original[missingMark - 1].duration + original[missingMark].duration + original[missingMark + 1].duration,
      },
      ...original.slice(missingMark + 2),
    ];
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 82, wpmHint: 20, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      {
        onCharacter: ({ value }) => { output += value; },
        onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
      },
    );
    let time = 0;
    for (const segment of damaged) {
      const end = time + segment.duration;
      while (time <= end) {
        decoder.process({
          timestamp: time,
          levelDb: segment.tone ? -34 : -74,
          purity: segment.tone ? 0.9 : 0.03,
          frequency: 700,
        });
        time += 5;
      }
    }
    decoder.flush(time);
    expect(output.trim()).toBe("DE F5LPY");
  });

  it("adapts toward a faster sender", () => {
    const result = decode("THE QUICK BROWN FOX", 27, 20, 0.06);
    expect(result.output).toBe("THE QUICK BROWN FOX");
    expect(result.metrics.wpm).toBeGreaterThan(24);
  });

  it("acquires slow CW from the default speed hint", () => {
    const result = decode("SLOW CODE", 12, 20, 0.05);
    expect(result.output).toBe("SLOW CODE");
    expect(result.metrics.wpm).toBeLessThan(14);
  });

  it("acquires high-speed CW from the default speed hint", () => {
    const result = decode("FAST CODE", 36, 20, 0.05);
    expect(result.output, result.decoded.join(", ")).toBe("FAST CODE");
    expect(result.metrics.wpm).toBeGreaterThan(32);
  });

  it("tracks the configured operating limits", () => {
    expect(decode("PARIS", 50, 50, 0.04).output).toBe("PARIS");
    expect(decode("QRS", 5, 5, 0.04).output).toBe("QRS");
  });

  it("does not mistake a run of dashes for slow dots", () => {
    const result = decode("TTT", 18, 18);
    expect(result.output, JSON.stringify(result.metrics)).toBe("TTT");
  });

  it("recovers systematic novice timing errors, not only random jitter", () => {
    const result = decodeTimingProfile("BAD FIST TEST", 15, {
      dot: 1.4,
      dash: 2.3,
      elementGap: 1.35,
      characterGap: 2.2,
      wordGap: 4.7,
    }, 15);
    expect(result.output, JSON.stringify(result.metrics)).toBe("BAD FIST TEST");
  });

  it("rejects loud broadband noise without tonal purity", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 90, wpmHint: 20, adaptiveSpeed: true },
      { onCharacter: ({ value }) => { output += value; } },
    );
    for (let time = 0; time < 3000; time += 5) {
      decoder.process({ timestamp: time, levelDb: -28, purity: 0.02, frequency: 700 });
    }
    decoder.flush(3100);
    expect(output).toBe("");
  });

  it("rejects a long unkeyed carrier instead of inventing a dash", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 90, wpmHint: 18, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      { onCharacter: ({ value }) => { output += value; } },
    );
    for (let time = 0; time <= 5000; time += 5) {
      decoder.process({ timestamp: time, levelDb: -18, purity: 0.98, frequency: 910 });
    }
    for (let time = 5005; time <= 6200; time += 5) {
      decoder.process({ timestamp: time, levelDb: -82, purity: 0.02, frequency: 910 });
    }
    decoder.flush(6300);
    expect(output).toBe("");
  });

  it("opens on a weak carrier only a few dB above the narrow-band floor", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 96, wpmHint: 18, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      { onCharacter: ({ value }) => { output += value; } },
    );
    const unit = 1200 / 18;
    let time = 0;
    for (const segment of signalFor("SOS", unit)) {
      const end = time + segment.duration;
      while (time <= end) {
        decoder.process({
          timestamp: time,
          levelDb: segment.tone ? -67.5 : -72,
          purity: segment.tone ? 0.19 : 0.045,
          frequency: 615,
        });
        time += 5;
      }
    }
    decoder.flush(time);
    expect(output.trim()).toBe("SOS");
  });

  it("rejects tonal codec residue between otherwise full-scale marks", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 82, wpmHint: 18, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      {
        onCharacter: ({ value }) => { output += value; },
        onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
      },
    );
    const unit = 1200 / 18;
    let time = 0;
    let carrierSeen = false;
    for (const segment of signalFor("SOS", unit)) {
      const end = time + segment.duration;
      while (time <= end) {
        if (segment.tone) carrierSeen = true;
        decoder.process({
          timestamp: time,
          levelDb: segment.tone ? -4 : carrierSeen ? -48 : -112,
          purity: segment.tone ? 0.99 : carrierSeen ? 0.62 : 0.01,
          frequency: 750,
        });
        time += 5;
      }
    }
    decoder.flush(time);
    expect(output.trim()).toBe("SOS");
  });

  it("reacquires CW after an abrupt deep QSB fade", () => {
    let output = "";
    const decoder = new AdaptiveMorseDecoder(
      { sensitivity: 82, wpmHint: 18, adaptiveSpeed: true, handKeyAssist: true, languageAssist: true },
      {
        onCharacter: ({ value }) => { output += value; },
        onWordGap: () => { if (!output.endsWith(" ")) output += " "; },
      },
    );
    const unit = 1200 / 18;
    let time = 0;
    const feed = (text: string, markLevelDb: number) => {
      for (const segment of signalFor(text, unit)) {
        const end = time + segment.duration;
        while (time <= end) {
          decoder.process({
            timestamp: time,
            levelDb: segment.tone ? markLevelDb : -82,
            purity: segment.tone ? 0.9 : 0.025,
            frequency: 700,
          });
          time += 5;
        }
      }
    };

    feed("SOS", -5);
    feed("SOS", -35);
    decoder.flush(time);

    expect(output.trim()).toBe("SOS SOS");
  });
});
