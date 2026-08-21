import { encodeText } from "./morse";

interface DemoOptions {
  text?: string;
  wpm?: number;
  frequency?: number;
}
export function createDemoBuffer(
  context: BaseAudioContext,
  { text = "CQ CQ DE MORSEWAVE TEST 599 K", wpm = 19, frequency = 692 }: DemoOptions = {},
) {
  const unitSeconds = 1.2 / wpm;
  const words = encodeText(text);
  const segments: Array<{ tone: boolean; units: number }> = [{ tone: false, units: 3 }];
  words.forEach((word, wordIndex) => {
    word.forEach((pattern, characterIndex) => {
      [...pattern].forEach((symbol, symbolIndex) => {
        segments.push({ tone: true, units: symbol === "." ? 1 : 3 });
        if (symbolIndex < pattern.length - 1) segments.push({ tone: false, units: 1 });
      });
      if (characterIndex < word.length - 1) segments.push({ tone: false, units: 3 });
    });
    if (wordIndex < words.length - 1) segments.push({ tone: false, units: 7 });
  });
  segments.push({ tone: false, units: 7 });

  const totalSeconds = segments.reduce((sum, segment) => sum + segment.units * unitSeconds, 0);
  const buffer = context.createBuffer(1, Math.ceil(totalSeconds * context.sampleRate), context.sampleRate);
  const channel = buffer.getChannelData(0);
  let cursor = 0;
  let phase = 0;
  let randomState = 1977;
  const random = () => {
    randomState = randomState * 48271 % 2147483647;
    return (randomState / 2147483647) * 2 - 1;
  };

  segments.forEach((segment, segmentIndex) => {
    // Mild human-like timing variation makes the demo exercise adaptation.
    const variation = 1 + Math.sin(segmentIndex * 1.73) * 0.035;
    const sampleCount = Math.round(segment.units * unitSeconds * variation * context.sampleRate);
    const edgeSamples = Math.max(1, Math.round(context.sampleRate * 0.004));
    for (let index = 0; index < sampleCount && cursor < channel.length; index += 1) {
      const driftedFrequency = frequency + Math.sin(cursor / context.sampleRate * 0.7) * 2.2;
      phase += 2 * Math.PI * driftedFrequency / context.sampleRate;
      const edge = Math.min(1, index / edgeSamples, (sampleCount - index - 1) / edgeSamples);
      const tone = segment.tone ? Math.sin(phase) * 0.31 * Math.max(0, edge) : 0;
      const noise = random() * 0.011 + Math.sin(phase * 0.37) * 0.002;
      channel[cursor] = tone + noise;
      cursor += 1;
    }
  });
  return buffer;
}
