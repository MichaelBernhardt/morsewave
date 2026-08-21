import { readFileSync } from "node:fs";
import { optimizeMorseRecording } from "../src/audio/offline/optimizerCore";

const path = process.argv[2];
if (!path) throw new Error("Usage: optimize-wav <pcm16-wav>");

const data = readFileSync(path);
const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
const sourceRate = view.getUint32(24, true);
const channelCount = view.getUint16(22, true);
const bits = view.getUint16(34, true);
let dataOffset = 12;
while (dataOffset + 8 <= data.length && data.toString("ascii", dataOffset, dataOffset + 4) !== "data") {
  dataOffset += 8 + view.getUint32(dataOffset + 4, true);
}
if (bits !== 16 || channelCount < 1 || dataOffset + 8 > data.length) {
  throw new Error("Expected an uncompressed 16-bit PCM WAV file.");
}
const interleavedSamples = view.getUint32(dataOffset + 4, true) / 2;
const sourceFrames = Math.floor(interleavedSamples / channelCount);
const sourceChannels = Array.from({ length: Math.min(2, channelCount) }, () => new Float32Array(sourceFrames));
for (let frame = 0; frame < sourceFrames; frame += 1) {
  for (let channel = 0; channel < sourceChannels.length; channel += 1) {
    sourceChannels[channel][frame] = view.getInt16(
      dataOffset + 8 + (frame * channelCount + channel) * 2,
      true,
    ) / 32768;
  }
}

const analysisRate = Math.min(12_000, sourceRate);
const channels = sourceRate === analysisRate
  ? sourceChannels
  : sourceChannels.map((source) => {
    const output = new Float32Array(Math.ceil(source.length * analysisRate / sourceRate));
    for (let index = 0; index < output.length; index += 1) {
      const position = index * sourceRate / analysisRate;
      const left = Math.min(source.length - 1, Math.floor(position));
      const right = Math.min(source.length - 1, left + 1);
      const fraction = position - left;
      output[index] = source[left] + (source[right] - source[left]) * fraction;
    }
    return output;
  });

const started = performance.now();
const result = optimizeMorseRecording(channels, analysisRate, {
  wpmHint: Number(process.env.MORSEWAVE_WPM || 20),
  languageAssist: process.env.MORSEWAVE_LANGUAGE !== "0",
  minimumFrequency: 200,
  maximumFrequency: Math.min(4000, analysisRate * 0.45),
}, (progress) => {
  if (process.env.MORSEWAVE_PROGRESS === "1") process.stderr.write(`${progress.detail}\n`);
});

console.log(JSON.stringify({
  path,
  analyzedSampleRate: analysisRate,
  elapsedMs: Math.round(performance.now() - started),
  ...result,
}, null, 2));
