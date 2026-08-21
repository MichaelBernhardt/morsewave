#!/usr/bin/env node

import { readFileSync } from "node:fs";

const path = process.argv[2];
if (!path) throw new Error("Usage: node scripts/inspect-wav.mjs <mono-pcm16-wav> [frequency]");

const data = readFileSync(path);
const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
const sampleRate = view.getUint32(24, true);
const channels = view.getUint16(22, true);
const bits = view.getUint16(34, true);
let offset = 12;
while (offset + 8 <= data.length && data.toString("ascii", offset, offset + 4) !== "data") {
  offset += 8 + view.getUint32(offset + 4, true);
}
if (channels !== 1 || bits !== 16 || offset + 8 > data.length) {
  throw new Error("Expected a mono 16-bit PCM WAV file.");
}
const sampleCount = view.getUint32(offset + 4, true) / 2;
const samples = new Float32Array(sampleCount);
for (let index = 0; index < sampleCount; index += 1) {
  samples[index] = view.getInt16(offset + 8 + index * 2, true) / 32768;
}

const frameSize = 256;
const frequencies = process.argv[3] ? [Number(process.argv[3])] : [700, 720, 740, 750, 760];

function magnitudeAt(buffer, frequency) {
  const omega = 2 * Math.PI * frequency / sampleRate;
  const coefficient = 2 * Math.cos(omega);
  let previous = 0;
  let previousTwo = 0;
  let windowSum = 0;
  for (let index = 0; index < buffer.length; index += 1) {
    const window = 0.5 - 0.5 * Math.cos(2 * Math.PI * index / (buffer.length - 1));
    const value = buffer[index] * window + coefficient * previous - previousTwo;
    previousTwo = previous;
    previous = value;
    windowSum += window;
  }
  const real = previous - previousTwo * Math.cos(omega);
  const imaginary = previousTwo * Math.sin(omega);
  return 2 * Math.hypot(real, imaginary) / Math.max(1, windowSum);
}

function percentile(ordered, fraction) {
  return ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * fraction))];
}

for (const frequency of frequencies) {
  const levels = [];
  const purities = [];
  for (let start = 0; start + frameSize <= samples.length; start += frameSize) {
    const frame = samples.subarray(start, start + frameSize);
    const amplitude = magnitudeAt(frame, frequency);
    let squareSum = 0;
    for (const sample of frame) squareSum += sample * sample;
    const rms = Math.sqrt(squareSum / frame.length);
    levels.push(20 * Math.log10(amplitude + 1e-8));
    purities.push(Math.min(1, amplitude * amplitude / Math.max(1e-10, 2 * rms * rms)));
  }
  levels.sort((left, right) => left - right);
  purities.sort((left, right) => left - right);
  console.log(JSON.stringify({
    frequency,
    sampleRate,
    frames: levels.length,
    levelDb: [0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1].map((fraction) =>
      Math.round(percentile(levels, fraction) * 10) / 10
    ),
    purity: [0.25, 0.5, 0.75, 0.95].map((fraction) =>
      Math.round(percentile(purities, fraction) * 100) / 100
    ),
  }));
}
