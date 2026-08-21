import { readFileSync } from "node:fs";
import { AdaptiveMorseDecoder, type SignalSample } from "../src/audio/AdaptiveMorseDecoder";

const path = process.argv[2];
if (!path) throw new Error("Usage: evaluate-wav <mono-pcm16-wav>");

const data = readFileSync(path);
const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
const sampleRate = view.getUint32(24, true);
const channels = view.getUint16(22, true);
const bits = view.getUint16(34, true);
let dataOffset = 12;
while (dataOffset + 8 <= data.length && data.toString("ascii", dataOffset, dataOffset + 4) !== "data") {
  dataOffset += 8 + view.getUint32(dataOffset + 4, true);
}
if (channels !== 1 || bits !== 16 || dataOffset + 8 > data.length) {
  throw new Error("Expected a mono 16-bit PCM WAV file.");
}
const sampleCount = view.getUint32(dataOffset + 4, true) / 2;
const samples = new Float32Array(sampleCount);
for (let index = 0; index < sampleCount; index += 1) {
  samples[index] = view.getInt16(dataOffset + 8 + index * 2, true) / 32768;
}

class OfflineAnalyzer {
  private readonly hopSize = 256;
  private hopIndex = 0;
  private readonly analysisSize = 1024;
  private readonly analysisRing = new Float32Array(this.analysisSize);
  private readonly analysisWindow = this.makeWindow(this.analysisSize);
  private readonly analysisWindowSum = this.analysisWindow.reduce((sum, value) => sum + value, 0);
  private analysisIndex = 0;
  private analysisCount = 0;
  private readonly tuneSize = 4096;
  private readonly tuneRing = new Float32Array(this.tuneSize);
  private readonly tuneWindow = this.makeWindow(this.tuneSize);
  private readonly tuneWindowSum = this.tuneWindow.reduce((sum, value) => sum + value, 0);
  private tuneIndex = 0;
  private tuneCount = 0;
  private targetFrequency = 700;
  private peakProminenceDb = 0;
  private carrierLocked = false;
  private candidateFrequency = this.targetFrequency;
  private candidateHits = 0;
  private carrierQuietScans = 0;
  private tuneLowEnvelope: Float32Array | null = null;
  private tuneHighEnvelope: Float32Array | null = null;
  private tunePreviousMagnitude: Float32Array | null = null;
  private tuneKeyingActivity: Float32Array | null = null;
  private tuneHistoryFrames = 0;

  private makeWindow(size: number) {
    return Float32Array.from({ length: size }, (_, index) =>
      0.5 - 0.5 * Math.cos(2 * Math.PI * index / (size - 1))
    );
  }

  private magnitudeAt(buffer: Float32Array, frequency: number) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const coefficient = 2 * Math.cos(omega);
    let previous = 0;
    let previousTwo = 0;
    const windowValues = buffer.length === this.tuneSize ? this.tuneWindow : this.analysisWindow;
    const windowSum = buffer.length === this.tuneSize ? this.tuneWindowSum : this.analysisWindowSum;
    for (let index = 0; index < buffer.length; index += 1) {
      const value = buffer[index] * windowValues[index] + coefficient * previous - previousTwo;
      previousTwo = previous;
      previous = value;
    }
    const real = previous - previousTwo * Math.cos(omega);
    const imaginary = previousTwo * Math.sin(omega);
    return 2 * Math.hypot(real, imaginary) / Math.max(1, windowSum);
  }

  private orderedTuneBuffer() {
    const output = new Float32Array(this.tuneSize);
    for (let index = 0; index < this.tuneSize; index += 1) {
      output[index] = this.tuneRing[(this.tuneIndex + index) % this.tuneSize];
    }
    return output;
  }

  private orderedAnalysisBuffer() {
    const output = new Float32Array(this.analysisSize);
    for (let index = 0; index < this.analysisSize; index += 1) {
      output[index] = this.analysisRing[(this.analysisIndex + index) % this.analysisSize];
    }
    return output;
  }

  private scanSpectrum() {
    if (this.tuneCount < this.tuneSize) return;
    const buffer = this.orderedTuneBuffer();
    const magnitudes: number[] = [];
    let bestMagnitude = 0;
    let bestFrequency = this.targetFrequency;
    for (let frequency = 200; frequency <= 4000; frequency += 30) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      magnitudes.push(magnitude);
      if (magnitude > bestMagnitude) {
        bestMagnitude = magnitude;
        bestFrequency = frequency;
      }
    }
    for (let frequency = Math.max(200, bestFrequency - 30); frequency <= Math.min(4000, bestFrequency + 30); frequency += 5) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      if (magnitude > bestMagnitude) {
        bestMagnitude = magnitude;
        bestFrequency = frequency;
      }
    }
    if (!this.tuneLowEnvelope || this.tuneLowEnvelope.length !== magnitudes.length) {
      this.tuneLowEnvelope = Float32Array.from(magnitudes);
      this.tuneHighEnvelope = Float32Array.from(magnitudes);
      this.tunePreviousMagnitude = Float32Array.from(magnitudes);
      this.tuneKeyingActivity = new Float32Array(magnitudes.length);
      this.tuneHistoryFrames = 1;
    } else {
      for (let index = 0; index < magnitudes.length; index += 1) {
        const magnitude = magnitudes[index];
        const low = this.tuneLowEnvelope[index];
        const high = this.tuneHighEnvelope![index];
        const previous = this.tunePreviousMagnitude![index] || 1e-8;
        const transitionDb = Math.min(32, Math.abs(20 * Math.log10((magnitude + 1e-8) / (previous + 1e-8))));
        this.tuneLowEnvelope[index] += (magnitude - low) * (magnitude < low ? 0.34 : 0.012);
        this.tuneHighEnvelope![index] += (magnitude - high) * (magnitude > high ? 0.34 : 0.022);
        this.tuneKeyingActivity![index] = this.tuneKeyingActivity![index] * 0.78 + transitionDb * 0.22;
        this.tunePreviousMagnitude![index] = magnitude;
      }
      this.tuneHistoryFrames += 1;
    }

    let bestIndex = Math.round((bestFrequency - 200) / 30);
    let bestSelectionScoreDb = -Infinity;
    let bestModulationDb = 0;
    if (this.tuneHistoryFrames >= 4) {
      for (let index = 0; index < magnitudes.length; index += 1) {
        const peak = this.tuneHighEnvelope[index] || 1e-8;
        const peakDb = 20 * Math.log10(peak + 1e-8);
        const modulationDb = this.tuneKeyingActivity[index] || 0;
        const binFrequency = 200 + index * 30;
        const acquisitionPenaltyDb = this.carrierLocked
          ? 0
          : Math.min(16, Math.abs(binFrequency - this.targetFrequency) / 80);
        const scoreDb = peakDb + Math.min(26, Math.max(0, modulationDb - 1.5) * 1.35) -
          acquisitionPenaltyDb;
        if (scoreDb > bestSelectionScoreDb) {
          bestSelectionScoreDb = scoreDb;
          bestIndex = index;
          bestModulationDb = modulationDb;
        }
      }
      bestFrequency = 200 + bestIndex * 30;
      bestMagnitude = magnitudes[bestIndex] || 0;
      const selectedFloor = this.tuneLowEnvelope[bestIndex] || 1e-8;
      if (bestMagnitude > selectedFloor * 1.45 || bestModulationDb < 3) {
        for (let frequency = Math.max(200, bestFrequency - 30); frequency <= Math.min(4000, bestFrequency + 30); frequency += 5) {
          const magnitude = this.magnitudeAt(buffer, frequency);
          if (magnitude > bestMagnitude) {
            bestMagnitude = magnitude;
            bestFrequency = frequency;
          }
        }
      }
    }

    const ordered = [...magnitudes].sort((left, right) => left - right);
    const baseline = ordered[Math.floor(ordered.length * 0.58)] || 1e-8;
    const peakEnvelope = this.tuneHighEnvelope?.[bestIndex] || bestMagnitude;
    const orderedPeaks = this.tuneHighEnvelope ? [...this.tuneHighEnvelope].sort((left, right) => left - right) : ordered;
    const peakBaseline = orderedPeaks[Math.floor(orderedPeaks.length * 0.58)] || baseline;
    const globalProminenceDb = 20 * Math.log10((peakEnvelope + 1e-8) / (peakBaseline + 1e-8));
    const bestDb = 20 * Math.log10(peakEnvelope + 1e-8);

    let localMagnitude = 0;
    let localFrequency = this.targetFrequency;
    for (let frequency = Math.max(200, this.targetFrequency - 70); frequency <= Math.min(4000, this.targetFrequency + 70); frequency += 10) {
      const magnitude = this.magnitudeAt(buffer, frequency);
      if (magnitude > localMagnitude) {
        localMagnitude = magnitude;
        localFrequency = frequency;
      }
    }
    const localProminenceDb = 20 * Math.log10((localMagnitude + 1e-8) / (baseline + 1e-8));
    const localDb = 20 * Math.log10(localMagnitude + 1e-8);
    const localCarrier = this.carrierLocked && localProminenceDb > 8 && localDb > -84;
    const localIndex = Math.max(0, Math.min(magnitudes.length - 1, Math.round((this.targetFrequency - 200) / 30)));
    const localPeak = this.tuneHighEnvelope?.[localIndex] || localMagnitude;
    const localModulationDb = this.tuneKeyingActivity?.[localIndex] || 0;
    const localSelectionScoreDb = 20 * Math.log10(localPeak + 1e-8) +
      Math.min(26, Math.max(0, localModulationDb - 1.5) * 1.35);
    if (!this.carrierLocked || localModulationDb > 2.4) {
      this.carrierQuietScans = 0;
    } else {
      this.carrierQuietScans += 1;
    }
    const releaseScans = Math.max(6, Math.ceil(3.5 * sampleRate / (this.hopSize * 9)));
    const keyedAlternative = this.tuneHistoryFrames >= 4 && bestModulationDb > 5.5 &&
      localModulationDb < 3.2 &&
      Math.abs(bestFrequency - this.targetFrequency) > 70 &&
      bestSelectionScoreDb > localSelectionScoreDb + 2.5;
    const strongCarrier = globalProminenceDb > 13 && bestDb > -84;
    const moderateCarrier = globalProminenceDb > 8 && bestDb > -54;
    if (localCarrier && !keyedAlternative) {
      this.targetFrequency += (localFrequency - this.targetFrequency) * 0.28;
      this.candidateHits = 0;
      this.peakProminenceDb = localProminenceDb;
    } else if (strongCarrier || moderateCarrier) {
      if (Math.abs(bestFrequency - this.candidateFrequency) <= 35) {
        this.candidateFrequency += (bestFrequency - this.candidateFrequency) * 0.45;
        this.candidateHits += strongCarrier ? 2 : 1;
      } else {
        this.candidateFrequency = bestFrequency;
        this.candidateHits = strongCarrier ? 2 : 1;
      }
      const canAcquire = !this.carrierLocked && this.tuneHistoryFrames >= 4 &&
        this.candidateHits >= 4;
      const nearbyCandidate = Math.abs(this.candidateFrequency - this.targetFrequency) <= 90;
      const canReplace = this.carrierLocked && strongCarrier && this.candidateHits >= 6 &&
        (nearbyCandidate || this.carrierQuietScans >= releaseScans);
      if (canAcquire || canReplace) {
        this.targetFrequency = this.candidateFrequency;
        this.carrierLocked = true;
        this.candidateHits = 0;
        this.carrierQuietScans = 0;
      }
      this.peakProminenceDb = globalProminenceDb;
    } else {
      this.candidateHits = Math.max(0, this.candidateHits - 1);
      this.peakProminenceDb = this.carrierLocked ? localProminenceDb : globalProminenceDb;
    }
    this.targetFrequency = Math.max(200, Math.min(4000, this.targetFrequency));
  }

  private analyzeFrame(buffer: Float32Array, timestamp: number): SignalSample {
    const amplitude = this.magnitudeAt(buffer, this.targetFrequency);
    let squareSum = 0;
    for (const sample of buffer) squareSum += sample * sample;
    const rms = Math.sqrt(squareSum / buffer.length);
    return {
      timestamp,
      levelDb: Math.max(-120, 20 * Math.log10(amplitude + 1e-8)),
      purity: Math.max(0, Math.min(1, amplitude * amplitude / Math.max(1e-10, 2 * rms * rms))),
      frequency: this.targetFrequency,
      prominenceDb: this.peakProminenceDb,
    };
  }

  process(callback: (sample: SignalSample) => void) {
    let frameCount = 0;
    const trace: Array<{ second: number; frequency: number; prominenceDb: number }> = [];
    let tracedSecond = -1;
    for (let inputIndex = 0; inputIndex < samples.length; inputIndex += 1) {
      const sample = samples[inputIndex];
      this.analysisRing[this.analysisIndex] = sample;
      this.analysisIndex = (this.analysisIndex + 1) % this.analysisSize;
      this.analysisCount = Math.min(this.analysisSize, this.analysisCount + 1);
      this.hopIndex += 1;
      this.tuneRing[this.tuneIndex] = sample;
      this.tuneIndex = (this.tuneIndex + 1) % this.tuneSize;
      this.tuneCount = Math.min(this.tuneSize, this.tuneCount + 1);
      if (this.hopIndex !== this.hopSize) continue;
      this.hopIndex = 0;
      if (this.analysisCount < this.analysisSize) continue;
      frameCount += 1;
      if (frameCount === 1 || frameCount % 9 === 0) this.scanSpectrum();
      const timestamp = inputIndex / sampleRate * 1000;
      callback(this.analyzeFrame(this.orderedAnalysisBuffer(), timestamp));
      const second = Math.floor(timestamp / 1000);
      if (second !== tracedSecond) {
        tracedSecond = second;
        trace.push({ second, frequency: Math.round(this.targetFrequency), prominenceDb: Math.round(this.peakProminenceDb) });
      }
    }
    return { frequency: this.targetFrequency, prominenceDb: this.peakProminenceDb, trace };
  }
}

let transcript = "";
const characters: Array<{ value: string; pattern: string; timestamp: number }> = [];
const decoder = new AdaptiveMorseDecoder({
  sensitivity: 82,
  wpmHint: 20,
  adaptiveSpeed: true,
  handKeyAssist: !process.argv.includes("--edge"),
  languageAssist: true,
}, {
  onCharacter: (character) => {
    transcript += character.value;
    characters.push({ value: character.value, pattern: character.pattern, timestamp: Math.round(character.timestamp) });
  },
  onWordGap: () => { if (transcript && !transcript.endsWith(" ")) transcript += " "; },
});

const edges: Array<{ keyed: boolean; timestamp: number; duration: number }> = [];
const frameLevels: number[] = [];
const frameThresholds: number[] = [];
const framePurities: number[] = [];
const keyedLevels: number[] = [];
const quietLevels: number[] = [];
let previousGate = false;
let previousEdge = 0;
let lastSample: SignalSample | undefined;
const decoderTrace: Array<{ second: number; wpm: number; thresholdDb: number }> = [];
let tracedDecoderSecond = -1;
const analyzer = new OfflineAnalyzer();
const tuner = analyzer.process((sample) => {
  lastSample = sample;
  const metrics = decoder.process(sample);
  frameLevels.push(sample.levelDb);
  frameThresholds.push(metrics.thresholdDb);
  framePurities.push(sample.purity);
  const second = Math.floor(sample.timestamp / 1000);
  if (second !== tracedDecoderSecond) {
    tracedDecoderSecond = second;
    decoderTrace.push({ second, wpm: Number(metrics.wpm.toFixed(1)), thresholdDb: Math.round(metrics.thresholdDb) });
  }
  (metrics.gateOpen ? keyedLevels : quietLevels).push(sample.levelDb);
  if (metrics.gateOpen !== previousGate) {
    edges.push({ keyed: previousGate, timestamp: sample.timestamp, duration: sample.timestamp - previousEdge });
    previousEdge = sample.timestamp;
    previousGate = metrics.gateOpen;
  }
});
decoder.flush(lastSample?.timestamp);

const marks = edges.filter((edge) => edge.keyed).map((edge) => Math.round(edge.duration));
const gaps = edges.filter((edge) => !edge.keyed && edge.timestamp > 100).map((edge) => Math.round(edge.duration));
const ordered = (values: number[]) => [...values].sort((left, right) => left - right);
const quantiles = (values: number[]) => {
  const sorted = ordered(values);
  return [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1].map((fraction) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]);
};

console.log(JSON.stringify({
  path,
  seconds: samples.length / sampleRate,
  transcript: transcript.trim(),
  characters,
  tuner,
  decoderTrace,
  metrics: decoder.getMetrics(),
  frontEnd: {
    levelDb: quantiles(frameLevels).map((value) => Math.round(value)),
    thresholdDb: quantiles(frameThresholds).map((value) => Math.round(value)),
    purity: quantiles(framePurities).map((value) => Number(value.toFixed(3))),
    keyedLevelDb: quantiles(keyedLevels).map((value) => Math.round(value)),
    quietLevelDb: quantiles(quietLevels).map((value) => Math.round(value)),
  },
  edges,
  marks: { count: marks.length, quantiles: quantiles(marks), first: marks.slice(0, 60) },
  gaps: { count: gaps.length, quantiles: quantiles(gaps), first: gaps.slice(0, 60) },
}, null, 2));
