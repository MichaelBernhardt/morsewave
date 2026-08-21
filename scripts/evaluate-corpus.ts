#!/usr/bin/env node

import { existsSync, readFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { optimizeMorseRecording } from "../src/audio/offline/optimizerCore";
import {
  assessCorpusCopy,
  type CorpusAssessment,
  type CorpusExpectations,
} from "./corpus-metrics";

interface CorpusSample {
  readonly id: string;
  readonly title: string;
  readonly file: string;
  readonly segment?: { readonly start: number; readonly duration: number };
  readonly expect?: CorpusExpectations;
  readonly decoder?: {
    readonly wpmHint?: number;
    readonly languageAssist?: boolean;
    readonly minimumFrequency?: number;
    readonly maximumFrequency?: number;
  };
}

interface CorpusManifest {
  readonly version: number;
  readonly samples: readonly CorpusSample[];
}

interface EvaluationRow {
  readonly id: string;
  readonly title: string;
  readonly file: string;
  readonly elapsedMs?: number;
  readonly rawTranscript?: string;
  readonly displayTranscript?: string;
  readonly frequency?: number;
  readonly fundamentalHz?: number;
  readonly wpm?: number;
  readonly score?: number;
  readonly confidence?: number;
  readonly converged?: boolean;
  readonly assessment?: CorpusAssessment;
  readonly error?: string;
  readonly missing?: boolean;
}

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = resolve(repository, "test-corpus/manifest.json");
const cachePath = resolve(repository, "test-corpus/cache");

function ascii(data: Buffer, start: number, length: number) {
  return data.toString("ascii", start, start + length);
}

function readPcm16Wav(path: string) {
  const data = readFileSync(path);
  if (ascii(data, 0, 4) !== "RIFF" || ascii(data, 8, 4) !== "WAVE") {
    throw new Error("expected a RIFF/WAVE recording");
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let formatTag = 0;
  let sourceRate = 0;
  let channelCount = 0;
  let bitsPerSample = 0;
  let blockAlign = 0;
  let dataOffset = -1;
  let dataBytes = 0;
  for (let offset = 12; offset + 8 <= data.length;) {
    const id = ascii(data, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const contents = offset + 8;
    if (contents + size > data.length) throw new Error(`truncated ${id} chunk`);
    if (id === "fmt " && size >= 16) {
      formatTag = view.getUint16(contents, true);
      channelCount = view.getUint16(contents + 2, true);
      sourceRate = view.getUint32(contents + 4, true);
      blockAlign = view.getUint16(contents + 12, true);
      bitsPerSample = view.getUint16(contents + 14, true);
    } else if (id === "data") {
      dataOffset = contents;
      dataBytes = size;
    }
    offset = contents + size + (size & 1);
  }
  if (formatTag !== 1 || bitsPerSample !== 16 || channelCount < 1 || dataOffset < 0 || !sourceRate) {
    throw new Error("expected uncompressed 16-bit PCM WAV audio");
  }
  const sourceFrames = Math.floor(dataBytes / Math.max(blockAlign, channelCount * 2));
  if (!sourceFrames) throw new Error("recording has no PCM frames");
  const sourceChannels = Array.from(
    { length: Math.min(2, channelCount) },
    () => new Float32Array(sourceFrames),
  );
  for (let frame = 0; frame < sourceFrames; frame += 1) {
    for (let channel = 0; channel < sourceChannels.length; channel += 1) {
      sourceChannels[channel][frame] = view.getInt16(
        dataOffset + frame * blockAlign + channel * 2,
        true,
      ) / 32768;
    }
  }

  const analysisRate = Math.min(12_000, sourceRate);
  if (analysisRate === sourceRate) return { channels: sourceChannels, sampleRate: analysisRate };
  const channels = sourceChannels.map((source) => {
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
  return { channels, sampleRate: analysisRate };
}

function segmentFile(sample: CorpusSample) {
  return resolve(cachePath, sample.segment ? `${sample.id}-segment.wav` : sample.file);
}

async function evaluate(sample: CorpusSample): Promise<EvaluationRow> {
  const file = segmentFile(sample);
  if (!existsSync(file)) {
    return { id: sample.id, title: sample.title, file, missing: true,
      error: `missing cached audio; run npm run corpus:fetch -- ${sample.id}` };
  }
  try {
    const { channels, sampleRate } = readPcm16Wav(file);
    const started = performance.now();
    const result = optimizeMorseRecording(channels, sampleRate, {
      wpmHint: sample.decoder?.wpmHint ?? 20,
      languageAssist: sample.decoder?.languageAssist ?? true,
      minimumFrequency: sample.decoder?.minimumFrequency ?? 200,
      maximumFrequency: Math.min(
        sample.decoder?.maximumFrequency ?? 4000,
        sampleRate * 0.45,
      ),
    });
    const assessment = assessCorpusCopy(sample.expect ?? {}, {
      rawTranscript: result.rawTranscript,
      frequency: result.frequency,
      fundamentalHz: result.fundamentalHz,
      wpm: result.wpm,
      converged: result.converged,
    });
    return {
      id: sample.id,
      title: sample.title,
      file,
      elapsedMs: Math.round(performance.now() - started),
      rawTranscript: result.rawTranscript,
      displayTranscript: result.transcript,
      frequency: result.frequency,
      fundamentalHz: result.fundamentalHz,
      wpm: result.wpm,
      score: result.score,
      confidence: result.confidence,
      converged: result.converged,
      assessment,
    };
  } catch (error) {
    return { id: sample.id, title: sample.title, file,
      error: error instanceof Error ? error.message : String(error) };
  }
}

function fixed(value: number | undefined, digits = 1) {
  return value === undefined ? "—" : value.toFixed(digits);
}

function checkMark(row: EvaluationRow, name: string) {
  const check = row.assessment?.checks.find((candidate) => candidate.name === name);
  return check ? (check.passed ? "✓" : "!") : "";
}

function matchCell(metric: CorpusAssessment["anchors"] | undefined) {
  return !metric?.total ? "—" : `${metric.matched}/${metric.total}`;
}

function clipped(value: string | undefined, limit: number) {
  if (!value) return "—";
  const compact = value.replace(/\s+/g, " ");
  return compact.length <= limit ? compact : `${compact.slice(0, limit - 1)}…`;
}

function printTable(rows: readonly EvaluationRow[]) {
  const cells = rows.map((row) => {
    const copy = row.assessment?.copy;
    const status = row.missing ? "MISS" : row.error ? "ERR" :
      row.assessment?.checks.length ? row.assessment.passed ? "PASS" : "FAIL" : "INFO";
    return [
      row.id,
      status,
      copy ? fixed(copy.characters.errorRate * 100) : "—",
      copy ? fixed(copy.words.errorRate * 100) : "—",
      copy ? fixed(copy.spacing.f1, 2) : "—",
      matchCell(row.assessment?.anchors),
      matchCell(row.assessment?.keywords),
      row.frequency === undefined
        ? "—"
        : `${Math.round(row.frequency)}${row.fundamentalHz === undefined
          ? ""
          : `→${Math.round(row.fundamentalHz)}`}${checkMark(row, "carrierHz")}`,
      row.wpm === undefined ? "—" : `${fixed(row.wpm)}${checkMark(row, "wpm")}`,
      row.converged === undefined ? "—" : `${row.converged ? "Y" : "N"}${checkMark(row, "converged")}`,
      clipped(row.rawTranscript, 44),
    ];
  });
  const headings = ["SAMPLE", "STATUS", "CER%", "WER%", "SP-F1", "ANCH", "KEY", "HZ", "WPM", "CONV", "RAW COPY"];
  const widths = headings.map((heading, index) => Math.max(
    heading.length,
    ...cells.map((row) => row[index].length),
  ));
  const line = (row: readonly string[]) => row.map((value, index) =>
    value.padEnd(widths[index])
  ).join("  ");
  console.log(line(headings));
  console.log(line(widths.map((width) => "-".repeat(width))));
  for (const row of cells) console.log(line(row));

  const failed = rows.filter((row) => row.error || row.assessment?.passed === false);
  for (const row of failed) {
    if (row.error) {
      console.log(`\n${row.id}: ${row.error}`);
      continue;
    }
    const checks = row.assessment?.checks.filter((check) => !check.passed) ?? [];
    console.log(`\n${row.id}: ${checks.map((check) =>
      `${check.name}=${String(check.actual)} (expected ${String(check.expected)})`
    ).join("; ")}`);
    const missingAnchors = row.assessment?.anchors.results.filter((item) => !item.matched).map((item) => item.value) ?? [];
    const missingKeywords = row.assessment?.keywords.results.filter((item) => !item.matched).map((item) => item.value) ?? [];
    if (missingAnchors.length) console.log(`  missing anchors: ${missingAnchors.join(" | ")}`);
    if (missingKeywords.length) console.log(`  missing keywords: ${missingKeywords.join(", ")}`);
  }

  const passed = rows.filter((row) => row.assessment?.checks.length && row.assessment.passed).length;
  const informational = rows.filter((row) => !row.error && !row.assessment?.checks.length).length;
  console.log(`\n${passed}/${rows.length} asserted samples passed` +
    `${failed.length ? `; ${failed.length} failed/missing` : ""}` +
    `${informational ? `; ${informational} informational` : ""}.`);
}

function usage() {
  console.log("Usage: npm run corpus:score -- [sample-id ...] [--json] [--no-fail] [--list]");
}

function validateManifest(manifest: CorpusManifest) {
  if (manifest.version !== 2 || !Array.isArray(manifest.samples)) {
    throw new Error("Expected corpus manifest version 2.");
  }
  const ids = new Set<string>();
  for (const sample of manifest.samples) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(sample.id) || ids.has(sample.id)) {
      throw new Error(`Invalid or duplicate corpus sample ID: ${sample.id}`);
    }
    ids.add(sample.id);
    if (!sample.title || !sample.file || basename(sample.file) !== sample.file) {
      throw new Error(`Invalid title or cache filename for ${sample.id}.`);
    }
    for (const [name, values] of Object.entries({
      anchors: sample.expect?.anchors,
      keywords: sample.expect?.keywords,
    })) {
      if (values !== undefined && (!Array.isArray(values) ||
        values.some((value) => typeof value !== "string" || !value.trim()))) {
        throw new Error(`Invalid ${name} for ${sample.id}.`);
      }
    }
    for (const [name, numeric] of Object.entries({
      carrierHz: sample.expect?.carrierHz,
      wpm: sample.expect?.wpm,
    })) {
      if (!numeric) continue;
      const ranges = numeric.anyOf ?? [numeric];
      if (!ranges.length || numeric.anyOf &&
        [numeric.target, numeric.tolerance, numeric.min, numeric.max].some((value) => value !== undefined)) {
        throw new Error(`Invalid ${name} alternatives for ${sample.id}.`);
      }
      for (const range of ranges) {
        const values = [range.target, range.tolerance, range.min, range.max]
          .filter((value) => value !== undefined);
        if (!values.length || values.some((value) => !Number.isFinite(value)) ||
          (range.tolerance !== undefined && range.tolerance < 0) ||
          (range.min !== undefined && range.max !== undefined && range.min > range.max)) {
          throw new Error(`Invalid ${name} expectation for ${sample.id}.`);
        }
      }
    }
  }
}

const arguments_ = process.argv.slice(2);
if (arguments_.includes("--help")) {
  usage();
  process.exit(0);
}
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as CorpusManifest;
validateManifest(manifest);
const ids = arguments_.filter((argument) => !argument.startsWith("--"));
const unknownFlags = arguments_.filter((argument) => argument.startsWith("--") &&
  !["--json", "--no-fail", "--list"].includes(argument));
if (unknownFlags.length) throw new Error(`Unknown option: ${unknownFlags.join(", ")}`);
const uniqueIds = new Set(manifest.samples.map((sample) => sample.id));
const unknownIds = ids.filter((id) => !uniqueIds.has(id));
if (unknownIds.length) throw new Error(`Unknown sample ID: ${unknownIds.join(", ")}`);
if (arguments_.includes("--list")) {
  for (const sample of manifest.samples) console.log(`${sample.id}\t${sample.title}`);
  process.exit(0);
}
const selected = ids.length
  ? manifest.samples.filter((sample) => ids.includes(sample.id))
  : manifest.samples;
const rows: EvaluationRow[] = [];
for (const sample of selected) rows.push(await evaluate(sample));

if (arguments_.includes("--json")) {
  console.log(JSON.stringify({ manifestVersion: manifest.version, rows }, null, 2));
} else {
  printTable(rows);
}
if (!arguments_.includes("--no-fail") && rows.some((row) => row.error || row.assessment?.passed === false)) {
  process.exitCode = rows.some((row) => row.missing) ? 2 : 1;
}
