#!/usr/bin/env node

import { createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { basename, dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = resolve(repository, "test-corpus/manifest.json");
const cache = resolve(repository, "test-corpus/cache");
const requested = new Set(process.argv.slice(2));
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const samples = requested.size
  ? manifest.samples.filter((sample) => requested.has(sample.id))
  : manifest.samples;

if (!samples.length) {
  throw new Error(`No matching samples. Available IDs: ${manifest.samples.map((sample) => sample.id).join(", ")}`);
}

mkdirSync(cache, { recursive: true });

async function download(url, destination) {
  if (existsSync(destination) && statSync(destination).size > 0) {
    console.log(`cached   ${basename(destination)}`);
    return;
  }
  const temporary = `${destination}.part`;
  console.log(`fetching ${url}`);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
  renameSync(temporary, destination);
}

const ffmpeg = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" }).status === 0;

for (const sample of samples) {
  const audioPath = resolve(cache, sample.file);
  await download(sample.audioUrl, audioPath);
  if (sample.transcriptUrl) {
    await download(sample.transcriptUrl, resolve(cache, `${sample.id}.txt`));
  }
  if (ffmpeg && sample.segment) {
    const segmentPath = resolve(cache, `${sample.id}-segment.wav`);
    const result = spawnSync("ffmpeg", [
      "-hide_banner", "-loglevel", "error", "-y",
      "-ss", String(sample.segment.start),
      "-t", String(sample.segment.duration),
      "-i", audioPath,
      "-ac", "1", "-ar", "48000", segmentPath,
    ], { stdio: "inherit" });
    if (result.status !== 0) throw new Error(`ffmpeg failed for ${sample.id}`);
    console.log(`segment  ${basename(segmentPath)}`);
  }
}

if (!ffmpeg) console.log("ffmpeg was not found; original recordings were downloaded without benchmark segments.");
console.log(`Corpus ready in ${cache}`);
