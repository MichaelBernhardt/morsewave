# Real-recording benchmark corpus

The manifest deliberately mixes exact-answer material with messy off-air captures. Audio stays out of Git: licenses and source attribution remain attached to every item, and the fetcher downloads the current source files into the ignored `cache/` directory.

```sh
node scripts/fetch-real-corpus.mjs
```

Pass one or more manifest IDs to fetch only those cases. When `ffmpeg` is installed, the fetcher also creates short mono 48 kHz WAV segments so evaluations do not require playing a 30-minute source file.

```sh
node scripts/fetch-real-corpus.mjs arrl-w1aw-18wpm f5lpy-off-air
npm run corpus:evaluate -- test-corpus/cache/arrl-w1aw-18wpm-segment.wav
```

`corpus:evaluate` runs the same timing decoder with the repository's offline narrow-band analyzer and prints a JSON diagnostic report. For an end-to-end Web Audio check, start Vite and a Chromium-family browser with remote debugging, open Morsewave, then run:

```sh
npm run dev
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/morsewave-cdp http://127.0.0.1:1420/
node scripts/evaluate-browser.mjs test-corpus/cache/arrl-w1aw-18wpm-segment.wav 45
```

The W1AW sample is the primary exact regression. The off-air files are adversarial listening checks: some start mid-character, have no trustworthy transcript, or contain multiple stations, so their `knownCopy` fields are anchors rather than character-error-rate references.
