# Morsewave

Morsewave is an open-source, private desktop CW/Morse decoder. It listens to a microphone, opens a recording, or generates a known practice signal, then performs all signal analysis and decoding on the device.

## What it does

- Tracks CW carriers from 200–4000 Hz with coarse/fine narrow-band Goertzel analysis
- Ranks carriers by keyed modulation as well as raw strength, reducing lock-on to steady heterodynes
- Holds tone lock through silence instead of chasing noise peaks
- Learns keying speed and timing continuously from 5–50 WPM
- Uses a 64-path timing beam to recover uneven hand-keying, short fades, bounce, and missing elements
- Uses adaptive noise-floor, spectral-purity, hysteresis, and debounce gates, with deep-QSB reacquisition
- Decodes International Morse letters, numerals, punctuation, and common prosigns such as `<BT>`, `<AR>`, `<SK>`, `<KN>`, and `<SOS>`
- Keeps the received ham shorthand intact while showing a separate advisory expansion and marker legend
- Shows live spectrum history, SNR, tone prominence, WPM, gate state, and confidence
- Accepts microphone input, drag-and-drop audio files, and a noisy built-in demo
- Runs uploaded recordings through a local multipass solver that tests several carriers, channels, gates, speeds, and spacing models before selecting a consensus transcript
- Separates simultaneous keyed stations, rewards sustained readable-message evidence, and searches compressed beginner word gaps without altering the transmitted Morse characters
- Copies or exports session transcripts with uncertain characters highlighted
- Persists decoder preferences and supports dark/light appearances
- Processes audio locally; there is no server, analytics client, or network upload

## Run it

Requirements: Node.js 22+, Rust 1.80+, and the platform prerequisites for [Tauri 2](https://v2.tauri.app/start/prerequisites/).

```bash
npm install
npm run tauri dev
```

For the browser UI alone:

```bash
npm run dev
```

## Verify and package

```bash
npm run check
npm run tauri build
```

Native installers are written under `src-tauri/target/release/bundle/`. A debug-packaged application can be produced with `npm run tauri build -- --debug`.

The local macOS build includes the microphone entitlement and is ad-hoc signed. Public macOS releases still need a Developer ID signature and Apple notarization to avoid Gatekeeper warnings.

## Signal path

Audio enters an `AudioWorklet`, where an overlapping 1024-sample detector advances in 256-sample hops. A 4096-sample rolling window scans the CW passband for a credible carrier and holds that carrier through key-up intervals. The stateful decoder then applies a two-sided adaptive gate, edge debounce, robust speed estimation, and a bounded timing beam before International Morse lookup. Only throttled measurements and decoded characters reach React for display.

Uploaded recordings additionally take a file-only optimization path before playback. A worker resamples one or two channels to 12 kHz, surveys multiple independent carrier tracks, refines their frequencies, and extracts coherent 24 ms envelopes. It reuses those envelopes across thousands of bounded gate, timing, speed, and spacing hypotheses; ranks them by Morse timing, validity, speed, message completeness, and language evidence; then combines credible passes into a timestamped consensus. Weak or inconsistent tracks automatically receive two additional high-threshold recovery bands, while clean tracks stop after the normal sweep. Recovery passes carry a reliability cost so they must win decisively rather than displace clean copy by a marginal score.

Compressed 3.4–7.4-unit hand-keyed word gaps are evaluated as competing hypotheses when language assistance is enabled. A separate decoder-quality guard prevents a long noise track from winning on text shape alone. If contextual Multipass copy looks useful but fails that physical guard, the app can expose it as an explicitly unselected candidate while the safer streaming decoder remains active. The live microphone path remains low-latency and does not wait for this search.

The optional **CW / ham prior** recognizes standard Q-signals, common amateur abbreviations, callsign shapes, and run-together procedural signs. Multipass also reads coherent operating sequences—such as `CQ … DE <callsign>` or `UR RST 5NN`—as a capped tie-breaker between otherwise credible physical hypotheses. That lexical hint cannot satisfy the independent signal/timing convergence gate. The received copy remains untouched; a separate, visibly non-authoritative operator aid expands exact markers such as `5NN` to `599` while preserving callsigns and unknown words.

In displayed copy, ambiguous continuous groups use common radio notation—for example, `-...-` is shown as `<BT>` while its raw International Morse value remains `=`. Separately keyed letters are not silently converted into run-together prosigns. Disable the prior to retain generic punctuation notation and hide the semantic aid.

The timing decoder is intentionally independent of browser APIs. Its test suite covers canonical copy, randomized and systematic novice timing errors, key bounce, deep fades, stuck carriers, weak signals, speed-harmonic ambiguity, and the full configured 5–50 WPM range.

## Real-recording corpus

The reproducible benchmark manifest includes official W1AW exact-answer audio plus licensed off-air captures with QSB, interference, static, clipped characters, competing carriers, and dirty-key chatter. It also includes a 1890 wax-cylinder hand-key recording, deeply notched Canadian amateur CW, and the CC0 SS Marine Electric distress recording. Audio is fetched into an ignored cache so provenance and license terms stay explicit without vendoring recordings.

```bash
npm run corpus:fetch -- arrl-w1aw-18wpm f5lpy-off-air
npm run corpus:evaluate -- test-corpus/cache/arrl-w1aw-18wpm-segment.wav
npm run corpus:optimize -- test-corpus/cache/arrl-w1aw-18wpm-segment.wav
```

See [`test-corpus/README.md`](test-corpus/README.md) for the full corpus and browser-level evaluation runner.

## Useful controls

- `Space`: start or stop microphone decoding
- `⌘/Ctrl + O`: open an audio recording
- `⌘/Ctrl + K`: clear the current copy
- `Esc`: stop decoding

Automatic tone lock and adaptive speed are good defaults. In a crowded passband, disable automatic tone lock and set the center frequency manually. For a weak but narrow carrier, raise sensitivity; for broadband false triggers, lower it.

Audio-file support follows the codecs available in the operating system WebView. WAV, MP3, M4A/AAC, and AIFF are broadly supported; additional formats vary by platform.

## Contributing

Issues and pull requests are welcome, especially reproducible recordings that expose a decoder failure. Please include the source/license, an expected transcript when known, the relevant time range, and receiver conditions. Do not commit downloaded corpus audio; `test-corpus/cache/` is intentionally ignored.

## License

Morsewave is available under the [MIT License](LICENSE). Corpus recordings retain the separate licenses recorded in `test-corpus/manifest.json` and are not part of the source distribution.
