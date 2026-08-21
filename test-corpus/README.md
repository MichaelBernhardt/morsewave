# Real-recording benchmark corpus

The manifest deliberately mixes exact-answer material with messy off-air captures. Audio stays out of Git: licenses and source attribution remain attached to every item, and the fetcher downloads the current source files into the ignored `cache/` directory.

```sh
node scripts/fetch-real-corpus.mjs
```

Pass one or more manifest IDs to fetch only those cases. When `ffmpeg` is installed, the fetcher also creates short source-channel 48 kHz WAV segments so evaluations preserve diversity paths without requiring a 30-minute source file.

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

## Reproducible score matrix

After fetching the recordings, run the complete manifest or selected IDs through the file optimizer:

```sh
npm run corpus:score
npm run corpus:score -- arrl-w1aw-18wpm cj3a-deep-notches
```

The concise matrix reports normalized raw-copy CER, WER, word-boundary F1, partial-anchor and exact-keyword recall, carrier, speed, and convergence. `rawTranscript` is used deliberately: display-only prosign aliases cannot improve a physical-copy score. CER ignores whitespace because spacing has its own metric; normalization is limited to Unicode compatibility, case, typographic quotes/dashes, and collapsed whitespace. WER is ordinary token edit distance. Boundary F1 compares word-break offsets in the whitespace-free copies.

Each manifest `expect` block may provide an exact `reference`, partial `anchors`, exact-token `keywords`, numeric `carrierHz`/`wpm` ranges, a `converged` assertion, and metric `limits`. A numeric expectation may use `anyOf` when two narrow physical bands are equally correct, as with M01's simultaneous 1 kHz and 2 kHz keyed tones. Checks are omitted when trustworthy ground truth is unavailable. The command exits nonzero on an assertion failure or missing cache file; use `--no-fail` for an informational baseline and `--json` for machine-readable output.

## Exact-reference challenge set

The WAE QTC cases come from [Christian Janssen DL1MGB's training collection](https://dl1mgb.com/thisthat/wae.php), which supplies the recording and its authoritative QTC list. JH3AIU is the recoverable QSB/QRM baseline; KL7HF intentionally catches selection of a strong wrong carrier instead of the approximately 791 Hz QTC lane. These recordings have no explicit redistribution license and remain cache-only.

The M01, M12, M14, and M23 recordings and station-format documentation come directly from [Priyom's Morse archive](https://priyom.org/number-stations/morse), licensed CC BY-NC-SA 4.0. Their references preserve physical CW notation: `T` is a one-dash cut zero, `=` is the BT pattern, and M14 retains the audible `<HH>J97` edit rather than silently correcting it. M01 accepts either of its narrow simultaneous tone bands, M12 rejects the competing 1497 Hz lane, and M23 asserts non-convergence on only four sparse identifier groups.

| Manifest ID | Primary stressor | Exact-reference policy |
| --- | --- | --- |
| `wae-jh3aiu-qtc` | Real contest QSB/QRM/QRN | Source QTC list with physically keyed report cut-zeros |
| `wae-kl7hf-qtc` | Strong false carrier over wanted copy | Source QTC list; approximately 791 Hz lane required |
| `priyom-m12` | False carrier and fades | Logged message excerpt with literal cut-zero ending |
| `priyom-m14` | Broadband recording edit | Literal waveform, including `<HH>J97` botch |
| `priyom-m01` | Simultaneous keyed 1/2 kHz tones | Repetition-confirmed excerpt; either true band accepted |
| `priyom-m23` | Very sparse slow identifier | Four complete `943` groups; terminal transient excluded |
| `priyom-m03-qrm` | Severe broadband QRM and a strong false winner | Full waveform reference reconstructed from paired and single repetitions |

The 256.8-second M08a sample remains in the ignored research cache but is not in the default matrix: its full multipass runtime is disproportionate to this focused set, while M01 already covers cut-number and simultaneous-tone behavior.

## Doppler and false-carrier challenges

`uba-lusat-doppler` comes from the [Royal Belgian Amateur Radio Union's off-air CW collection](https://old.uba.be/fr/accueil/cw/cw-air-et-ailleurs). UBA publishes both the [43.2-second received LUSAT recording](https://old.uba.be/sites/default/files/media/cw/ubacw_lusat.mp3) and its [exact telemetry line](https://old.uba.be/sites/default/files/uploads/cw/ubacw_lusat.txt), and explains the cut-number fields. During the pass, the wanted audio tone falls by roughly 440 Hz because of orbital Doppler. This case therefore scores exact copy while allowing the selected carrier to lie anywhere along the observed ridge. UBA states no redistribution license, so the audio remains cache-only.

A scratch quadratic-ridge receiver tracked the active carrier from about 1823 to 1384 Hz with 0.93 Hz median residual error, then recovered every manifest anchor at 11.5 WPM. That experiment is deliberately not in production yet: the safe follow-up is a same-channel, non-converging rescue used only after all fixed-carrier results fail, with minimum span/prominence/dominance and maximum residual-width guards. Keeping this case red prevents a static-frequency decoder from appearing successful through a relaxed expectation.

`priyom-m03-qrm` uses Priyom's [M03 station description and format](https://priyom.org/number-stations/morse/m03) with its [real off-air sample](https://priyom.org/media/117838/m03.ogg). The wanted lane is near 1007 Hz, but severe broadband interference can make an unrelated carrier win an unrestricted search. Priyom does not publish a separate transcript for this sample, so its strong reference is conservatively reconstructed only where the paired transmission and later single-group repetition agree; `T` remains the documented one-dash cut zero. The reference is suitable for character error and carrier selection checks, but not strict word-boundary scoring. Priyom is CC BY-NC-SA 4.0; the audio still remains cache-only so the repository contains metadata and attribution rather than redistributing it.

## Pileup lanes and operator handoffs

`dxu-ft5zm-wb7b` comes from [DX University's real off-air pileup collection](https://dx-university.com/wp/audio-files/). The source documents that its stereo channels place FT5ZM on one side and the callers on the other, and identifies the extended WB7B exchange beginning near 08:09. The migrated site's old audio link is unavailable, so the manifest fetches the preserved original from the Internet Archive. The approximately 697 Hz fundamental is asserted rather than accepting its strong third harmonic. Only the source-confirmed `WB7B` and `5NN` anchors are scored; the many overlapping callers do not have a trustworthy full transcript.

`varc-session7-handoff` is a 48-second excerpt from [Verulam Amateur Radio Club's recorded CW class](https://www.verulam-arc.org.uk/cw-class/). The club publishes a line-by-line transcript of the live 40 m G4IYT/G0OIK QSO, explicitly retaining sending and copying mistakes. The excerpt ends G4IYT's turn and begins G0OIK's turn on essentially the same audio carrier, so both callsign-order anchors are required while convergence must remain false if the timing reset is not solved. CER is intentionally omitted because the short file starts and ends inside longer published transmissions.

Neither source states a redistribution license. Their recordings and derived stereo WAV excerpts remain cache-only and must not be committed.

Downloaded originals, converted WAV segments, and fetched transcripts remain under the ignored `test-corpus/cache/` directory. The harness commits only source metadata and expectations, never third-party audio.
