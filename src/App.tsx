import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Check,
  CircleHelp,
  Copy,
  Download,
  FileAudio,
  Keyboard,
  Mic,
  Moon,
  Play,
  Radio,
  ShieldCheck,
  Sparkles,
  Square,
  Sun,
  Trash2,
  Upload,
  Volume2,
  X,
  Zap,
} from "lucide-react";
import { AudioEngine, type AnalysisFrame, type AudioSourceKind } from "./audio/AudioEngine";
import type { DecodedCharacter } from "./audio/AdaptiveMorseDecoder";
import { interpretHamCopy, type HamMarker } from "./audio/hamInterpreter";
import type {
  OfflineDecodeCandidate,
  OfflineDecodeProgress,
  OfflineDecodeResult,
} from "./audio/OfflineMorseOptimizer";
import { Panel, RangeControl, Toggle } from "./components/Controls";
import { SignalScope } from "./components/SignalScope";

type RunStatus = "idle" | "starting" | "optimizing" | "listening" | "playing" | "complete" | "error";
type Theme = "dark" | "light";

interface Settings {
  autoTune: boolean;
  frequency: number;
  sensitivity: number;
  adaptiveSpeed: boolean;
  wpmHint: number;
  handKeyAssist: boolean;
  languageAssist: boolean;
  theme: Theme;
}

interface TranscriptToken extends DecodedCharacter {
  id: number;
  space?: boolean;
}

const DEFAULT_SETTINGS: Settings = {
  autoTune: true,
  frequency: 700,
  sensitivity: 72,
  adaptiveSpeed: true,
  wpmHint: 20,
  handKeyAssist: true,
  languageAssist: true,
  theme: "dark",
};

const STATUS_COPY: Record<RunStatus, string> = {
  idle: "RX standby",
  starting: "Arming input",
  optimizing: "Multipass solve",
  listening: "RX active",
  playing: "File analysis",
  complete: "Buffer complete",
  error: "Input fault",
};

function loadSettings(): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem("morsewave.settings") ?? "{}");
    return { ...DEFAULT_SETTINGS, ...saved };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function formatDuration(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function sourceLabel(source: AudioSourceKind) {
  if (source === "microphone") return "Microphone";
  if (source === "file") return "Audio file";
  return "Guided demo";
}

interface StationLane {
  candidate: OfflineDecodeCandidate;
  selected: boolean;
}

const clampQuality = (value: number) => Math.max(0, Math.min(1, value));

function stationLanesFor(result: OfflineDecodeResult | null): StationLane[] {
  if (!result) return [];

  return result.candidates
    .map((candidate) => ({
      candidate,
      selected: candidate.channel === result.channel &&
        Math.abs(candidate.frequency - result.frequency) < 5,
    }))
    .filter(({ candidate }) => {
      const copyLength = candidate.transcript.replace(/\s/g, "").length;
      const isReportedSelectedFundamental = result.fundamentalHz &&
        Math.abs(result.frequency - result.fundamentalHz) >= 55 &&
        Math.abs(candidate.frequency - result.fundamentalHz) < 55 &&
        Math.abs(candidate.frequency - result.frequency) >= 55;
      return !isReportedSelectedFundamental && copyLength >= 2 &&
        candidate.validRatio >= 0.78 &&
        candidate.timingQuality >= 0.58 &&
        candidate.decoderScore >= 0.35;
    })
    .sort((left, right) => Number(right.selected) - Number(left.selected) ||
      right.candidate.decoderScore - left.candidate.decoderScore)
    .reduce<StationLane[]>((lanes, lane) => {
      if (lanes.length >= 4 || lanes.some(({ candidate }) =>
        Math.abs(candidate.frequency - lane.candidate.frequency) < 55)) return lanes;
      return [...lanes, lane];
    }, []);
}

function carrierDisplay(frequency: number, fundamentalHz?: number) {
  if (!fundamentalHz || Math.abs(frequency - fundamentalHz) < 55) {
    return { primary: `${Math.round(frequency)} HZ`, detail: null };
  }
  const harmonic = Math.max(2, Math.round(frequency / fundamentalHz));
  return {
    primary: `${Math.round(fundamentalHz)} HZ F0`,
    detail: `DECODE ${Math.round(frequency)} HZ · H${harmonic}`,
  };
}

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [status, setStatus] = useState<RunStatus>("idle");
  const [source, setSource] = useState<AudioSourceKind>("microphone");
  const [frame, setFrame] = useState<AnalysisFrame | null>(null);
  const [tokens, setTokens] = useState<TranscriptToken[]>([]);
  const [lastCharacter, setLastCharacter] = useState<TranscriptToken | null>(null);
  const [sourceName, setSourceName] = useState("Built-in audio input");
  const [sourceDuration, setSourceDuration] = useState<number | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [offlineResult, setOfflineResult] = useState<OfflineDecodeResult | null>(null);
  const [offlineProgress, setOfflineProgress] = useState<OfflineDecodeProgress | null>(null);
  const engineRef = useRef<AudioEngine | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const tokenId = useRef(0);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef(status);

  const active = status === "listening" || status === "playing" || status === "starting" || status === "optimizing";

  const appendCharacter = useCallback((character: DecodedCharacter) => {
    const token: TranscriptToken = { ...character, id: tokenId.current++ };
    setTokens((current) => [...current, token]);
    setLastCharacter(token);
  }, []);

  const appendWordGap = useCallback((timestamp: number) => {
    setTokens((current) => {
      if (current.length === 0 || current.at(-1)?.space) return current;
      return [
        ...current,
        { id: tokenId.current++, value: " ", pattern: "/", confidence: 1, timestamp, space: true },
      ];
    });
  }, []);

  useEffect(() => {
    const engine = new AudioEngine(
      {
        sensitivity: settings.sensitivity,
        wpmHint: settings.wpmHint,
        adaptiveSpeed: settings.adaptiveSpeed,
        handKeyAssist: settings.handKeyAssist,
        languageAssist: settings.languageAssist,
      },
      {
        onCharacter: appendCharacter,
        onWordGap: appendWordGap,
        onAnalysis: setFrame,
        onEnded: () => setStatus("complete"),
        onError: (nextError) => {
          setError(nextError.message);
          setStatus("error");
        },
      },
    );
    engineRef.current = engine;
    return () => {
      engine.dispose();
      engineRef.current = null;
    };
    // Engine lifetime is intentionally independent from mutable settings.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appendCharacter, appendWordGap]);

  useEffect(() => {
    localStorage.setItem("morsewave.settings", JSON.stringify(settings));
    document.documentElement.dataset.theme = settings.theme;
    engineRef.current?.updateDecoderConfig({
      sensitivity: settings.sensitivity,
      wpmHint: settings.wpmHint,
      adaptiveSpeed: settings.adaptiveSpeed,
      handKeyAssist: settings.handKeyAssist,
      languageAssist: settings.languageAssist,
    });
    engineRef.current?.updateTunerConfig({
      frequency: settings.frequency,
      autoTune: settings.autoTune,
    });
  }, [settings]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    if (!active || startedAt === null) return;
    const update = () => setElapsed((performance.now() - startedAt) / 1000);
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, [active, startedAt]);

  useEffect(() => {
    const element = transcriptRef.current;
    if (!element) return;
    const rawCopy = element.querySelector<HTMLElement>(":scope > p[role='log']");
    const top = rawCopy
      ? Math.max(0, rawCopy.offsetTop + rawCopy.offsetHeight - element.clientHeight + 16)
      : element.scrollHeight;
    element.scrollTo({ top, behavior: "smooth" });
  }, [tokens.length]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 2400);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const beginRun = useCallback((kind: AudioSourceKind, name: string) => {
    setSource(kind);
    setSourceName(name);
    setSourceDuration(null);
    setStartedAt(performance.now());
    setElapsed(0);
    setError(null);
    setFrame(null);
    setOfflineResult(null);
    setOfflineProgress(null);
    setStatus("starting");
  }, []);

  const startMicrophone = useCallback(async () => {
    if (statusRef.current === "starting") return;
    beginRun("microphone", "Built-in audio input");
    try {
      await engineRef.current?.startMicrophone();
      setStatus("listening");
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "The microphone could not be opened.";
      setError(message);
      setStatus("error");
    }
  }, [beginRun]);

  const startDemo = useCallback(async () => {
    if (statusRef.current === "starting") return;
    engineRef.current?.stop();
    setTokens([]);
    setLastCharacter(null);
    beginRun("demo", "19 WPM · noisy practice signal");
    try {
      const metadata = await engineRef.current?.startDemo();
      setSourceDuration(metadata?.duration ?? null);
      setStatus("playing");
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "The demo signal could not be started.";
      setError(message);
      setStatus("error");
    }
  }, [beginRun]);

  const startFile = useCallback(async (file: File) => {
    if (!file) return;
    engineRef.current?.stop();
    setTokens([]);
    setLastCharacter(null);
    beginRun("file", file.name);
    setStatus("optimizing");
    try {
      const metadata = await engineRef.current?.startFile(file, {
        wpmHint: settings.wpmHint,
        languageAssist: settings.languageAssist,
        minimumFrequency: 200,
        maximumFrequency: 4000,
      }, setOfflineProgress);
      setSourceDuration(metadata?.duration ?? null);
      if (metadata?.optimization) setOfflineResult(metadata.optimization);
      if (metadata?.optimization?.converged) {
        const optimizedTokens = metadata.optimization.tokens.map((token) => ({
          ...token,
          id: tokenId.current++,
        }));
        setTokens(optimizedTokens);
        setLastCharacter([...optimizedTokens].reverse().find((token) => !token.space) ?? null);
        setOfflineProgress(null);
      } else if (metadata?.optimization) {
        setOfflineProgress(null);
        setToast("Multipass inconclusive · streaming decoder engaged");
      } else {
        setOfflineProgress(null);
        setToast("Streaming decoder engaged");
      }
      setStartedAt(performance.now());
      setStatus("playing");
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      const message = cause instanceof Error ? cause.message : "The file could not be opened.";
      setError(message);
      setStatus("error");
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [beginRun, settings.languageAssist, settings.wpmHint]);

  const stop = useCallback(() => {
    engineRef.current?.stop();
    setStatus("idle");
    setStartedAt(null);
  }, []);

  const clearSession = useCallback(() => {
    setTokens([]);
    setLastCharacter(null);
    setElapsed(0);
    setOfflineResult(null);
    setOfflineProgress(null);
    tokenId.current = 0;
    engineRef.current?.resetDecoder(true);
    setToast("Session cleared");
  }, []);

  const transcript = useMemo(() => tokens.map((token) => token.value).join("").trim(), [tokens]);
  const hamInterpretation = useMemo(
    () => settings.languageAssist && transcript ? interpretHamCopy(transcript) : null,
    [settings.languageAssist, transcript],
  );
  const hamMarkers = useMemo(() => {
    const unique = new Map<string, HamMarker>();
    for (const marker of hamInterpretation?.markers ?? []) {
      unique.set(`${marker.token}:${marker.expansion}`, marker);
    }
    return [...unique.values()];
  }, [hamInterpretation]);
  const offlineCandidateInterpretation = useMemo(() => {
    if (!settings.languageAssist || !offlineResult || offlineResult.converged || offlineResult.hamEvidence <= 0) {
      return null;
    }
    return interpretHamCopy(offlineResult.transcript);
  }, [offlineResult, settings.languageAssist]);
  const stationLanes = useMemo(() => stationLanesFor(offlineResult), [offlineResult]);
  const characters = useMemo(() => tokens.filter((token) => !token.space), [tokens]);
  const recentCharacters = useMemo(() => characters.slice(-5).reverse(), [characters]);
  const averageConfidence = useMemo(() => {
    const recent = characters.slice(-12);
    if (recent.length === 0) return 0;
    return recent.reduce((sum, token) => sum + token.confidence, 0) / recent.length;
  }, [characters]);
  const wordCount = transcript ? transcript.split(/\s+/).length : 0;

  const copyTranscript = useCallback(async () => {
    if (!transcript) return;
    await navigator.clipboard.writeText(transcript);
    setToast("Transcript copied");
  }, [transcript]);

  const exportTranscript = useCallback(() => {
    if (!transcript) return;
    const header = [
      "MORSEWAVE DECODE SESSION",
      `Source: ${sourceName}`,
      `Created: ${new Date().toLocaleString()}`,
      `Estimated speed: ${Math.round(frame?.metrics.wpm ?? settings.wpmHint)} WPM`,
      "",
    ].join("\n");
    const url = URL.createObjectURL(new Blob([header + transcript + "\n"], { type: "text/plain" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `morsewave-${new Date().toISOString().slice(0, 10)}.txt`;
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setToast("Transcript exported");
  }, [frame?.metrics.wpm, settings.wpmHint, sourceName, transcript]);

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select, [contenteditable='true']")) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "o") {
        event.preventDefault();
        fileInputRef.current?.click();
      } else if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        clearSession();
      } else if (event.key === "Escape" && statusRef.current !== "idle") {
        stop();
      } else if (event.code === "Space" && !event.metaKey && !event.ctrlKey) {
        event.preventDefault();
        if (["listening", "playing", "starting"].includes(statusRef.current)) stop();
        else void startMicrophone();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [clearSession, startMicrophone, stop]);

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files[0];
    if (file) void startFile(file);
  };

  const trustedOfflineResult = offlineResult?.converged ? offlineResult : null;
  const detectedFrequency = trustedOfflineResult?.frequency ?? frame?.frequency ?? settings.frequency;
  const estimatedWpm = trustedOfflineResult?.wpm ?? frame?.metrics.wpm ?? settings.wpmHint;
  const signalSnr = frame?.metrics.snrDb ?? 0;
  const signalLocked = Boolean(trustedOfflineResult || active && frame && frame.prominenceDb >= 5);
  const timingJitter = frame?.metrics.timingJitter ?? 0;
  const recoveredEdges = frame?.metrics.recoveredEdges ?? 0;

  return (
    <div
      className={`app-shell ${dragActive ? "is-dragging" : ""}`}
      onDragEnter={(event) => { event.preventDefault(); setDragActive(true); }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragActive(false);
      }}
      onDrop={handleDrop}
    >
      <header className="topbar">
        <div className="brand" aria-label="Morsewave">
          <div className="brand-mark" aria-hidden="true">
            <i /><i /><i /><i />
          </div>
          <div>
            <strong>Morsewave</strong>
            <span>CW signal workstation</span>
          </div>
        </div>
        <div className="topbar-center">
          <span className="privacy-chip"><ShieldCheck size={13} /> LOCAL DSP / NO UPLINK</span>
          <span className={`status-chip status-${status}`}>
            <i /> {STATUS_COPY[status]}
          </span>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" onClick={() => setShowHelp(true)} aria-label="Help and shortcuts" title="Help and shortcuts">
            <CircleHelp size={18} />
          </button>
          <button
            className="icon-button"
            onClick={() => setSettings((current) => ({ ...current, theme: current.theme === "dark" ? "light" : "dark" }))}
            aria-label={`Use ${settings.theme === "dark" ? "light" : "dark"} appearance`}
            title="Toggle appearance"
          >
            {settings.theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <div className="command-bar">
        <div className="source-picker" aria-label="Audio source">
          <span className="source-picker-label">INPUT</span>
          <button className={source === "microphone" ? "is-selected" : ""} onClick={() => void startMicrophone()}>
            <Mic size={15} /> Microphone
          </button>
          <button className={source === "file" ? "is-selected" : ""} onClick={() => fileInputRef.current?.click()}>
            <FileAudio size={15} /> File
          </button>
          <button className={source === "demo" ? "is-selected" : ""} onClick={() => void startDemo()}>
            <Sparkles size={15} /> Demo
          </button>
          <input
            ref={fileInputRef}
            className="visually-hidden"
            type="file"
            accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.aiff,.aif"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void startFile(file);
            }}
          />
        </div>
        <div className="source-context">
          <Volume2 size={15} />
          <span>{sourceName}</span>
          {sourceDuration !== null && <em>{formatDuration(sourceDuration)}</em>}
        </div>
        <button
          className={`transport-button ${active ? "is-stop" : ""}`}
          onClick={active ? stop : startMicrophone}
          disabled={status === "starting"}
        >
          {active ? <Square size={14} fill="currentColor" /> : <Play size={15} fill="currentColor" />}
          {status === "starting" ? "ARMING…" : status === "optimizing" ? "CANCEL SOLVE" : active ? "STOP RX" : "ARM RX"}
          <kbd>Space</kbd>
        </button>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <span><Zap size={17} /> {error}</span>
          <button onClick={() => { setError(null); setStatus("idle"); }} aria-label="Dismiss error"><X size={16} /></button>
        </div>
      )}

      <main className="dashboard">
        <div className="primary-grid">
          <Panel
            eyebrow="Decode buffer"
            title="Raw on-air copy"
            className="transcript-panel"
            action={
              <div className="panel-tools">
                <button onClick={() => void copyTranscript()} disabled={!transcript} title="Copy transcript"><Copy size={15} /></button>
                <button onClick={exportTranscript} disabled={!transcript} title="Export transcript"><Download size={15} /></button>
                <button onClick={clearSession} disabled={!tokens.length} title="Clear session"><Trash2 size={15} /></button>
              </div>
            }
          >
            <div ref={transcriptRef} className={`transcript ${tokens.length ? "has-copy" : ""}`} aria-label="Decoded Morse transcript">
              {tokens.length ? (
                <>
                  <p
                    role="log"
                    aria-live="polite"
                    aria-atomic="false"
                    aria-relevant="additions text"
                    aria-label="Raw recovered ham copy"
                  >
                    {tokens.map((token) => token.space ? (
                      <span key={token.id}> </span>
                    ) : (
                      <span
                        key={token.id}
                        className={token.confidence < 0.55 ? "low-confidence" : ""}
                        title={`${token.pattern}${token.rawValue ? ` · raw ${token.rawValue}` : ""} · ${Math.round(token.confidence * 100)}% confidence`}
                      >
                        {token.value}
                      </span>
                    ))}
                    {active && <span className="decode-cursor" aria-hidden="true" />}
                  </p>
                  {hamInterpretation && hamMarkers.length > 0 && (
                    <section className="copy-advisory" aria-labelledby="copy-advisory-title">
                      <div className="copy-advisory-heading">
                        <span id="copy-advisory-title">HAM SEMANTIC LAYER</span>
                        <small>OPERATOR AID · NON-AUTHORITATIVE</small>
                      </div>
                      <ul className="marker-strip" aria-label="Recognized CW operating markers">
                        {hamMarkers.map((marker) => (
                          <li className="marker-chip" key={`${marker.token}:${marker.expansion}`}>
                            <code>{marker.token}</code>
                            <span>{marker.expansion}</span>
                          </li>
                        ))}
                      </ul>
                      <p className="advisory-expansion">
                        <span>ADVISORY EXPANSION</span>
                        <span>{hamInterpretation.expandedText}</span>
                      </p>
                    </section>
                  )}
                </>
              ) : (
                <div className="empty-transcript">
                  <div className="empty-icon"><Radio size={25} /></div>
                  <div>
                    <strong>{status === "optimizing" ? (offlineProgress?.detail ?? "SURVEYING CARRIER TRACKS").toUpperCase() : active ? "ACQUIRING KEYED CARRIER" : "INPUT PATH NOT ARMED"}</strong>
                    <p>{status === "optimizing" ? `Adaptive file solver · ${Math.round((offlineProgress?.fraction ?? 0) * 100)}% · testing carrier, gate, speed, and spacing hypotheses.` : active ? "Estimating carrier, noise floor, and mark/gap clusters. Ambiguous hand timing remains buffered." : "Arm the audio input, load a recording, or inject the internal reference signal."}</p>
                    {status === "optimizing" && <div className="optimizer-progress"><i style={{ width: `${Math.round((offlineProgress?.fraction ?? 0) * 100)}%` }} /></div>}
                  </div>
                  {!active && <button onClick={() => void startDemo()}><Play size={14} fill="currentColor" /> Inject reference</button>}
                </div>
              )}
              {offlineCandidateInterpretation && offlineResult && (
                <section className="multipass-candidate" aria-labelledby="multipass-candidate-title">
                  <div className="multipass-candidate-heading">
                    <span id="multipass-candidate-title">UNSELECTED MULTIPASS CANDIDATE</span>
                    <small>
                      {Math.round(offlineResult.frequency)} HZ · {offlineResult.wpm.toFixed(1)} WPM ·
                      {` CONTEXT ${(offlineResult.hamEvidence * 100).toFixed(1)} / 10`}
                    </small>
                  </div>
                  <p>{offlineResult.transcript}</p>
                  <div className="candidate-reading">
                    <span>CONTEXT READ</span>
                    <span>{offlineCandidateInterpretation.expandedText}</span>
                  </div>
                  <em>Held out because the physical timing/validity convergence gate did not pass.</em>
                </section>
              )}
              {offlineResult && stationLanes.length > 0 && (
                <section
                  className="station-lanes"
                  aria-labelledby="station-lanes-title"
                  aria-describedby="station-lanes-note"
                >
                  <div className="station-lanes-heading">
                    <span id="station-lanes-title">STATION LANES</span>
                    <small>INDEPENDENT CARRIERS · ALTERNATIVES ARE NON-AUTHORITATIVE</small>
                  </div>
                  <ol aria-label="Credible distinct-frequency Morse carrier candidates">
                    {stationLanes.map(({ candidate, selected }, index) => (
                      <li className={`station-lane ${selected ? "is-selected" : ""}`} key={`${candidate.channel}:${Math.round(candidate.frequency)}`}>
                        <div className="station-lane-header">
                          <span className="station-lane-index" aria-hidden="true">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <strong>{carrierDisplay(
                            candidate.frequency,
                            candidate.fundamentalHz ?? (selected ? offlineResult.fundamentalHz : undefined),
                          ).primary}</strong>
                          <span>{carrierDisplay(
                            candidate.frequency,
                            candidate.fundamentalHz ?? (selected ? offlineResult.fundamentalHz : undefined),
                          ).detail
                            ? `${carrierDisplay(
                              candidate.frequency,
                              candidate.fundamentalHz ?? (selected ? offlineResult.fundamentalHz : undefined),
                            ).detail} · ${candidate.wpm.toFixed(1)} WPM`
                            : `${candidate.wpm.toFixed(1)} WPM`}</span>
                          <span className="station-lane-metrics">
                            QUALITY {Math.round(clampQuality(candidate.decoderScore) * 100)}%
                            {` · VALID ${Math.round(clampQuality(candidate.validRatio) * 100)}%`}
                            {` · TIMING ${Math.round(clampQuality(candidate.timingQuality) * 100)}%`}
                          </span>
                          <span className={`station-lane-state ${selected ? "is-selected" : ""}`}>
                            {selected ? (offlineResult.converged ? "SELECTED" : "SELECTED CANDIDATE") : "ALT · NON-AUTH"}
                          </span>
                        </div>
                        <div className="station-lane-copy">
                          <span>RAW COPY</span>
                          <code>{candidate.transcript}</code>
                        </div>
                      </li>
                    ))}
                  </ol>
                  <p id="station-lanes-note">
                    One physical carrier per lane. Copy is never merged across frequencies.
                  </p>
                </section>
              )}
              {offlineResult?.neuralAdvisories?.length ? (
                <section
                  className="station-lanes neural-lanes"
                  aria-labelledby="neural-lanes-title"
                  aria-describedby="neural-lanes-note"
                >
                  <div className="station-lanes-heading">
                    <span id="neural-lanes-title">CWFORMER LANE COPY</span>
                    <small>LOCAL INT8 · ADVISORY ONLY · NEVER USED FOR LOCK</small>
                  </div>
                  <ol aria-label="Independent non-authoritative neural station transcripts">
                    {offlineResult.neuralAdvisories.map((advisory, index) => (
                      <li
                        className="station-lane neural-lane"
                        key={`${advisory.channel}:${Math.round(advisory.sourceFrequency)}`}
                      >
                        <div className="station-lane-header">
                          <span className="station-lane-index" aria-hidden="true">
                            N{index + 1}
                          </span>
                          <strong>{Math.round(advisory.sourceFrequency)} HZ</strong>
                          <span>CH {advisory.channel + 1} → {advisory.conditionedFrequency} HZ</span>
                          <span className="station-lane-metrics">
                            CTC μ {Math.round(clampQuality(advisory.meanTokenConfidence) * 100)}%
                            {` · ${(advisory.inferenceMs / 1_000).toFixed(1)} S`}
                          </span>
                          <span className="station-lane-state neural-lane-state">NON-AUTH</span>
                        </div>
                        <div className="station-lane-copy neural-lane-copy">
                          <span>MODEL COPY</span>
                          <code>{advisory.transcript}</code>
                        </div>
                      </li>
                    ))}
                  </ol>
                  <p id="neural-lanes-note">
                    One model pass per classical carrier. Lanes and text are never merged; raw Multipass copy remains authoritative.
                  </p>
                </section>
              ) : null}
            </div>
            <div className="copy-footer">
              <span>{characters.length} characters</span>
              <i />
              <span>{wordCount} words</span>
              <i />
              <span>{formatDuration(elapsed)} elapsed</span>
              {offlineResult && <span className="optimizer-summary">
                MULTIPASS {offlineResult.converged ? "LOCK" : "CANDIDATE"} · {offlineResult.passCount.toLocaleString()} hypotheses · {offlineResult.fundamentalHz && Math.abs(offlineResult.frequency - offlineResult.fundamentalHz) >= 55
                  ? `F0 ${Math.round(offlineResult.fundamentalHz)} Hz · decode ${Math.round(offlineResult.frequency)} Hz H${Math.max(2, Math.round(offlineResult.frequency / offlineResult.fundamentalHz))}`
                  : `${Math.round(offlineResult.frequency)} Hz`} · {Math.round(offlineResult.confidence * 100)}%
              </span>}
              <span className="copy-quality"><Check size={13} /> Confidence shading is visible on uncertain copy</span>
            </div>
          </Panel>

          <Panel
            eyebrow="IF analysis"
            title="Narrow-band spectrum"
            className="scope-panel"
            action={<span className={`lock-badge ${signalLocked ? "is-locked" : ""}`}><i /> {trustedOfflineResult ? `Track ${trustedOfflineResult.channel + 1} selected` : signalLocked ? "Tone locked" : "Seeking"}</span>}
          >
            <SignalScope frame={frame} active={active} />
            <div className="scope-readout">
              <div><span>Tone</span><strong>{Math.round(detectedFrequency)}<small> Hz</small></strong></div>
              <div><span>Noise floor</span><strong>{Math.round(frame?.metrics.noiseFloorDb ?? -78)}<small> dBFS</small></strong></div>
              <div><span>Peak Δ</span><strong>{Math.round(frame?.prominenceDb ?? 0)}<small> dB</small></strong></div>
              <div><span>Gate</span><strong className={frame?.metrics.gateOpen ? "accent" : ""}>{frame?.metrics.gateOpen ? "KEYED" : "QUIET"}</strong></div>
            </div>
          </Panel>
        </div>

        <section className="metric-strip" aria-label="Decoder metrics">
          <div className="metric-card">
            <span className="metric-icon"><Activity size={16} /></span>
            <div><span>Signal / noise</span><strong>{signalSnr > 0 ? signalSnr.toFixed(1) : "—"}<small> dB</small></strong></div>
            <div className="mini-meter"><i style={{ width: `${Math.min(100, signalSnr / 24 * 100)}%` }} /></div>
          </div>
          <div className="metric-card">
            <span className="metric-icon amber"><Zap size={16} /></span>
            <div><span>Estimated speed</span><strong>{estimatedWpm.toFixed(1)}<small> WPM</small></strong></div>
            <em>{trustedOfflineResult ? "Consensus" : settings.adaptiveSpeed ? "Tracking" : "Fixed"}</em>
          </div>
          <div className="metric-card">
            <span className="metric-icon violet"><Radio size={16} /></span>
            <div><span>Timing deviation</span><strong>{frame ? timingJitter.toFixed(1) : "—"}<small>% σ</small></strong></div>
            <em>{offlineResult ? `${offlineResult.passCount.toLocaleString()} PASSES` : settings.handKeyAssist ? "BEAM / 64" : "EDGE / 1"}</em>
          </div>
          <div className="metric-card current-symbol-card">
            <div><span>Current character</span><strong className="morse-symbols" aria-label={frame?.metrics.currentPattern || "None"}>
              {frame?.metrics.currentPattern ? [...frame.metrics.currentPattern].map((symbol, index) => (
                <i key={index} className={symbol === "." ? "dit" : "dah"} />
              )) : <b>···</b>}
            </strong></div>
            <em>{recoveredEdges ? `${recoveredEdges} edges repaired` : lastCharacter ? `${lastCharacter.value} · ${lastCharacter.pattern}` : "No event"}</em>
          </div>
        </section>

        <div className="secondary-grid">
          <Panel eyebrow="DSP configuration" title="Decoder matrix" className="controls-panel" action={<span className="dsp-badge">GOERTZEL / 1024·256</span>}>
            <div className="control-columns">
              <div className="control-group">
                <Toggle
                  checked={settings.autoTune}
                  onChange={(autoTune) => setSettings((current) => ({ ...current, autoTune }))}
                  label="Automatic tone lock"
                  description="Tracks frequency drift"
                />
                <RangeControl
                  label="Center frequency"
                  value={settings.autoTune ? detectedFrequency : settings.frequency}
                  minimum={200}
                  maximum={4000}
                  step={5}
                  unit=" Hz"
                  hint="The expected pitch of the CW tone. Automatic lock scans the full range."
                  disabled={settings.autoTune}
                  onChange={(frequency) => setSettings((current) => ({ ...current, frequency }))}
                />
              </div>
              <div className="control-divider" />
              <div className="control-group">
                <Toggle
                  checked={settings.adaptiveSpeed}
                  onChange={(adaptiveSpeed) => setSettings((current) => ({ ...current, adaptiveSpeed }))}
                  label="Adaptive keying speed"
                  description="Learns from every mark"
                />
                <RangeControl
                  label="Starting speed"
                  value={settings.wpmHint}
                  minimum={5}
                  maximum={50}
                  unit=" WPM"
                  hint="An initial estimate; adaptive mode quickly follows the sender."
                  onChange={(wpmHint) => setSettings((current) => ({ ...current, wpmHint }))}
                />
              </div>
              <div className="control-divider" />
              <div className="control-group sensitivity-group">
                <Toggle
                  checked={settings.handKeyAssist}
                  onChange={(handKeyAssist) => setSettings((current) => ({ ...current, handKeyAssist }))}
                  label="Hand-key timing solver"
                  description="64-path fuzzy segmentation"
                />
                <Toggle
                  checked={settings.languageAssist}
                  onChange={(languageAssist) => setSettings((current) => ({ ...current, languageAssist }))}
                  label="CW / ham prior"
                  description="Prosigns, Q-signals & shorthand"
                />
                <div className="sensitivity-header"><span>Sensitivity</span><strong>{settings.sensitivity}%</strong></div>
                <input
                  className="sensitivity-dial"
                  type="range"
                  min="0"
                  max="100"
                  value={settings.sensitivity}
                  style={{ "--range-progress": `${settings.sensitivity}%` } as React.CSSProperties}
                  onChange={(event) => setSettings((current) => ({ ...current, sensitivity: Number(event.target.value) }))}
                  aria-label="Decoder sensitivity"
                />
                <div className="range-extents"><span>Clean</span><span>Weak signal</span></div>
              </div>
            </div>
          </Panel>

          <Panel eyebrow="Event log" title={`Recent copy · ${characters.length ? Math.round(averageConfidence * 100) : 0}% mean`} className="recent-panel" action={<span className="session-live"><i /> {active ? "STREAM" : "HOLD"}</span>}>
            <div className="recent-list">
              {recentCharacters.length ? recentCharacters.map((token) => (
                <div className="recent-row" key={token.id}>
                  <span className="recent-character">{token.value}</span>
                  <span className="recent-pattern">{[...token.pattern].map((symbol, index) => <i className={symbol === "." ? "dit" : "dah"} key={index} />)}</span>
                  <span className="confidence-number">{Math.round(token.confidence * 100)}%</span>
                  <span className="confidence-bar"><i style={{ width: `${token.confidence * 100}%` }} /></span>
                </div>
              )) : (
                <div className="recent-empty"><Activity size={20} /><span>Decoded characters will appear here with timing confidence.</span></div>
              )}
            </div>
          </Panel>
        </div>
      </main>

      <footer className="app-footer">
        <span><ShieldCheck size={13} /> AUDIO PATH: LOCAL</span>
        <span className="footer-center">INTL MORSE · IF 200–4000 HZ · GOERTZEL 1024/256 · 5–50 WPM</span>
        <button onClick={() => setShowHelp(true)}><Keyboard size={13} /> Shortcuts</button>
      </footer>

      {dragActive && (
        <div className="drop-overlay" aria-hidden="true">
          <div><Upload size={34} /><strong>Drop to decode</strong><span>WAV, MP3, M4A, FLAC, OGG, or AIFF</span></div>
        </div>
      )}

      {toast && <div className="toast" role="status"><Check size={15} /> {toast}</div>}

      {showHelp && (
        <div className="modal-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.currentTarget === event.target) setShowHelp(false);
        }}>
          <section className="help-modal" role="dialog" aria-modal="true" aria-labelledby="help-title">
            <button className="modal-close" onClick={() => setShowHelp(false)} aria-label="Close help"><X size={18} /></button>
            <div className="help-mark"><Radio size={27} /></div>
            <span className="eyebrow">Quick start</span>
            <h2 id="help-title">Make clean copy from messy signals.</h2>
            <p>Morsewave analyzes audio locally, locks onto the strongest CW tone, then learns the sender’s actual timing as they key.</p>
            <div className="help-steps">
              <div><b>1</b><span><strong>Choose a source</strong><small>Microphone, recording, or the built-in practice signal.</small></span></div>
              <div><b>2</b><span><strong>Let tone lock settle</strong><small>A green “Tone locked” badge means the carrier stands above the noise.</small></span></div>
              <div><b>3</b><span><strong>Fine-tune only if needed</strong><small>Raise sensitivity for weak CW; use manual frequency for crowded bands.</small></span></div>
            </div>
            <div className="shortcut-grid">
              <span><kbd>Space</kbd> Start / stop</span>
              <span><kbd>⌘ O</kbd> Open audio</span>
              <span><kbd>⌘ K</kbd> Clear copy</span>
              <span><kbd>Esc</kbd> Stop decoder</span>
            </div>
            <button className="modal-primary" onClick={() => { setShowHelp(false); void startDemo(); }}><Sparkles size={15} /> Try the guided demo</button>
          </section>
        </div>
      )}
    </div>
  );
}
