import {
  AdaptiveMorseDecoder,
  type DecodedCharacter,
  type DecoderConfig,
  type DecoderMetrics,
} from "./AdaptiveMorseDecoder";
import { createDemoBuffer } from "./demoSignal";
import {
  optimizeMorseBuffer,
  type OfflineDecodeOptions,
  type OfflineDecodeProgress,
  type OfflineDecodeResult,
} from "./OfflineMorseOptimizer";

export type AudioSourceKind = "microphone" | "file" | "demo";

export interface AnalysisFrame {
  timestamp: number;
  levelDb: number;
  broadbandDb: number;
  purity: number;
  frequency: number;
  prominenceDb: number;
  spectrum?: Float32Array;
  metrics: DecoderMetrics;
}

export interface AudioEngineCallbacks {
  onCharacter: (character: DecodedCharacter) => void;
  onWordGap: (timestamp: number) => void;
  onAnalysis: (frame: AnalysisFrame) => void;
  onEnded: () => void;
  onError: (error: Error) => void;
}

export interface TunerConfig {
  frequency: number;
  autoTune: boolean;
  minimumFrequency?: number;
  maximumFrequency?: number;
}

export interface FileDecodeMetadata {
  duration: number;
  sampleRate: number;
  optimization?: OfflineDecodeResult;
}

interface WorkletFrame {
  type: "analysis";
  timestamp: number;
  levelDb: number;
  broadbandDb: number;
  purity: number;
  frequency: number;
  prominenceDb: number;
  spectrum?: Float32Array;
}

export class AudioEngine {
  private context: AudioContext | null = null;
  private processor: AudioWorkletNode | null = null;
  private source: AudioNode | null = null;
  private stream: MediaStream | null = null;
  private playbackSource: AudioBufferSourceNode | null = null;
  private monitorGain: GainNode | null = null;
  private analysisSplitter: ChannelSplitterNode | null = null;
  private silentGain: GainNode | null = null;
  private lastUiUpdate = 0;
  private latestSpectrum: Float32Array | undefined;
  private tunerConfig: TunerConfig = { frequency: 700, autoTune: true };
  private fileAnalysisAbort: AbortController | null = null;
  private suppressDecoderOutput = false;
  private readonly decoder: AdaptiveMorseDecoder;
  private readonly callbacks: AudioEngineCallbacks;

  constructor(decoderConfig: DecoderConfig, callbacks: AudioEngineCallbacks) {
    this.callbacks = callbacks;
    this.decoder = new AdaptiveMorseDecoder(decoderConfig, {
      onCharacter: (character) => {
        if (!this.suppressDecoderOutput) callbacks.onCharacter(character);
      },
      onWordGap: (timestamp) => {
        if (!this.suppressDecoderOutput) callbacks.onWordGap(timestamp);
      },
    });
  }

  get isRunning() {
    return this.source !== null;
  }

  async startMicrophone() {
    this.stop();
    const context = await this.ensureContext();
    this.suppressDecoderOutput = false;
    this.postTunerConfig(this.tunerConfig);
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone capture is not available on this system.");
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
        video: false,
      });
      const source = context.createMediaStreamSource(this.stream);
      this.attachSource(source, false);
    } catch (cause) {
      this.stop();
      const message = cause instanceof DOMException && cause.name === "NotAllowedError"
        ? "Microphone access was denied. Allow it in system privacy settings, then try again."
        : "The microphone could not be opened.";
      throw new Error(message, { cause });
    }
  }

  async startFile(
    file: File,
    options: Partial<OfflineDecodeOptions> = {},
    onProgress?: (progress: OfflineDecodeProgress) => void,
  ): Promise<FileDecodeMetadata> {
    this.stop();
    const context = await this.ensureContext();
    let buffer: AudioBuffer;
    try {
      buffer = await context.decodeAudioData(await file.arrayBuffer());
    } catch (cause) {
      throw new Error("This audio file could not be decoded. Try WAV, MP3, M4A, FLAC, or OGG.", { cause });
    }
    const abort = new AbortController();
    this.fileAnalysisAbort = abort;
    let optimization: OfflineDecodeResult;
    try {
      optimization = await optimizeMorseBuffer(buffer, {
        ...options,
        signal: abort.signal,
        onProgress,
      });
    } catch (cause) {
      if (abort.signal.aborted || cause instanceof DOMException && cause.name === "AbortError") throw cause;
      // Silence, voice-only recordings, or an unavailable worker should not
      // make the audio file unusable. Fall back to the proven streaming path.
      this.suppressDecoderOutput = false;
      this.postTunerConfig(this.tunerConfig);
      this.startBuffer(buffer);
      return { duration: buffer.duration, sampleRate: buffer.sampleRate };
    } finally {
      if (this.fileAnalysisAbort === abort) this.fileAnalysisAbort = null;
    }
    if (abort.signal.aborted) throw new DOMException("Decode cancelled.", "AbortError");
    this.suppressDecoderOutput = optimization.converged;
    if (optimization.converged) {
      this.postTunerConfig({ frequency: optimization.frequency, autoTune: false });
    } else this.postTunerConfig(this.tunerConfig);
    this.startBuffer(buffer, optimization.converged ? optimization.channel : undefined);
    return { duration: buffer.duration, sampleRate: buffer.sampleRate, optimization };
  }

  async startDemo() {
    this.stop();
    const context = await this.ensureContext();
    this.suppressDecoderOutput = false;
    this.postTunerConfig(this.tunerConfig);
    const buffer = createDemoBuffer(context);
    this.startBuffer(buffer);
    return { duration: buffer.duration, sampleRate: buffer.sampleRate };
  }

  stop() {
    this.fileAnalysisAbort?.abort();
    this.fileAnalysisAbort = null;
    if (this.playbackSource) {
      this.playbackSource.onended = null;
      try { this.playbackSource.stop(); } catch { /* Already stopped. */ }
      this.playbackSource.disconnect();
    }
    this.playbackSource = null;
    this.source?.disconnect();
    this.source = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.monitorGain?.disconnect();
    this.monitorGain = null;
    this.analysisSplitter?.disconnect();
    this.analysisSplitter = null;
    this.decoder.flush(this.context ? this.context.currentTime * 1000 : undefined);
    this.suppressDecoderOutput = false;
  }

  dispose() {
    this.stop();
    this.processor?.disconnect();
    this.silentGain?.disconnect();
    void this.context?.close();
    this.processor = null;
    this.context = null;
  }

  updateDecoderConfig(config: Partial<DecoderConfig>) {
    this.decoder.updateConfig(config);
  }

  updateTunerConfig(config: Partial<TunerConfig>) {
    this.tunerConfig = { ...this.tunerConfig, ...config };
    this.postTunerConfig(this.tunerConfig);
  }

  private postTunerConfig(config: TunerConfig) {
    this.processor?.port.postMessage({
      type: "config",
      targetFrequency: config.frequency,
      autoTune: config.autoTune,
      minimumFrequency: config.minimumFrequency,
      maximumFrequency: config.maximumFrequency,
    });
  }

  resetDecoder(keepCalibration = false) {
    this.decoder.reset(keepCalibration);
  }

  private async ensureContext() {
    if (!this.context) {
      this.context = new AudioContext({ latencyHint: "interactive" });
      await this.context.audioWorklet.addModule("/cw-processor.js");
      this.processor = new AudioWorkletNode(this.context, "cw-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.processor.port.onmessage = (event: MessageEvent<WorkletFrame>) => this.handleFrame(event.data);
      this.processor.port.onmessageerror = () => this.callbacks.onError(new Error("Audio analysis stopped unexpectedly."));
      this.silentGain = this.context.createGain();
      this.silentGain.gain.value = 0;
      this.processor.connect(this.silentGain).connect(this.context.destination);
      this.updateTunerConfig(this.tunerConfig);
    }
    if (this.context.state === "suspended") await this.context.resume();
    return this.context;
  }

  private attachSource(source: AudioNode, monitor: boolean, analysisChannel?: number) {
    if (!this.context || !this.processor) throw new Error("Audio engine is not ready.");
    this.decoder.reset();
    this.lastUiUpdate = 0;
    this.latestSpectrum = undefined;
    this.source = source;
    if (analysisChannel !== undefined && this.context) {
      this.analysisSplitter = this.context.createChannelSplitter(Math.max(2, analysisChannel + 1));
      source.connect(this.analysisSplitter);
      this.analysisSplitter.connect(this.processor, analysisChannel, 0);
    } else source.connect(this.processor);
    if (monitor) {
      this.monitorGain = this.context.createGain();
      this.monitorGain.gain.value = 0.55;
      source.connect(this.monitorGain).connect(this.context.destination);
    }
  }

  private startBuffer(buffer: AudioBuffer, analysisChannel?: number) {
    if (!this.context) throw new Error("Audio engine is not ready.");
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    this.playbackSource = source;
    this.attachSource(source, true, analysisChannel);
    source.onended = () => {
      if (this.playbackSource !== source) return;
      this.decoder.flush(this.context ? this.context.currentTime * 1000 : undefined);
      source.disconnect();
      this.playbackSource = null;
      this.source = null;
      this.monitorGain?.disconnect();
      this.monitorGain = null;
      this.analysisSplitter?.disconnect();
      this.analysisSplitter = null;
      this.suppressDecoderOutput = false;
      this.callbacks.onEnded();
    };
    source.start();
  }

  private handleFrame(frame: WorkletFrame) {
    if (frame.type !== "analysis" || !this.source) return;
    if (frame.spectrum) this.latestSpectrum = frame.spectrum;
    const metrics = this.decoder.process({
      timestamp: frame.timestamp,
      levelDb: frame.levelDb,
      purity: frame.purity,
      frequency: frame.frequency,
      prominenceDb: frame.prominenceDb,
    });
    // Decoder runs at audio cadence; the UI is deliberately capped near 30 fps.
    if (frame.timestamp - this.lastUiUpdate < 32 && !frame.spectrum) return;
    this.lastUiUpdate = frame.timestamp;
    this.callbacks.onAnalysis({ ...frame, spectrum: this.latestSpectrum, metrics });
    this.latestSpectrum = undefined;
  }
}
