import type {
  OfflineDecodeOptions,
  OfflineDecodeProgress,
  OfflineDecodeResult as ClassicalOfflineDecodeResult,
  OfflineWorkerRequest,
  OfflineWorkerResponse,
} from "./offline/optimizerCore";
import type { NeuralDecodeAdvisory } from "./offline/neuralAdvisory";

const ANALYSIS_SAMPLE_RATE = 12_000;

export const DEFAULT_OFFLINE_DECODE_OPTIONS: OfflineDecodeOptions = {
  minimumFrequency: 200,
  maximumFrequency: 4000,
  wpmHint: 20,
  languageAssist: true,
};

export interface OptimizeMorseBufferOptions extends Partial<OfflineDecodeOptions> {
  signal?: AbortSignal;
  onProgress?: (progress: OfflineDecodeProgress) => void;
}

/**
 * Neural copy is deliberately additive. The classical result remains intact
 * and owns playback channel, frequency, WPM, confidence, and convergence.
 */
export interface OfflineDecodeResult extends ClassicalOfflineDecodeResult {
  neuralAdvisories?: NeuralDecodeAdvisory[];
}

/**
 * Runs the expensive file-only receiver search away from both the React thread
 * and the real-time AudioWorklet. The original AudioBuffer is never modified;
 * it can be replayed after the optimizer returns timestamped transcript tokens.
 */
export async function optimizeMorseBuffer(
  buffer: AudioBuffer,
  options: OptimizeMorseBufferOptions = {},
): Promise<OfflineDecodeResult> {
  const { signal, onProgress, ...decodeOverrides } = options;
  if (signal?.aborted) throw new DOMException("Decode cancelled.", "AbortError");
  onProgress?.({ phase: "survey", fraction: 0, detail: "Preparing analysis channels" });

  const sampleRate = Math.min(ANALYSIS_SAMPLE_RATE, buffer.sampleRate);
  const frameCount = Math.max(1, Math.ceil(buffer.duration * sampleRate));
  // Preserve normal stereo files so the worker can evaluate L, R, and mono.
  // Web Audio applies its standard speaker downmix when the source has >2 tracks.
  const channelCount = Math.max(1, Math.min(2, buffer.numberOfChannels));
  const renderContext = new OfflineAudioContext(channelCount, frameCount, sampleRate);
  const source = renderContext.createBufferSource();
  source.buffer = buffer;
  source.connect(renderContext.destination);
  source.start();
  const rendered = await renderContext.startRendering();
  if (signal?.aborted) throw new DOMException("Decode cancelled.", "AbortError");

  // Copy before transfer: getChannelData() is owned by the rendered AudioBuffer.
  const channels = Array.from({ length: rendered.numberOfChannels }, (_, channel) =>
    rendered.getChannelData(channel).slice()
  );
  const worker = new Worker(new URL("./offline/optimizer.worker.ts", import.meta.url), { type: "module" });

  return await new Promise<OfflineDecodeResult>((resolve, reject) => {
    const finish = () => {
      signal?.removeEventListener("abort", abort);
      worker.terminate();
    };
    const abort = () => {
      finish();
      reject(new DOMException("Decode cancelled.", "AbortError"));
    };
    signal?.addEventListener("abort", abort, { once: true });
    worker.onmessage = (event: MessageEvent<OfflineWorkerResponse>) => {
      const message = event.data;
      if (message.type === "progress") {
        onProgress?.(message.progress);
        return;
      }
      finish();
      if (message.type === "result") resolve(message.result as OfflineDecodeResult);
      else reject(new Error(message.message));
    };
    worker.onerror = (event) => {
      finish();
      reject(new Error(event.message || "Offline Morse analysis failed."));
    };

    const request: OfflineWorkerRequest = {
      type: "optimize",
      channels: channels.map((channel) => channel.buffer as ArrayBuffer),
      sampleRate,
      options: {
        ...DEFAULT_OFFLINE_DECODE_OPTIONS,
        ...decodeOverrides,
        minimumFrequency: Math.max(
          40,
          decodeOverrides.minimumFrequency ?? DEFAULT_OFFLINE_DECODE_OPTIONS.minimumFrequency ?? 200,
        ),
        maximumFrequency: Math.min(
          sampleRate * 0.45,
          decodeOverrides.maximumFrequency ?? DEFAULT_OFFLINE_DECODE_OPTIONS.maximumFrequency ?? 4000,
        ),
      },
    };
    worker.postMessage(request, request.channels);
  });
}

export type {
  OfflineDecodeCandidate,
  OfflineDecodeOptions,
  OfflineDecodeProgress,
  OfflineDecodeToken,
} from "./offline/optimizerCore";
export type { NeuralDecodeAdvisory } from "./offline/neuralAdvisory";
