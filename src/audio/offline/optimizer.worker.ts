/// <reference lib="webworker" />

import {
  optimizeMorseRecording,
  type OfflineDecodeProgress,
  type OfflineDecodeResult,
  type OfflineWorkerRequest,
  type OfflineWorkerResponse,
} from "./optimizerCore";
import {
  selectNeuralAdvisoryLanes,
  type NeuralDecodeAdvisory,
  type NeuralWorkerRequest,
  type NeuralWorkerResponse,
} from "./neuralAdvisory";

const worker = self as DedicatedWorkerGlobalScope;
const NEURAL_INFERENCE_BUDGET_MS = 60_000;

interface AdvisoryOfflineDecodeResult extends OfflineDecodeResult {
  neuralAdvisories?: NeuralDecodeAdvisory[];
}

function postProgress(progress: OfflineDecodeProgress) {
  const response: OfflineWorkerResponse = { type: "progress", progress };
  worker.postMessage(response);
}

function decodeNeuralAdvisories(
  classicalResult: OfflineDecodeResult,
  channels: Float32Array[],
  sampleRate: number,
) : Promise<NeuralDecodeAdvisory[]> {
  const lanes = selectNeuralAdvisoryLanes(classicalResult, channels, sampleRate);
  if (!lanes.length) return Promise.resolve([]);

  return new Promise<NeuralDecodeAdvisory[]>((resolve) => {
    let settled = false;
    const neuralWorker = new Worker(new URL("./cwformer.worker.ts", import.meta.url), {
      type: "module",
      name: "morsewave-cwformer",
    });
    const finish = (advisories: NeuralDecodeAdvisory[]) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      neuralWorker.terminate();
      resolve(advisories);
    };
    const timeout = setTimeout(() => {
      console.warn("CWformer advisory exceeded its 60-second hard budget.");
      finish([]);
    }, NEURAL_INFERENCE_BUDGET_MS);
    neuralWorker.onmessage = (event: MessageEvent<NeuralWorkerResponse>) => {
      const message = event.data;
      if (message.type === "progress") {
        postProgress({
          phase: "refine",
          fraction: 0.91 + 0.08 * message.fraction,
          detail: message.detail,
        });
      } else if (message.type === "result") finish(message.advisories);
      else {
        console.warn(`CWformer advisory unavailable: ${message.message}`);
        finish([]);
      }
    };
    neuralWorker.onerror = (event) => {
      console.warn("CWformer advisory worker failed.", event.message);
      finish([]);
    };
    const request: NeuralWorkerRequest = {
      type: "decode",
      lanes,
      channels: channels.map((channel) => channel.buffer as ArrayBuffer),
      sampleRate,
    };
    try {
      neuralWorker.postMessage(request, request.channels);
    } catch (cause) {
      console.warn("CWformer advisory could not be started.", cause);
      finish([]);
    }
  });
}

worker.onmessage = async (event: MessageEvent<OfflineWorkerRequest>) => {
  if (event.data?.type !== "optimize") return;
  try {
    const channels = event.data.channels.map((buffer) => new Float32Array(buffer));
    const result: AdvisoryOfflineDecodeResult = optimizeMorseRecording(
      channels,
      event.data.sampleRate,
      event.data.options,
      (progress) => postProgress({
        ...progress,
        phase: progress.phase === "complete" ? "refine" : progress.phase,
        fraction: Math.min(0.9, progress.fraction * 0.9),
        detail: progress.phase === "complete" ? "Classical station lanes locked" : progress.detail,
      }),
    );

    // Neural inference is advisory and fail-closed. Any asset, CSP, runtime, or
    // model failure leaves the complete classical result byte-for-byte intact.
    try {
      const neuralAdvisories = await decodeNeuralAdvisories(
        result,
        channels,
        event.data.sampleRate,
      );
      if (neuralAdvisories.length) result.neuralAdvisories = neuralAdvisories;
    } catch (cause) {
      console.warn("CWformer advisory unavailable; returning classical result.", cause);
    }

    postProgress({ phase: "complete", fraction: 1, detail: "File analysis complete" });
    const response: OfflineWorkerResponse = { type: "result", result };
    worker.postMessage(response);
  } catch (cause) {
    const response: OfflineWorkerResponse = {
      type: "error",
      message: cause instanceof Error ? cause.message : "Offline optimization failed.",
    };
    worker.postMessage(response);
  }
};

export {};
