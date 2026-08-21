/// <reference lib="webworker" />

import {
  optimizeMorseRecording,
  type OfflineWorkerRequest,
  type OfflineWorkerResponse,
} from "./optimizerCore";

const worker = self as DedicatedWorkerGlobalScope;

worker.onmessage = (event: MessageEvent<OfflineWorkerRequest>) => {
  if (event.data?.type !== "optimize") return;
  try {
    const channels = event.data.channels.map((buffer) => new Float32Array(buffer));
    const result = optimizeMorseRecording(
      channels,
      event.data.sampleRate,
      event.data.options,
      (progress) => {
        const response: OfflineWorkerResponse = { type: "progress", progress };
        worker.postMessage(response);
      },
    );
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
