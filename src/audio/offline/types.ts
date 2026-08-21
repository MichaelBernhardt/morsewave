// Compatibility barrel for consumers that prefer a types-only module. The
// optimizer core owns the protocol so the worker and UI wrapper cannot drift.
export type {
  OfflineDecodeCandidate,
  OfflineDecodeOptions,
  OfflineDecodeProgress,
  OfflineDecodeResult,
  OfflineDecodeToken,
  OfflineWorkerRequest,
  OfflineWorkerResponse,
} from "./optimizerCore";
