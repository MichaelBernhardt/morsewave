import {
  HAM_CW_ABBREVIATIONS,
  HAM_PROSIGN_NAMES,
  HAM_Q_SIGNALS,
  normalizeHamToken,
} from "./hamVocabulary";

/**
 * Advisory readings follow the ARRL Quickstart Q-signal/RST tables, ARRL
 * Radio Lab operating abbreviations, and the ARRL NTS CW procedural guide.
 * They are annotations only; ITU Morse symbols and received spacing remain
 * the authoritative copy.
 */

export type HamMarkerCategory =
  | "prosign"
  | "operating-signal"
  | "q-signal"
  | "abbreviation"
  | "cut-number";

export interface HamMarker {
  /** The literal token in the decoder copy. */
  readonly token: string;
  /** A human-readable advisory reading; never fed back into the decoder. */
  readonly expansion: string;
  readonly category: HamMarkerCategory;
}

export interface HamInterpretation {
  /** The physical decoder copy, returned byte-for-byte unchanged. */
  readonly transcript: string;
  /** A separate, advisory reading with only exact known tokens expanded. */
  readonly expandedText: string;
  readonly markers: readonly HamMarker[];
}

export type HamEvidenceKind = "sequence";

export interface HamEvidenceDetail {
  readonly kind: HamEvidenceKind;
  readonly label: string;
  readonly tokens: readonly string[];
  /** Additive contribution before the overall evidence cap is applied. */
  readonly contribution: number;
}

export interface HamMessageEvidence {
  /**
   * A bounded evidence budget for otherwise credible physical hypotheses.
   * It is not a probability; callers apply a smaller ranking weight and must
   * retain independent timing/signal convergence checks.
   */
  readonly score: number;
  readonly cap: number;
  readonly details: readonly HamEvidenceDetail[];
}

/** Maximum raw evidence returned to a caller before its ranking weight. */
export const HAM_MESSAGE_EVIDENCE_CAP = 0.1;

interface QSignalReading {
  readonly statement: string;
  readonly question: string;
}

interface ParsedToken {
  readonly raw: string;
  readonly normalized: string;
  readonly bracketed: boolean;
  readonly question: boolean;
}

interface MarkerReading {
  readonly expansion: string;
  readonly category: HamMarkerCategory;
}

const TOKEN_PATTERN = /<[A-Za-z]+>|[A-Za-z0-9]+(?:\/[A-Za-z0-9]+)*\??/g;

const PROSIGN_EXPANSIONS: Readonly<Record<string, string>> = Object.freeze({
  AA: "new line",
  AR: "end of message",
  AS: "wait",
  BK: "break",
  BT: "separator",
  CL: "closing station",
  HH: "error; correction follows",
  K: "invitation to transmit",
  KA: "attention; message begins",
  KN: "invitation for the named station to transmit",
  SK: "end of contact",
  SN: "understood",
  SOS: "distress",
});

/**
 * Common amateur-service readings. Less common entries in HAM_Q_SIGNALS are
 * still recognized as exact Q-signals, but are deliberately left unexpanded
 * rather than assigning a guessed meaning.
 */
const Q_SIGNAL_EXPANSIONS: Readonly<Record<string, QSignalReading>> = Object.freeze({
  QRA: { statement: "my station name is", question: "What is your station name?" },
  QRG: { statement: "your exact frequency is", question: "What is my exact frequency?" },
  QRH: { statement: "your frequency varies", question: "Does my frequency vary?" },
  QRI: { statement: "your tone is", question: "How is my tone?" },
  QRK: { statement: "your readability is", question: "What is my readability?" },
  QRL: { statement: "this frequency is busy", question: "Is this frequency busy?" },
  QRM: { statement: "interference from other stations", question: "Are you receiving interference?" },
  QRN: { statement: "static interference", question: "Are you troubled by static?" },
  QRO: { statement: "increase power", question: "Shall I increase power?" },
  QRP: { statement: "decrease power / low-power operation", question: "Shall I decrease power?" },
  QRQ: { statement: "send faster", question: "Shall I send faster?" },
  QRS: { statement: "send more slowly", question: "Shall I send more slowly?" },
  QRT: { statement: "stop sending", question: "Shall I stop sending?" },
  QRU: { statement: "I have nothing for you", question: "Have you anything for me?" },
  QRV: { statement: "I am ready", question: "Are you ready?" },
  QRX: { statement: "stand by / call again at", question: "When will you call again?" },
  QRZ: { statement: "you are being called by", question: "Who is calling me?" },
  QSA: { statement: "your signal strength is", question: "What is my signal strength?" },
  QSB: { statement: "your signal is fading", question: "Is my signal fading?" },
  QSK: { statement: "I can hear between signals", question: "Can you hear me between signals?" },
  QSL: { statement: "acknowledge / confirm receipt", question: "Can you acknowledge receipt?" },
  QSM: { statement: "repeat your last message", question: "Shall I repeat my last message?" },
  QSO: {
    statement: "radio contact / communicate directly",
    question: "Can you communicate directly?",
  },
  QSP: { statement: "I will relay", question: "Will you relay?" },
  QST: {
    statement: "bulletin / general announcement to all amateurs",
    question: "bulletin / general announcement to all amateurs",
  },
  QSY: { statement: "change frequency", question: "Shall I change frequency?" },
  QTA: { statement: "cancel the message", question: "Shall I cancel the message?" },
  QTC: { statement: "I have messages for you", question: "How many messages have you to send?" },
  QTH: { statement: "my location is", question: "What is your location?" },
  QTR: { statement: "the correct time is", question: "What is the correct time?" },
});

const ABBREVIATION_EXPANSIONS: Readonly<Record<string, string>> = Object.freeze({
  AA: "all after",
  ABT: "about",
  ADR: "address",
  AFTR: "after",
  AGN: "again",
  ANT: "antenna",
  B4: "before",
  BK: "break; back to you",
  BN: "been",
  CFM: "confirm",
  CLD: "called",
  CLG: "calling",
  CLR: "clear",
  CQ: "calling any station",
  CU: "see you",
  CUD: "could",
  CUL: "see you later",
  CW: "continuous wave",
  DE: "from",
  DR: "dear",
  DX: "long-distance contact",
  ES: "and",
  FB: "fine business",
  FER: "for",
  FM: "from",
  GB: "goodbye",
  GE: "good evening",
  GL: "good luck",
  GM: "good morning",
  GN: "good night",
  GND: "ground",
  GUD: "good",
  HI: "laughter",
  HR: "here",
  HV: "have",
  HW: "how",
  MNI: "many",
  MSG: "message",
  NIL: "nothing",
  NM: "no more",
  NR: "number",
  NW: "now",
  OM: "old man",
  OP: "operator",
  OPR: "operator",
  PSE: "please",
  PWR: "power",
  RCD: "received",
  RCVR: "receiver",
  RFI: "radio-frequency interference",
  RIG: "station equipment",
  RPRT: "report",
  RPT: "repeat",
  RR: "roger roger",
  RST: "readability, strength, tone",
  RX: "receiver",
  SKED: "schedule",
  SRI: "sorry",
  STN: "station",
  TFC: "traffic",
  TMW: "tomorrow",
  TKS: "thanks",
  TNX: "thanks",
  TU: "thank you",
  TX: "transmitter",
  UR: "your / you're",
  URS: "yours",
  VY: "very",
  WDS: "words",
  WKD: "worked",
  WKG: "working",
  WL: "will",
  WUD: "would",
  WX: "weather",
  XCVR: "transceiver",
  XMTR: "transmitter",
  XTAL: "crystal",
  XYL: "wife",
  YL: "young lady",
  "73": "best regards",
  "88": "love and kisses",
});

const OPERATING_SIGNAL_EXPANSIONS: Readonly<Record<string, string>> = Object.freeze({
  K: "go ahead; any station may reply",
  R: "received; roger",
});

// RR is standard conversational CW, but is intentionally absent from the
// decoder's lexical scoring set because a two-letter prior is too ambiguous
// during physical dot/dash recovery.
const INTERPRETER_ONLY_ABBREVIATIONS: ReadonlySet<string> = new Set(["RR"]);

function parseToken(raw: string): ParsedToken {
  const question = raw.endsWith("?");
  const withoutQuestion = question ? raw.slice(0, -1) : raw;
  const bracketed = withoutQuestion.startsWith("<") && withoutQuestion.endsWith(">");
  return {
    raw,
    normalized: normalizeHamToken(withoutQuestion),
    bracketed,
    question,
  };
}

function transcriptTokens(transcript: string): ParsedToken[] {
  return [...transcript.matchAll(TOKEN_PATTERN)].map((match) => parseToken(match[0]));
}

function markerReading(token: ParsedToken): MarkerReading | undefined {
  const { bracketed, normalized, question } = token;

  if (bracketed) {
    if (!question && HAM_PROSIGN_NAMES.has(normalized)) {
      return {
        expansion: PROSIGN_EXPANSIONS[normalized] ?? `${normalized} prosign`,
        category: "prosign",
      };
    }
    return undefined;
  }

  if (!question && normalized === "5NN") {
    return { expansion: "599", category: "cut-number" };
  }

  if (HAM_Q_SIGNALS.has(normalized)) {
    const reading = Q_SIGNAL_EXPANSIONS[normalized];
    if (reading) {
      return {
        expansion: question ? reading.question : reading.statement,
        category: "q-signal",
      };
    }
    // Keep uncommon but valid Q-signals visible in advisory text. The marker
    // still lets a details panel identify their category without inventing a
    // semantic expansion that may vary by service.
    return { expansion: token.raw, category: "q-signal" };
  }

  // Written-out character groups are not the same physical symbols as their
  // run-together bracketed counterparts (`AA` versus `<AA>`, for example).
  const writtenAbbreviation = ABBREVIATION_EXPANSIONS[normalized];
  if (!question && HAM_CW_ABBREVIATIONS.has(normalized) && writtenAbbreviation) {
    return { expansion: writtenAbbreviation, category: "abbreviation" };
  }

  if (!question && OPERATING_SIGNAL_EXPANSIONS[normalized]) {
    return {
      expansion: OPERATING_SIGNAL_EXPANSIONS[normalized],
      category: "operating-signal",
    };
  }

  const isKnownAbbreviation = HAM_CW_ABBREVIATIONS.has(normalized) ||
    INTERPRETER_ONLY_ABBREVIATIONS.has(normalized);
  const abbreviation = writtenAbbreviation;
  if (!question && isKnownAbbreviation && abbreviation) {
    return { expansion: abbreviation, category: "abbreviation" };
  }

  return undefined;
}

/**
 * Build a separate operator aid from literal decoder copy. This function is
 * deterministic and token-exact: it does no spell-correction, word splitting,
 * or callsign guessing, and it never changes `transcript`.
 */
export function interpretHamCopy(transcript: string): HamInterpretation {
  const markers: HamMarker[] = [];
  const tokenExpansion = transcript.replace(TOKEN_PATTERN, (raw) => {
    const reading = markerReading(parseToken(raw));
    if (!reading) return raw;
    markers.push(Object.freeze({
      token: raw,
      expansion: reading.expansion,
      category: reading.category,
    }));
    return reading.expansion;
  });
  const expandedText = tokenExpansion
    .replace(/\byour \/ you're readability, strength, tone\b/g, "your signal report is")
    .replace(
      /\b(your signal report is|my location is|my station name is|the correct time is) IS\b/g,
      "$1",
    )
    .replace(/\boperator IS\b/g, "operator is");

  return Object.freeze({
    transcript,
    expandedText,
    markers: Object.freeze(markers),
  });
}

function isLikelyCallsign(value: string) {
  if (!/^[A-Z0-9]{3,10}(?:\/[A-Z0-9]{1,6})?$/.test(value)) return false;
  const base = value.split("/")[0];
  return /[A-Z]/.test(base) && /\d/.test(base) && value !== "5NN";
}

function isSignalReport(value: string) {
  return value === "5NN" || /^[1-5][1-9N][1-9N]$/.test(value);
}

/**
 * Return bounded lexical evidence for ranking close, physically credible
 * decoder variants. The raw evidence is capped at 0.10 and never penalizes a
 * transcript. Callers must apply a smaller weight and retain independent
 * timing, valid-Morse, and confidence gates before declaring convergence.
 */
export function hamMessageEvidence(transcript: string): HamMessageEvidence {
  const parsed = transcriptTokens(transcript);
  const words = parsed.map((token) => token.normalized);
  const details: HamEvidenceDetail[] = [];

  const add = (
    kind: HamEvidenceKind,
    label: string,
    start: number,
    length: number,
    contribution: number,
  ) => {
    details.push(Object.freeze({
      kind,
      label,
      tokens: Object.freeze(parsed.slice(start, start + length).map((token) => token.raw)),
      contribution,
    }));
  };

  // Isolated abbreviations are cheap to create accidentally, so this layer
  // scores only coherent operating sequences. Exact individual terms are
  // already handled by the smaller general-language tie-breaker.
  for (let index = 0; index < words.length; index += 1) {
    if (words[index] === "BK" && words[index + 1] === "RR" && words[index + 2] === "FB") {
      add("sequence", "break / acknowledgement / fine business", index, 3, 0.015);
    } else if (words[index] === "RR" && words[index + 1] === "FB" && words[index - 1] !== "BK") {
      add("sequence", "acknowledgement / fine business", index, 2, 0.008);
    }

    if (words[index] === "UR" && words[index + 1] === "RST" && isSignalReport(words[index + 2] ?? "")) {
      add("sequence", "your signal report", index, 3, 0.03);
    } else if (words[index] === "RST" && isSignalReport(words[index + 1] ?? "") && words[index - 1] !== "UR") {
      add("sequence", "signal report", index, 2, 0.018);
    }

    if (isSignalReport(words[index]) && isSignalReport(words[index + 1] ?? "")) {
      add("sequence", "repeated signal report", index, 2, 0.012);
    }

    if (words[index] === "OP" && words[index + 1] === "IS") {
      add("sequence", "operator identification", index, 2, 0.016);
    }

    if (words[index] === "QTH") {
      add("sequence", "location exchange", index, 1, 0.01);
    }

    const longCqCall = words[index] === "CQ" && words[index + 1] === "CQ" &&
      words[index + 2] === "DE" && isLikelyCallsign(words[index + 3] ?? "");
    const cqCall = words[index] === "CQ" && words[index + 1] === "DE" &&
      isLikelyCallsign(words[index + 2] ?? "") && words[index - 1] !== "CQ";
    if (longCqCall) {
      add("sequence", "general call and station identification", index, 4, 0.04);
    } else if (cqCall) {
      add("sequence", "general call and station identification", index, 3, 0.035);
    }

    if (words[index] === "DE" && isLikelyCallsign(words[index + 1] ?? "") && words[index - 1] !== "CQ") {
      add("sequence", "station identification", index, 2, 0.018);
    }
  }

  const score = Math.min(
    HAM_MESSAGE_EVIDENCE_CAP,
    details.reduce((sum, detail) => sum + detail.contribution, 0),
  );
  return Object.freeze({
    score,
    cap: HAM_MESSAGE_EVIDENCE_CAP,
    details: Object.freeze(details),
  });
}
