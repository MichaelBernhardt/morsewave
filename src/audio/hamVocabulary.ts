/**
 * Amateur-CW vocabulary used only as a bounded tie-breaker. The physical
 * dot/dash stream remains authoritative.
 *
 * Sources:
 * - ARRL Operating Aids: https://www.arrl.org/operating-aids
 * - ARRL Radio Lab Handbook, Unit 5, tables 5.2/5.3
 * - ARRL NTS Methods and Practices Guidelines, section 3.6
 */

export const HAM_Q_SIGNALS: ReadonlySet<string> = new Set(
  (
    "QRA QRG QRH QRI QRJ QRK QRL QRM QRN QRO QRP QRQ QRS QRT QRU QRV QRW QRX QRZ " +
    "QSA QSB QSD QSG QSK QSL QSM QSN QSO QSP QSQ QSR QSS QST QSU QSV QSW QSX QSY QSZ " +
    "QTA QTB QTC QTH QTR QNA QNB QNC QND QNE QNF QNG QNH QNI QNJ QNK QNL QNM QNN QNO " +
    "QNP QNQ QNR QNS QNT QNU QNV QNW QNX QNY QNZ"
  ).split(" "),
);

export const HAM_CW_ABBREVIATIONS: ReadonlySet<string> = new Set(
  (
    "AA AB ABT ADEE ADDEE ADR AFTR AGN ANC ANT ARL BCI BCL BK BN BTWN BUG B4 C CFM CK CL " +
    "CLD CLG CLR CQ CTY CU CUD CUL CW DE DLD DLVD DR DX EC EE ES FB FER FM GA GB GBA GE GG " +
    "GL GM GN GND GRP GUD HI HR HV HW LID MA MILS MGR MNI MSG N NCS NIL NM NR NW OB OC OM " +
    "OP OPR OT PBL PLC PLS PREC PRFX PFX PSE PWR PX R RCD RCVR REF RFI RIG RPRT RPT RTTY " +
    "RX RST SASE SED SIG SINE SKED SRI SSB STN SVC T TFC TMW TNX TKS TT TU TVI TX TXT UR URS " +
    "VFO VY WA WB WD WDS WKD WKG WL WUD WX XCVR XMTR XTAL XYL YF YL 73 88"
  ).split(" "),
);

export const HAM_QSO_WORDS: ReadonlySet<string> = new Set(
  (
    "AMATEUR ANTENNA BAND BREAK CALL CALLING CITY CONTEST COPY COUNTRY DIPOLE FADING FIST " +
    "FREQUENCY GRID GROUND KEY KEYER LOCATION LOG MESSAGE NAME NET OPERATOR POWER RADIO REPORT " +
    "RIG SIGNAL SPEED STATION TRAFFIC TRANSMITTER WEATHER WATTS"
  ).split(" "),
);

export const HAM_PROSIGN_NAMES: ReadonlySet<string> = new Set(
  "AA AR AS BK BT CL HH K KA KN SK SN SOS"
    .split(" "),
);

export const HAM_CW_TERMS: ReadonlySet<string> = new Set([
  ...HAM_Q_SIGNALS,
  ...HAM_CW_ABBREVIATIONS,
  ...HAM_QSO_WORDS,
  ...HAM_PROSIGN_NAMES,
]);

export function normalizeHamToken(value: string) {
  return value.toUpperCase().replace(/^<|>$/g, "");
}

export function isHamCwTerm(value: string) {
  return HAM_CW_TERMS.has(normalizeHamToken(value));
}

/**
 * One-letter operating signals are too ambiguous to act as lexical evidence
 * on their own. They remain available for annotation, while general scoring
 * starts at three characters (plus the distinctive 73/88 groups) and is
 * always subordinate to the received Morse timing.
 */
export function isHamCwScoringTerm(value: string) {
  const normalized = normalizeHamToken(value);
  return (normalized.length >= 3 || /^(73|88)$/.test(normalized)) && HAM_CW_TERMS.has(normalized);
}
