import { describe, expect, it } from "vitest";
import {
  HAM_MESSAGE_EVIDENCE_CAP,
  hamMessageEvidence,
  interpretHamCopy,
} from "./hamInterpreter";

describe("ham copy interpreter", () => {
  it("expands a conversational QSO without changing the decoder copy", () => {
    const transcript = "BK RR FB STEVE UR RST 5NN 5NN OP IS ED ED QTH I";
    const interpreted = interpretHamCopy(transcript);

    expect(interpreted.transcript).toBe(transcript);
    expect(interpreted.expandedText).toBe(
      "break; back to you roger roger fine business STEVE your signal report is " +
      "599 599 operator is ED ED my location is I",
    );
    expect(interpreted.markers.map(({ token, expansion, category }) => ({
      token,
      expansion,
      category,
    }))).toEqual([
      { token: "BK", expansion: "break; back to you", category: "abbreviation" },
      { token: "RR", expansion: "roger roger", category: "abbreviation" },
      { token: "FB", expansion: "fine business", category: "abbreviation" },
      { token: "UR", expansion: "your / you're", category: "abbreviation" },
      { token: "RST", expansion: "readability, strength, tone", category: "abbreviation" },
      { token: "5NN", expansion: "599", category: "cut-number" },
      { token: "5NN", expansion: "599", category: "cut-number" },
      { token: "OP", expansion: "operator", category: "abbreviation" },
      { token: "QTH", expansion: "my location is", category: "q-signal" },
    ]);
  });

  it("distinguishes a bracketed prosign and Q-signal question form", () => {
    const interpreted = interpretHamCopy("CQ <BT> QTH?");

    expect(interpreted.expandedText).toBe("calling any station separator What is your location?");
    expect(interpreted.markers).toEqual([
      { token: "CQ", expansion: "calling any station", category: "abbreviation" },
      { token: "<BT>", expansion: "separator", category: "prosign" },
      { token: "QTH?", expansion: "What is your location?", category: "q-signal" },
    ]);
  });

  it("does not duplicate an explicit IS after a statement-form Q signal", () => {
    const interpreted = interpretHamCopy("QTH IS BROADSTAIRS QTC?");

    expect(interpreted.expandedText).toBe(
      "my location is BROADSTAIRS How many messages have you to send?",
    );
  });

  it("does not collapse written AA into the run-together AA prosign", () => {
    const interpreted = interpretHamCopy("AA <AA>");

    expect(interpreted.expandedText).toBe("all after new line");
    expect(interpreted.markers).toEqual([
      { token: "AA", expansion: "all after", category: "abbreviation" },
      { token: "<AA>", expansion: "new line", category: "prosign" },
    ]);
  });

  it("does not infer run-together prosigns from separately keyed letter groups", () => {
    const transcript = "AS AR BT CL HH KA KN SK SN SOS <AS> <AR> <BT> <KN> <SK> <SOS>";
    const interpreted = interpretHamCopy(transcript);

    expect(interpreted.expandedText).toBe(
      "AS AR BT CL HH KA KN SK SN SOS wait end of message separator " +
      "invitation for the named station to transmit end of contact distress",
    );
    expect(interpreted.markers.map((marker) => marker.token)).toEqual([
      "<AS>", "<AR>", "<BT>", "<KN>", "<SK>", "<SOS>",
    ]);
  });

  it("preserves callsigns, unknown words, ordinary English, case, and punctuation", () => {
    const transcript = "W1AW ZS6ABC called Steve in Broadstairs, HELLO WORLD.";
    const interpreted = interpretHamCopy(transcript);

    expect(interpreted.transcript).toBe(transcript);
    expect(interpreted.expandedText).toBe(transcript);
    expect(interpreted.markers).toEqual([]);
  });

  it("expands only the known terms around a callsign", () => {
    const interpreted = interpretHamCopy("CQ DE W1AW PSE QSL TNX 73");

    expect(interpreted.expandedText).toBe(
      "calling any station from W1AW please acknowledge / confirm receipt thanks best regards",
    );
    expect(interpreted.markers.map((marker) => marker.token)).toEqual([
      "CQ", "DE", "PSE", "QSL", "TNX", "73",
    ]);
  });
});

describe("bounded ham message evidence", () => {
  it("strongly prefers a coherent exchange to gibberish or isolated fragments", () => {
    const coherent = hamMessageEvidence(
      "BK RR FB STEVE UR RST 5NN 5NN OP IS ED ED QTH I",
    );
    const fragments = hamMessageEvidence("RST QTH");
    const gibberish = hamMessageEvidence("XAZ VVV 1234 JQJQ");

    expect(coherent.score).toBeLessThanOrEqual(HAM_MESSAGE_EVIDENCE_CAP);
    expect(coherent.score).toBeGreaterThanOrEqual(0.08);
    expect(coherent.score).toBeGreaterThan(fragments.score * 5);
    expect(fragments.score).toBeGreaterThan(0);
    expect(gibberish.score).toBe(0);
    expect(gibberish.details).toEqual([]);
  });

  it("recognizes coherent CQ/DE/callsign ordering but not a callsign alone", () => {
    const directed = hamMessageEvidence("CQ CQ DE W1AW");
    const callsignOnly = hamMessageEvidence("W1AW");

    expect(directed.details.some((detail) =>
      detail.label === "general call and station identification"
    )).toBe(true);
    expect(directed.score).toBeGreaterThan(callsignOnly.score);
    expect(callsignOnly.score).toBe(0);
  });

  it("caps the raw evidence budget independently of convergence", () => {
    const strongest = hamMessageEvidence(
      "CQ CQ DE W1AW BK RR FB UR RST 5NN 5NN OP IS QTH TNX 73",
    );

    expect(strongest.score).toBeLessThanOrEqual(HAM_MESSAGE_EVIDENCE_CAP);
    expect(strongest.score).toBe(HAM_MESSAGE_EVIDENCE_CAP);
  });
});
