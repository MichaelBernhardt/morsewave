import { describe, expect, it } from "vitest";
import { HAM_CW_ABBREVIATIONS, HAM_Q_SIGNALS, isHamCwTerm } from "./hamVocabulary";
import {
  annotateMorseValue,
  decodeMorse,
  encodeText,
  isMorseCharacter,
  isMorsePrefix,
  PROSIGN_TO_MORSE,
} from "./morse";

describe("International Morse and amateur-CW procedural signs", () => {
  it("keeps printable aliases unless the ham prior is enabled", () => {
    const aliases = [
      ["-...-", "=", "<BT>"],
      [".-.-.", "+", "<AR>"],
      [".-...", "&", "<AS>"],
      ["-.--.", "(", "<KN>"],
    ] as const;
    for (const [pattern, printable, prosign] of aliases) {
      expect(decodeMorse(pattern)).toBe(printable);
      expect(decodeMorse(pattern, true)).toBe(prosign);
      expect(annotateMorseValue(pattern, printable, true)).toBe(prosign);
    }
  });

  it("decodes unambiguous procedural signs in either mode", () => {
    expect(decodeMorse("-...-.-")).toBe("<BK>");
    expect(decodeMorse("...---...")).toBe("<SOS>");
    expect(isMorseCharacter("...---...")).toBe(true);
    expect(isMorsePrefix("...---..")).toBe(true);
  });

  it("encodes bracketed prosigns as one run-together Morse group", () => {
    expect(encodeText("CQ <BT> QTH")).toEqual([
      ["-.-.", "--.-"],
      ["-...-"],
      ["--.-", "-", "...."],
    ]);
    expect(encodeText("<SOS>")).toEqual([["...---..."]]);
    expect(encodeText("A<FOO>B")).toEqual([[
      ".-", "..-.", "---", "---", "-...",
    ]]);
    expect(PROSIGN_TO_MORSE["<KN>"]).toBe("-.--.");
  });

  it("contains the operational markers used to rank amateur copy", () => {
    for (const signal of ["QRL", "QRM", "QRN", "QRZ", "QSB", "QSL", "QSO", "QTH"]) {
      expect(HAM_Q_SIGNALS.has(signal)).toBe(true);
    }
    for (const abbreviation of ["CQ", "DE", "ES", "FB", "OM", "PSE", "RST", "TNX", "VY", "73"]) {
      expect(HAM_CW_ABBREVIATIONS.has(abbreviation)).toBe(true);
      expect(isHamCwTerm(abbreviation)).toBe(true);
    }
    expect(isHamCwTerm("<BT>")).toBe(true);
    expect(isHamCwTerm("K")).toBe(true);
  });
});
