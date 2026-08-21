export const MORSE_TO_TEXT: Readonly<Record<string, string>> = Object.freeze({
  ".-": "A",
  "-...": "B",
  "-.-.": "C",
  "-..": "D",
  ".": "E",
  "..-.": "F",
  "--.": "G",
  "....": "H",
  "..": "I",
  ".---": "J",
  "-.-": "K",
  ".-..": "L",
  "--": "M",
  "-.": "N",
  "---": "O",
  ".--.": "P",
  "--.-": "Q",
  ".-.": "R",
  "...": "S",
  "-": "T",
  "..-": "U",
  "...-": "V",
  ".--": "W",
  "-..-": "X",
  "-.--": "Y",
  "--..": "Z",
  ".----": "1",
  "..---": "2",
  "...--": "3",
  "....-": "4",
  ".....": "5",
  "-....": "6",
  "--...": "7",
  "---..": "8",
  "----.": "9",
  "-----": "0",
  ".-.-.-": ".",
  "--..--": ",",
  "..--..": "?",
  ".----.": "'",
  "-.-.--": "!",
  "-..-.": "/",
  "-.--.": "(",
  "-.--.-": ")",
  ".-...": "&",
  "---...": ":",
  "-.-.-.": ";",
  "-...-": "=",
  ".-.-.": "+",
  "-....-": "-",
  "..--.-": "_",
  ".-..-.": "\"",
  "...-..-": "$",
  ".--.-.": "@",
  ".-.-": "<AA>",
  "...-.": "<SN>",
  "-.-..-..": "<CL>",
  "...---...": "<SOS>",
  "...-.-": "<SK>",
  "-.-.-": "<KA>",
  "........": "<HH>",
});

/**
 * Procedural readings for patterns that are also assigned printable symbols.
 * In amateur CW, these run-together groups are generally more useful than the
 * punctuation aliases. Generic decoding keeps the printable value unless the
 * caller explicitly enables the ham/prosign prior.
 */
export const MORSE_TO_PROSIGN: Readonly<Record<string, string>> = Object.freeze({
  ".-.-": "<AA>",
  ".-.-.": "<AR>",
  ".-...": "<AS>",
  "-...-": "<BT>",
  "-...-.-": "<BK>",
  "-.-..-..": "<CL>",
  "........": "<HH>",
  "-.-.-": "<KA>",
  "-.--.": "<KN>",
  "...-.-": "<SK>",
  "...-.": "<SN>",
  "...---...": "<SOS>",
});

export const PROSIGN_TO_MORSE: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(Object.entries(MORSE_TO_PROSIGN).map(([pattern, value]) => [value, pattern])),
);

export const TEXT_TO_MORSE: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(
    Object.entries(MORSE_TO_TEXT)
      .filter(([, value]) => value.length === 1)
      .map(([pattern, value]) => [value, pattern]),
  ),
);

export function decodeMorse(pattern: string, preferProsigns = false): string {
  return (preferProsigns ? MORSE_TO_PROSIGN[pattern] : undefined) ??
    MORSE_TO_TEXT[pattern] ?? MORSE_TO_PROSIGN[pattern] ?? "�";
}

export function annotateMorseValue(pattern: string, rawValue: string, preferProsigns = false): string {
  return preferProsigns ? MORSE_TO_PROSIGN[pattern] ?? rawValue : rawValue;
}

const MORSE_PATTERNS = [...new Set([...Object.keys(MORSE_TO_TEXT), ...Object.keys(MORSE_TO_PROSIGN)])];

export function isMorseCharacter(pattern: string): boolean {
  return Object.hasOwn(MORSE_TO_TEXT, pattern) || Object.hasOwn(MORSE_TO_PROSIGN, pattern);
}

export function isMorsePrefix(pattern: string): boolean {
  return MORSE_PATTERNS.some((candidate) => candidate.startsWith(pattern));
}

export function encodeText(text: string): string[][] {
  return text
    .trim()
    .toUpperCase()
    .split(/\s+/)
    .map((word) => (word.match(/<[^>]+>|./g) ?? []).flatMap((character) => {
      const prosign = PROSIGN_TO_MORSE[character];
      if (prosign) return [prosign];
      // Preserve the old forgiving behavior for an unknown bracketed label:
      // ignore the brackets but still encode ordinary characters inside.
      if (character.startsWith("<") && character.endsWith(">")) {
        return [...character.slice(1, -1)].map((value) => TEXT_TO_MORSE[value]).filter(Boolean);
      }
      const encoded = TEXT_TO_MORSE[character];
      return encoded ? [encoded] : [];
    }));
}
