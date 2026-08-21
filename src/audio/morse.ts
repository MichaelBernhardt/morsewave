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
  "...-.-": "<SK>",
  "-.-.-": "<KA>",
  "........": "<HH>",
});

export const TEXT_TO_MORSE: Readonly<Record<string, string>> = Object.freeze(
  Object.fromEntries(
    Object.entries(MORSE_TO_TEXT)
      .filter(([, value]) => value.length === 1)
      .map(([pattern, value]) => [value, pattern]),
  ),
);

export function decodeMorse(pattern: string): string {
  return MORSE_TO_TEXT[pattern] ?? "�";
}

const MORSE_PATTERNS = Object.keys(MORSE_TO_TEXT);

export function isMorseCharacter(pattern: string): boolean {
  return Object.hasOwn(MORSE_TO_TEXT, pattern);
}

export function isMorsePrefix(pattern: string): boolean {
  return MORSE_PATTERNS.some((candidate) => candidate.startsWith(pattern));
}

export function encodeText(text: string): string[][] {
  return text
    .trim()
    .toUpperCase()
    .split(/\s+/)
    .map((word) => [...word].map((character) => TEXT_TO_MORSE[character]).filter(Boolean));
}
