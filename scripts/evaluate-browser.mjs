#!/usr/bin/env node

import { resolve } from "node:path";

const [, , recordingArgument, timeoutArgument = "90"] = process.argv;

if (!recordingArgument) {
  console.error("Usage: node scripts/evaluate-browser.mjs <recording> [timeout-seconds]");
  process.exit(2);
}

const recording = resolve(recordingArgument);
const timeoutMs = Number(timeoutArgument) * 1000;
const endpoint = process.env.MORSEWAVE_CDP ?? "http://127.0.0.1:9222";
const appUrl = process.env.MORSEWAVE_URL ?? "http://127.0.0.1:1420/";

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

async function findAppTarget() {
  const targets = await fetch(`${endpoint}/json/list`).then((response) => response.json());
  const target = targets.find((candidate) => candidate.type === "page" && candidate.url.startsWith(appUrl));
  if (!target) throw new Error(`No Morsewave page is attached at ${endpoint}. Open ${appUrl} first.`);
  return target;
}

const target = await findAppTarget();
const socket = new WebSocket(target.webSocketDebuggerUrl);
const pending = new Map();
const exceptions = [];
let commandId = 0;

socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const request = pending.get(message.id);
    if (!request) return;
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    exceptions.push(message.params.exceptionDetails.text);
  }
});

await new Promise((resolveOpen, rejectOpen) => {
  socket.addEventListener("open", resolveOpen, { once: true });
  socket.addEventListener("error", rejectOpen, { once: true });
});

function command(method, params = {}) {
  const id = ++commandId;
  return new Promise((resolveCommand, rejectCommand) => {
    pending.set(id, { resolve: resolveCommand, reject: rejectCommand });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await command("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function waitUntil(predicate, limitMs, label) {
  const deadline = Date.now() + limitMs;
  while (Date.now() < deadline) {
    const result = await predicate();
    if (result) return result;
    await delay(200);
  }
  throw new Error(`Timed out waiting for ${label}.`);
}

try {
  await command("Runtime.enable");
  await command("DOM.enable");
  await evaluate(`localStorage.setItem("morsewave.settings", JSON.stringify({
    autoTune: true,
    frequency: 700,
    sensitivity: 82,
    adaptiveSpeed: true,
    wpmHint: 18,
    handKeyAssist: true,
    languageAssist: true,
    theme: "dark"
  })); location.reload(); true`);
  await waitUntil(
    () => evaluate(`document.readyState === "complete" && Boolean(document.querySelector('input[type="file"]'))`),
    10_000,
    "application reload",
  );

  const documentNode = await command("DOM.getDocument", { depth: -1, pierce: true });
  const inputNode = await command("DOM.querySelector", {
    nodeId: documentNode.root.nodeId,
    selector: 'input[type="file"]',
  });
  if (!inputNode.nodeId) throw new Error("The recording input was not found.");
  await command("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [recording] });

  const finalState = await waitUntil(
    () => evaluate(`(() => {
      const status = document.querySelector('.status-chip')?.textContent?.trim() ?? '';
      if (status.includes('Buffer complete') || status.includes('Input fault')) return status;
      return '';
    })()`),
    timeoutMs,
    "recording completion",
  );

  const report = await evaluate(`(() => ({
    recording: ${JSON.stringify(recording)},
    status: document.querySelector('.status-chip')?.textContent?.trim() ?? '',
    source: document.querySelector('.source-context')?.textContent?.replace(/\\s+/g, ' ').trim() ?? '',
    transcript: document.querySelector('.transcript p')?.textContent?.trim() ?? '',
    metrics: Array.from(document.querySelectorAll('.metric-card')).map((element) =>
      element.textContent?.replace(/\\s+/g, ' ').trim() ?? ''
    ),
    scope: document.querySelector('.scope-readout')?.textContent?.replace(/\\s+/g, ' ').trim() ?? '',
    error: document.querySelector('[role="alert"]')?.textContent?.replace(/\\s+/g, ' ').trim() ?? ''
  }))()`);
  report.status = finalState;
  report.runtimeExceptions = exceptions;
  console.log(JSON.stringify(report, null, 2));
} finally {
  socket.close();
}
