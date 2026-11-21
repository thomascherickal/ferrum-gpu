// Argument builders for the Export and Serve tabs, with the same refusals the server applies, so a
// mistake is caught before a job is started. Pure: `{ args }` or `{ error }`, tested under node.

import { QUANTS } from './train_args.js';
import { blank, text, whole, numeric, deviceFlags } from './fields.js';

/** `export-gguf`: re-write a model in another precision. */
export function exportGgufArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to export (a .gguf file).' };
  if (blank(f.out)) return { error: 'Choose where to write the new file (a path ending in .gguf).' };
  if (text(f.out) === text(f.model)) return { error: 'The new file must be a different file from the model.' };
  const args = ['--model', text(f.model), '--out', text(f.out)];
  if (!blank(f.quant)) {
    if (!QUANTS.includes(f.quant)) return { error: `Precision must be one of ${QUANTS.join(', ')}.` };
    args.push('--quant', f.quant);
  }
  return { args };
}

/** The file name at the end of a path, either separator. */
const base = (p) => text(p).split(/[\\/]/).filter(Boolean).pop() ?? '';

/** Is `p` the program called `name` (`ollama` or `lms`), with or without `.exe`? The server allows either
 *  name for `--bin`; each form here accepts only its own tool. */
export const allowedTool = (p, name) => [name, `${name}.exe`].includes(base(p));

/** `export-ollama`: register a GPT-2 model with Ollama, or only write the Modelfile. */
export function exportOllamaArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to export (a .gguf file).' };
  const args = ['--model', text(f.model)];
  if (!blank(f.modelfile)) {
    args.push('--modelfile-only', text(f.modelfile));
  } else {
    if (blank(f.name)) return { error: 'Name the model as Ollama will know it, or choose a Modelfile to write instead.' };
    if (!/^[A-Za-z0-9][A-Za-z0-9._\-/:]*$/.test(text(f.name))) return { error: 'The Ollama name may use letters, digits and . _ - / : only.' };
    args.push('--name', text(f.name));
  }
  if (!blank(f.bin)) {
    if (!allowedTool(f.bin, 'ollama')) return { error: 'The tool must be the ollama program.' };
    args.push('--bin', text(f.bin));
  }
  return { args };
}

/** `export-lmstudio`: import a GPT-2 model into LM Studio with `lms import`. */
export function exportLmstudioArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to export (a .gguf file).' };
  const args = ['--model', text(f.model)];
  if (!blank(f.repo)) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(text(f.repo))) {
      return { error: 'The LM Studio name is publisher/name, using letters, digits and . _ - only.' };
    }
    args.push('--repo', text(f.repo));
  }
  if (!blank(f.bin)) {
    if (!allowedTool(f.bin, 'lms')) return { error: 'The tool must be the lms program.' };
    args.push('--bin', text(f.bin));
  }
  return { args };
}

const LOOPBACK = /^(localhost|127(\.\d{1,3}){3}|::1|\[::1\])$/;
export const isLoopback = (h) => LOOPBACK.test(text(h));

/** The flags of `ferrum-gpu-server` (the Serve tab). */
export function serveArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to serve (a .gguf file).' };
  const args = ['--model', text(f.model)];
  const host = blank(f.host) ? '127.0.0.1' : text(f.host);
  if (!/^[A-Za-z0-9.:\-\[\]]+$/.test(host)) return { error: 'The address may use letters, digits and . : - [ ] only.' };
  if (!isLoopback(host) && blank(f.token)) {
    return { error: 'Listening beyond this computer needs a token: set one, or keep the address at 127.0.0.1.' };
  }
  if (!blank(f.host)) args.push('--host', host);
  const bad = numeric(args, [[f.port, '--port', (v) => whole(v, 0, 65535), 'The port must be a whole number from 0 (any free port) to 65535.']]);
  if (bad) return bad;
  if (!blank(f.token)) {
    if (/\s/.test(f.token)) return { error: 'The token must not contain spaces.' };
    args.push('--token', text(f.token));
  }
  return deviceFlags(f, args, { devices: ['cpu', 'gpu'], deviceMessage: 'Device must be automatic, GPU or CPU.' }) ?? { args };
}

/** One line about a tool: where it is, or why it is not. */
export function toolText(t) {
  return t.path ? `${t.name}: found at ${t.path}` : `${t.name}: not found (${t.note ?? 'unknown reason'})`;
}
