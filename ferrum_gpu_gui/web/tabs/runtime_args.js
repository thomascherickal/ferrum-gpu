// Argument builders for the Export and Serve tabs, with the same refusals the server applies, so a
// mistake is caught before a job is started. Pure: `{ args }` or `{ error }`, tested under node.

const blank = (v) => String(v ?? '').trim() === '';
const text = (v) => String(v ?? '').trim();

function whole(v, min, max = Number.MAX_SAFE_INTEGER) {
  const s = text(v);
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= min && n <= max ? n : null;
}

/** `export-gguf`: re-write a model in another precision. */
export function exportGgufArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to export (a .gguf file).' };
  if (blank(f.out)) return { error: 'Choose where to write the new file (a path ending in .gguf).' };
  if (text(f.out) === text(f.model)) return { error: 'The new file must be a different file from the model.' };
  const args = ['--model', text(f.model), '--out', text(f.out)];
  if (!blank(f.quant)) {
    if (!['f32', 'int8', 'int4'].includes(f.quant)) return { error: 'Precision must be f32, int8 or int4.' };
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
  if (!blank(f.port)) {
    const n = whole(f.port, 0, 65535);
    if (n === null) return { error: 'The port must be a whole number from 0 (any free port) to 65535.' };
    args.push('--port', String(n));
  }
  if (!blank(f.token)) {
    if (/\s/.test(f.token)) return { error: 'The token must not contain spaces.' };
    args.push('--token', text(f.token));
  }
  if (!blank(f.device) && f.device !== 'auto') {
    if (!['cpu', 'gpu'].includes(f.device)) return { error: 'Device must be automatic, GPU or CPU.' };
    args.push('--device', f.device);
  }
  if (!blank(f.adapter)) args.push('--gpu-adapter', text(f.adapter));
  if (!blank(f.gpuMem)) {
    const n = whole(f.gpuMem, 1);
    if (n === null) return { error: 'The GPU memory budget must be a whole number of MiB, 1 or more.' };
    args.push('--gpu-mem-mb', String(n));
  }
  if (!blank(f.threads)) {
    const n = whole(f.threads, 1);
    if (n === null) return { error: 'Threads must be a whole number, 1 or more.' };
    args.push('--threads', String(n));
  }
  return { args };
}

/** One line about a tool: where it is, or why it is not. */
export function toolText(t) {
  return t.path ? `${t.name}: found at ${t.path}` : `${t.name}: not found (${t.note ?? 'unknown reason'})`;
}
