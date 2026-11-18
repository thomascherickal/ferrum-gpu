// Argument builders for the Fine-tune, Evaluate and Generate tabs, and the text of a generation.
// Pure: `{ args }` or `{ error }`, tested under node. Flag spellings follow ferrum_gpu_core::commands.
import { trainArgs, QUANTS, DEVICES, GPU_WEIGHTS } from './train_args.js';
import { stopStrings } from '../stop.js';

const blank = (v) => String(v ?? '').trim() === '';
const text = (v) => String(v ?? '').trim();

function whole(v, min, max = Number.MAX_SAFE_INTEGER) {
  const s = text(v);
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= min && n <= max ? n : null;
}

function real(v, ok) {
  const n = Number(text(v));
  return text(v) !== '' && Number.isFinite(n) && ok(n) ? n : null;
}

/**
 * `finetune-gguf`: continue training a model. It takes the training options of `train` except the ones
 * that fix a new model's shape and tokenizer (size, vocabulary, tokenizer file, tied head), which a
 * fine-tune inherits from the model it starts from.
 */
export function finetuneArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to fine-tune (a .gguf file).' };
  const r = trainArgs({ ...f, preset: '', vocab: '', tokenizer: '', tie: false, embed: '', heads: '', blocks: '', hidden: '' });
  if (r.error) return r;
  return { args: ['--model', text(f.model), ...r.args] };
}

/** `eval`: the loss and perplexity of a text under a model. It always runs on the CPU. */
export function evalArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to evaluate (a .gguf file).' };
  if (blank(f.data)) return { error: 'Choose the text to measure the model on.' };
  const args = ['--model', text(f.model), '--data', text(f.data)];
  if (!blank(f.ctx)) {
    const n = whole(f.ctx, 2);
    if (n === null) return { error: 'The window size must be a whole number of at least 2 tokens.' };
    args.push('--ctx', String(n));
  }
  if (!blank(f.quant)) {
    if (!QUANTS.includes(f.quant)) return { error: `Precision must be one of ${QUANTS.join(', ')}.` };
    args.push('--quant', f.quant);
  }
  if (!blank(f.json)) args.push('--json', text(f.json));
  return { args };
}

/** `generate`: continue a prompt. `stop` is the stop-strings box (one per line, `\n` and `\t` unescaped). */
export function generateArgs(f) {
  if (blank(f.model)) return { error: 'Choose the model to generate from (a .gguf file).' };
  // token ids replace the prompt text (and skip the chat template): "1 2 3"
  let ids = null;
  if (!blank(f.ids)) {
    ids = text(f.ids).split(/\s+/);
    if (!ids.every((x) => /^\d+$/.test(x))) return { error: 'Token ids are whole numbers separated by spaces, for example 1 15043.' };
  }
  if (blank(f.prompt) && !ids) return { error: 'Type a prompt to continue, or give token ids.' };
  const args = ['--model', text(f.model)];
  if (!blank(f.prompt)) args.push('--prompt', String(f.prompt));
  if (ids) args.push('--ids', ids.join(' '));
  if (f.raw) args.push('--raw');
  if (!blank(f.maxNew)) {
    const n = whole(f.maxNew, 1, 4096);
    if (n === null) return { error: 'New tokens must be a whole number from 1 to 4096.' };
    args.push('--max-new', String(n));
  }
  if (!blank(f.temperature)) {
    const n = real(f.temperature, (x) => x >= 0 && x <= 10);
    if (n === null) return { error: 'Temperature must be between 0 and 10.' };
    args.push('--temperature', String(n));
  }
  if (!blank(f.topK)) {
    const n = whole(f.topK, 0, 256);
    if (n === null) return { error: 'Top-k must be a whole number from 0 to 256.' };
    args.push('--top-k', String(n));
  }
  if (!blank(f.topP)) {
    const n = real(f.topP, (x) => x > 0 && x <= 1);
    if (n === null) return { error: 'Top-p must be above 0 and at most 1.' };
    args.push('--top-p', String(n));
  }
  if (!blank(f.repetition)) {
    const n = real(f.repetition, (x) => x >= 1 && x <= 10);
    if (n === null) return { error: 'The repetition penalty must be between 1 (off) and 10.' };
    args.push('--repetition-penalty', String(n));
  }
  if (!blank(f.seed)) {
    const n = whole(f.seed, 0);
    if (n === null) return { error: 'Seed must be a whole number, 0 or more.' };
    args.push('--seed', String(n));
  }
  for (const s of stopStrings(f.stop ?? '')) args.push('--stop', s);
  if (!blank(f.device) && f.device !== 'auto') {
    if (!DEVICES.includes(f.device)) return { error: `Device must be one of ${DEVICES.join(', ')}.` };
    args.push('--device', f.device);
  }
  if (!blank(f.adapter)) args.push('--gpu-adapter', text(f.adapter));
  if (!blank(f.gpuMem)) {
    const n = whole(f.gpuMem, 1);
    if (n === null) return { error: 'The GPU memory budget must be a whole number of MiB, 1 or more.' };
    args.push('--gpu-mem-mb', String(n));
  }
  if (!blank(f.gpuWeights)) {
    if (!GPU_WEIGHTS.includes(f.gpuWeights)) return { error: `GPU weights must be one of ${GPU_WEIGHTS.join(', ')}.` };
    if (f.gpuWeights !== 'f32') args.push('--gpu-weights', f.gpuWeights);
  }
  if (!blank(f.threads)) {
    const n = whole(f.threads, 1);
    if (n === null) return { error: 'Threads must be a whole number, 1 or more.' };
    args.push('--threads', String(n));
  }
  return { args };
}

// lines the commands print around the text itself
const NOISE = /^(device|note|preflight|model|tokenizer|adapter|steps):/;

/** The generated text: the output lines without the commands' own status lines, joined. */
export function outputText(lines) {
  return lines
    .filter((l) => !l.e && !NOISE.test(l.t))
    .map((l) => l.t)
    .join('\n')
    .replace(/^\n+/, '');
}
