// Argument builders for the Fine-tune, Evaluate and Generate tabs, and the text of a generation.
// Pure: `{ args }` or `{ error }`, tested under node. Flag spellings follow ferrum_gpu_core::commands.
import { trainArgs, QUANTS, DEVICES, GPU_WEIGHTS } from './train_args.js';
import { blank, text, whole, real, numeric, deviceFlags } from './fields.js';
import { stopStrings } from '../stop.js';

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
  const bad = numeric(args, [[f.ctx, '--ctx', (v) => whole(v, 2), 'The window size must be a whole number of at least 2 tokens.']]);
  if (bad) return bad;
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
  const bad = numeric(args, [
    [f.maxNew, '--max-new', (v) => whole(v, 1, 4096), 'New tokens must be a whole number from 1 to 4096.'],
    [f.temperature, '--temperature', (v) => real(v, (x) => x >= 0 && x <= 10), 'Temperature must be between 0 and 10.'],
    [f.topK, '--top-k', (v) => whole(v, 0, 256), 'Top-k must be a whole number from 0 to 256.'],
    [f.topP, '--top-p', (v) => real(v, (x) => x > 0 && x <= 1), 'Top-p must be above 0 and at most 1.'],
    [f.repetition, '--repetition-penalty', (v) => real(v, (x) => x >= 1 && x <= 10), 'The repetition penalty must be between 1 (off) and 10.'],
    [f.seed, '--seed', (v) => whole(v, 0), 'Seed must be a whole number, 0 or more.'],
  ]);
  if (bad) return bad;
  for (const s of stopStrings(f.stop ?? '')) args.push('--stop', s);
  const dev = deviceFlags(f, args, { devices: DEVICES, deviceMessage: `Device must be one of ${DEVICES.join(', ')}.`, weights: GPU_WEIGHTS });
  return dev ?? { args };
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
