// Building the `train` command's arguments from the Train tab's form, and refusing bad values before a
// job is started. Pure: returns `{ args }` or `{ error }` (the first problem found), tested under node.
// Flag spellings and ranges follow ferrum_gpu_core::commands (docs/USER-GUIDE.md section 3).

const blank = (v) => String(v ?? '').trim() === '';
const text = (v) => String(v ?? '').trim();

/** Parse a whole number in [min, max]; null on failure. */
function whole(v, min, max = Number.MAX_SAFE_INTEGER) {
  const s = text(v);
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= min && n <= max ? n : null;
}

/** Parse a finite number with `lo < n <= hi` (open below) or `lo <= n < hi` (open above) semantics via `ok`. */
function real(v, ok) {
  const s = text(v);
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) && ok(n) ? n : null;
}

export const LENGTH_KINDS = ['epochs', 'steps', 'tokens'];
export const PRESETS = ['tiny', 'small', 'medium'];
export const QUANTS = ['int8', 'int4', 'f32'];
export const DEVICES = ['auto', 'cpu', 'gpu'];

/** The arguments of `train` (GPT-2, the default architecture). */
export function trainArgs(f) {
  if (blank(f.data)) return { error: 'Choose the training text (a file, a folder, or a token file).' };
  if (blank(f.out)) return { error: 'Choose where to save the model (a path ending in .gguf).' };
  const args = ['--data', text(f.data), '--out', text(f.out), '--verbose'];
  const need = (flag, value) => args.push(flag, String(value));

  if (!blank(f.preset)) {
    if (!PRESETS.includes(f.preset)) return { error: `Size must be one of ${PRESETS.join(', ')}.` };
    need('--preset', f.preset);
  }
  // how long: exactly one of epochs, steps or tokens
  if (!blank(f.length)) {
    const kind = f.lengthKind || 'epochs';
    if (!LENGTH_KINDS.includes(kind)) return { error: `The run length is epochs, steps or tokens, not ${kind}.` };
    const max = kind === 'steps' ? 100000 : Number.MAX_SAFE_INTEGER;
    const n = whole(f.length, 1, max);
    if (n === null) return { error: `${kind[0].toUpperCase()}${kind.slice(1)} must be a whole number from 1${kind === 'steps' ? ' to 100000' : ''}.` };
    need(`--${kind}`, n);
  }
  if (!blank(f.batch)) {
    const n = whole(f.batch, 1, 256);
    if (n === null) return { error: 'Batch must be a whole number from 1 to 256.' };
    need('--batch', n);
  }
  if (!blank(f.context)) {
    const n = whole(f.context, 1);
    if (n === null) return { error: 'Context must be a whole number of at least 1.' };
    need('--context', n);
  }
  if (!blank(f.lr)) {
    const n = real(f.lr, (x) => x > 0 && x <= 1);
    if (n === null) return { error: 'Learning rate must be above 0 and at most 1.' };
    need('--lr', n);
  }
  if (!blank(f.seed)) {
    const n = whole(f.seed, 0);
    if (n === null) return { error: 'Seed must be a whole number, 0 or more.' };
    need('--seed', n);
  }
  if (!blank(f.vocab)) {
    const n = whole(f.vocab, 0);
    if (n === null || (n !== 0 && n < 256)) return { error: 'Vocabulary must be 0 (one token per byte) or at least 256.' };
    need('--vocab', n);
  }
  if (!blank(f.tokenizer)) need('--tokenizer', text(f.tokenizer));

  // optimiser
  if (!blank(f.weightDecay)) {
    const n = real(f.weightDecay, (x) => x >= 0 && x <= 1);
    if (n === null) return { error: 'Weight decay must be between 0 and 1.' };
    need('--weight_decay', n);
  }
  if (!blank(f.clip)) {
    const n = real(f.clip, (x) => x >= 0);
    if (n === null) return { error: 'Gradient clip must be 0 (off) or more.' };
    need('--clip', n);
  }
  if (f.cosine) args.push('--cosine');
  if (!blank(f.warmup)) {
    if (!f.cosine) return { error: 'Warm-up steps need the cosine schedule: tick it, or clear the warm-up.' };
    const n = whole(f.warmup, 0);
    if (n === null) return { error: 'Warm-up must be a whole number of steps, 0 or more.' };
    need('--warmup', n);
  }
  if (f.tie) args.push('--tie');
  if (!blank(f.dropout)) {
    const n = real(f.dropout, (x) => x >= 0 && x < 1);
    if (n === null) return { error: 'Dropout must be at least 0 and below 1.' };
    need('--dropout', n);
  }
  if (f.qat === false) args.push('--no-qat');
  if (!blank(f.quant)) {
    if (!QUANTS.includes(f.quant)) return { error: `Saved precision must be one of ${QUANTS.join(', ')}.` };
    need('--quant', f.quant);
  }

  // validation and stopping
  let valOn = false;
  if (!blank(f.val)) {
    const n = real(f.val, (x) => x > 0 && x < 1);
    if (n === null) return { error: 'The validation share must be above 0 and below 1 (for example 0.1).' };
    need('--val', n);
    valOn = true;
  }
  if (!blank(f.patience)) {
    if (!valOn) return { error: 'Early stopping needs a validation share: set one, or clear the patience.' };
    const n = whole(f.patience, 1);
    if (n === null) return { error: 'Patience must be a whole number of epochs, 1 or more.' };
    need('--patience', n);
  }
  if (!blank(f.maxSeconds)) {
    const n = real(f.maxSeconds, (x) => x > 0);
    if (n === null) return { error: 'The time limit must be a positive number of seconds.' };
    need('--max-seconds', n);
  }
  if (!blank(f.checkpoint)) {
    need('--checkpoint', text(f.checkpoint));
    if (!blank(f.checkpointEvery)) {
      const n = whole(f.checkpointEvery, 1);
      if (n === null) return { error: 'Checkpoint every must be a whole number of epochs, 1 or more.' };
      need('--checkpoint-every', n);
    }
  } else if (!blank(f.checkpointEvery)) {
    return { error: 'Checkpoint every needs a checkpoint file.' };
  }
  if (!blank(f.resume)) need('--resume', text(f.resume));
  if (f.sample) args.push('--sample');

  // device
  if (!blank(f.device) && f.device !== 'auto') {
    if (!DEVICES.includes(f.device)) return { error: `Device must be one of ${DEVICES.join(', ')}.` };
    need('--device', f.device);
  }
  if (!blank(f.adapter)) need('--gpu-adapter', text(f.adapter));
  if (!blank(f.gpuMem)) {
    const n = whole(f.gpuMem, 1);
    if (n === null) return { error: 'The GPU memory budget must be a whole number of MiB, 1 or more.' };
    need('--gpu-mem-mb', n);
  }
  if (!blank(f.threads)) {
    const n = whole(f.threads, 1);
    if (n === null) return { error: 'Threads must be a whole number, 1 or more.' };
    need('--threads', n);
  }
  return { args };
}
