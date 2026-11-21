// The sections of the `train` command line, each appending its flags and returning null or the first
// problem as `{ error }`. Flag spellings and ranges follow ferrum_gpu_core::commands.
import { blank, text, whole, real, numeric, deviceFlags } from './fields.js';
import { LENGTH_KINDS, PRESETS, QUANTS, DEVICES } from './options.js';

/** Size, run length, batch, context, learning rate and seed. */
function basics(f, args) {
  if (!blank(f.preset)) {
    if (!PRESETS.includes(f.preset)) return { error: `Size must be one of ${PRESETS.join(', ')}.` };
    args.push('--preset', f.preset);
  }
  // how long: exactly one of epochs, steps or tokens
  if (!blank(f.length)) {
    const kind = f.lengthKind || 'epochs';
    if (!LENGTH_KINDS.includes(kind)) return { error: `The run length is epochs, steps or tokens, not ${kind}.` };
    const n = whole(f.length, 1, kind === 'steps' ? 100000 : Number.MAX_SAFE_INTEGER);
    if (n === null) return { error: `${kind[0].toUpperCase()}${kind.slice(1)} must be a whole number from 1${kind === 'steps' ? ' to 100000' : ''}.` };
    args.push(`--${kind}`, String(n));
  }
  return numeric(args, [
    [f.batch, '--batch', (v) => whole(v, 1, 256), 'Batch must be a whole number from 1 to 256.'],
    [f.context, '--context', (v) => whole(v, 1), 'Context must be a whole number of at least 1.'],
    [f.lr, '--lr', (v) => real(v, (x) => x > 0 && x <= 1), 'Learning rate must be above 0 and at most 1.'],
    [f.seed, '--seed', (v) => whole(v, 0), 'Seed must be a whole number, 0 or more.'],
  ]);
}

const SHAPE = [['embed', 'Width'], ['heads', 'Heads'], ['blocks', 'Blocks'], ['hidden', 'Hidden size'], ['stride', 'Window stride']];

/** The shape of the model over the preset (each part optional), its vocabulary and tokenizer. */
function shape(f, args) {
  const got = {};
  for (const [key, label] of SHAPE) {
    if (blank(f[key])) continue;
    const n = whole(f[key], key === 'stride' ? 0 : 1);
    if (n === null) return { error: `${label} must be a whole number${key === 'stride' ? ', 0 or more (0 = windows do not overlap)' : ' of at least 1'}.` };
    got[key] = n;
    args.push(`--${key}`, String(n));
  }
  if (got.embed && got.heads && got.embed % got.heads !== 0) {
    return { error: `Width ${got.embed} must divide evenly by the ${got.heads} heads.` };
  }
  const vocab = (v) => { const n = whole(v, 0); return n === null || (n !== 0 && n < 256) ? null : n; };
  const bad = numeric(args, [[f.vocab, '--vocab', vocab, 'Vocabulary must be 0 (one token per byte) or at least 256.']]);
  if (!bad && !blank(f.tokenizer)) args.push('--tokenizer', text(f.tokenizer));
  return bad;
}

/** Weight decay, gradient clip, the cosine schedule and warm-up, tied head, dropout, QAT, precision. */
function optimiser(f, args) {
  const bad = numeric(args, [
    [f.weightDecay, '--weight_decay', (v) => real(v, (x) => x >= 0 && x <= 1), 'Weight decay must be between 0 and 1.'],
    [f.clip, '--clip', (v) => real(v, (x) => x >= 0), 'Gradient clip must be 0 (off) or more.'],
  ]);
  if (bad) return bad;
  if (f.cosine) args.push('--cosine');
  if (!blank(f.warmup) && !f.cosine) return { error: 'Warm-up steps need the cosine schedule: tick it, or clear the warm-up.' };
  const warm = numeric(args, [[f.warmup, '--warmup', (v) => whole(v, 0), 'Warm-up must be a whole number of steps, 0 or more.']]);
  if (warm) return warm;
  if (f.tie) args.push('--tie');
  const drop = numeric(args, [[f.dropout, '--dropout', (v) => real(v, (x) => x >= 0 && x < 1), 'Dropout must be at least 0 and below 1.']]);
  if (drop) return drop;
  if (f.qat === false) args.push('--no-qat');
  if (!blank(f.quant)) {
    if (!QUANTS.includes(f.quant)) return { error: `Saved precision must be one of ${QUANTS.join(', ')}.` };
    args.push('--quant', f.quant);
  }
  return null;
}

/** Validation share, early stopping, the time limit, checkpoints, resume, sampling and metrics. */
function stopping(f, args) {
  const bad = numeric(args, [[f.val, '--val', (v) => real(v, (x) => x > 0 && x < 1), 'The validation share must be above 0 and below 1 (for example 0.1).']]);
  if (bad) return bad;
  if (!blank(f.patience) && blank(f.val)) return { error: 'Early stopping needs a validation share: set one, or clear the patience.' };
  const rest = numeric(args, [
    [f.patience, '--patience', (v) => whole(v, 1), 'Patience must be a whole number of epochs, 1 or more.'],
    [f.maxSeconds, '--max-seconds', (v) => real(v, (x) => x > 0), 'The time limit must be a positive number of seconds.'],
  ]);
  if (rest) return rest;
  if (blank(f.checkpoint) && !blank(f.checkpointEvery)) return { error: 'Checkpoint every needs a checkpoint file.' };
  if (!blank(f.checkpoint)) args.push('--checkpoint', text(f.checkpoint));
  const every = numeric(args, [[f.checkpointEvery, '--checkpoint-every', (v) => whole(v, 1), 'Checkpoint every must be a whole number of epochs, 1 or more.']]);
  if (every) return every;
  if (!blank(f.resume)) args.push('--resume', text(f.resume));
  if (f.sample) args.push('--sample');
  if (!blank(f.metricsCsv)) args.push('--metrics-csv', text(f.metricsCsv));
  return null;
}

const device = (f, args) => deviceFlags(f, args, { devices: DEVICES, deviceMessage: `Device must be one of ${DEVICES.join(', ')}.` });

export const TRAIN_SECTIONS = [basics, shape, optimiser, stopping, device];
