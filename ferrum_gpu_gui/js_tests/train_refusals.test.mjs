// Node tests of the `train` builder's refusals, the option lists and a fully loaded request.
// usage: node train_refusals.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { trainArgs, LENGTH_KINDS, PRESETS, QUANTS, DEVICES } from '../web/tabs/train_args.js';

const base = { data: 'corpus.txt', out: 'm.gguf' };

// ---- refusals, each naming the problem
const bad = (extra, needle) => {
  const r = trainArgs({ ...base, ...extra });
  ok(r.error && r.error.includes(needle), `${JSON.stringify(extra)} -> ${r.error}`);
  eq(r.args, undefined);
};
ok(/training text/.test(trainArgs({ out: 'm.gguf' }).error));
ok(/training text/.test(trainArgs({ data: '  ', out: 'm.gguf' }).error));
ok(/save the model/.test(trainArgs({ data: 'c' }).error));
bad({ preset: 'huge' }, 'Size must be');
bad({ length: '0' }, 'Epochs');
bad({ length: '-3', lengthKind: 'steps' }, 'Steps');
bad({ length: '100001', lengthKind: 'steps' }, 'Steps');
bad({ length: '1.5', lengthKind: 'tokens' }, 'Tokens');
bad({ length: '5', lengthKind: 'minutes' }, 'epochs, steps or tokens');
bad({ batch: '0' }, 'Batch');
bad({ batch: '257' }, 'Batch');
bad({ context: '0' }, 'Context');
bad({ lr: '0' }, 'Learning rate');
bad({ lr: '1.5' }, 'Learning rate');
bad({ lr: 'abc' }, 'Learning rate');
bad({ seed: '-1' }, 'Seed');
bad({ vocab: '100' }, 'Vocabulary');
bad({ vocab: 'x' }, 'Vocabulary');
bad({ weightDecay: '2' }, 'Weight decay');
bad({ clip: '-1' }, 'clip');
bad({ warmup: '10' }, 'cosine');
bad({ cosine: true, warmup: '-1' }, 'Warm-up');
bad({ dropout: '1' }, 'Dropout');
bad({ dropout: '-0.1' }, 'Dropout');
bad({ quant: 'int2' }, 'precision');
bad({ val: '0' }, 'validation share');
bad({ val: '1' }, 'validation share');
bad({ patience: '3' }, 'validation share');
bad({ val: '0.1', patience: '0' }, 'Patience');
bad({ maxSeconds: '0' }, 'time limit');
bad({ checkpointEvery: '2' }, 'needs a checkpoint file');
bad({ checkpoint: 'c', checkpointEvery: '0' }, 'Checkpoint every');
bad({ device: 'tpu' }, 'Device');
bad({ gpuMem: '0' }, 'memory budget');
bad({ threads: '0' }, 'Threads');
// the option lists the page offers are the ones the builder accepts
for (const k of LENGTH_KINDS) ok(trainArgs({ ...base, length: '5', lengthKind: k }).args);
for (const k of PRESETS) ok(trainArgs({ ...base, preset: k }).args);
for (const k of QUANTS) ok(trainArgs({ ...base, quant: k }).args);
for (const k of DEVICES) ok(trainArgs({ ...base, device: k }).args);
// a fully loaded request keeps every flag, in a stable order
const all = trainArgs({ ...base, preset: 'tiny', length: '3', lengthKind: 'epochs', batch: '8', context: '32', lr: '0.01', seed: '1',
  vocab: '300', tokenizer: 't.gguf', weightDecay: '0.01', clip: '1', cosine: true, warmup: '5', tie: true, dropout: '0.1', qat: false,
  quant: 'int8', val: '0.2', patience: '2', maxSeconds: '30', checkpoint: 'c', checkpointEvery: '1', resume: 'r', sample: true,
  device: 'gpu', adapter: 'MX330', gpuMem: '1500', threads: '2' });
eq(all.args.filter((a) => a.startsWith('--')).length, 30);

summary();
