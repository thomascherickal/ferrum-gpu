// Node tests of the Train tab's pure logic: the argument builder and its refusals, the loss chart's
// layout and drawing (against a fake canvas that records its calls), and the series helpers.
// usage: node train.test.mjs
import assert from 'node:assert/strict';
import { trainArgs, LENGTH_KINDS, PRESETS, QUANTS, DEVICES } from '../web/tabs/train_args.js';
import { plan, draw, lossLabel } from '../web/charts/loss.js';
import { collect, runSummary } from '../web/tabs/gtrain.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

// ---- trainArgs: the minimum, then each option on its own
const base = { data: 'corpus.txt', out: 'm.gguf' };
eq(trainArgs(base), { args: ['--data', 'corpus.txt', '--out', 'm.gguf', '--verbose'] });
eq(trainArgs({ data: ' corpus.txt ', out: ' m.gguf ' }).args.slice(0, 4), ['--data', 'corpus.txt', '--out', 'm.gguf']);
const one = (extra) => trainArgs({ ...base, ...extra }).args.slice(5);
eq(one({ preset: 'small' }), ['--preset', 'small']);
eq(one({ length: '30', lengthKind: 'epochs' }), ['--epochs', '30']);
eq(one({ length: '30' }), ['--epochs', '30']);                       // epochs is the default unit
eq(one({ length: '500', lengthKind: 'steps' }), ['--steps', '500']);
eq(one({ length: '1000000', lengthKind: 'tokens' }), ['--tokens', '1000000']);
eq(one({ batch: '16' }), ['--batch', '16']);
eq(one({ context: '64' }), ['--context', '64']);
eq(one({ lr: '0.003' }), ['--lr', '0.003']);
eq(one({ seed: '7' }), ['--seed', '7']);
eq(one({ vocab: '300' }), ['--vocab', '300']);
eq(one({ vocab: '0' }), ['--vocab', '0']);
eq(one({ tokenizer: 'tok.gguf' }), ['--tokenizer', 'tok.gguf']);
eq(one({ weightDecay: '0.01' }), ['--weight_decay', '0.01']);
eq(one({ clip: '1' }), ['--clip', '1']);
eq(one({ clip: '0' }), ['--clip', '0']);
eq(one({ cosine: true }), ['--cosine']);
eq(one({ cosine: true, warmup: '20' }), ['--cosine', '--warmup', '20']);
eq(one({ tie: true }), ['--tie']);
eq(one({ dropout: '0.1' }), ['--dropout', '0.1']);
eq(one({ qat: true }), []);                                          // on is the CLI default
eq(one({ qat: false }), ['--no-qat']);
eq(one({ quant: 'int4' }), ['--quant', 'int4']);
eq(one({ val: '0.1' }), ['--val', '0.1']);
eq(one({ val: '0.1', patience: '3' }), ['--val', '0.1', '--patience', '3']);
eq(one({ maxSeconds: '60' }), ['--max-seconds', '60']);
eq(one({ checkpoint: 'c.ckpt' }), ['--checkpoint', 'c.ckpt']);
eq(one({ checkpoint: 'c.ckpt', checkpointEvery: '2' }), ['--checkpoint', 'c.ckpt', '--checkpoint-every', '2']);
eq(one({ resume: 'c.ckpt' }), ['--resume', 'c.ckpt']);
eq(one({ sample: true }), ['--sample']);
eq(one({ device: 'auto' }), []);
eq(one({ device: 'gpu' }), ['--device', 'gpu']);
eq(one({ device: 'cpu', threads: '4' }), ['--device', 'cpu', '--threads', '4']);
eq(one({ adapter: 'MX330' }), ['--gpu-adapter', 'MX330']);
eq(one({ gpuMem: '1500' }), ['--gpu-mem-mb', '1500']);
// blanks are left to the command's defaults
eq(trainArgs({ ...base, preset: '', length: '', batch: ' ', lr: '', vocab: '', val: '', device: '' }).args, ['--data', 'corpus.txt', '--out', 'm.gguf', '--verbose']);

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

// ---- the chart layout
const m = { left: 10, right: 10, top: 10, bottom: 10 };
ok(plan([], [], 100, 100, m).empty);
eq(plan([], [], 100, 100, m).box, { x: 10, y: 10, w: 80, h: 80 });
const p = plan([[0, 4], [10, 2]], [[10, 3]], 100, 100, m);
eq(p.range, { x0: 0, x1: 10, y0: 2, y1: 4 });
eq(p.train, [[10, 10], [90, 90]]);                 // the high loss at the top left, the low one at the bottom right
eq(p.val, [[90, 50]]);
const flat = plan([[5, 3], [6, 3]], [], 100, 100, m);
ok(flat.train.every(([, y]) => Number.isFinite(y)), 'a flat series has no division by zero');
eq(plan([[1, NaN], [Infinity, 2], [2, 2]], [], 100, 100, m).train.length, 1);   // non-finite points are dropped
ok(plan([[0, 1]], [], 5, 5, m).box.w >= 1, 'a tiny area still has a positive box');
eq(lossLabel(1.23456), '1.235'); eq(lossLabel(NaN), '–'); eq(lossLabel(Infinity), '–');

// ---- drawing on a fake context
function fakeCtx() {
  const log = [];
  const rec = (name) => (...a) => { log.push([name, ...a]); };
  return { log, ...Object.fromEntries(['clearRect', 'strokeRect', 'fillText', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'fillRect'].map((n) => [n, rec(n)])) };
}
const css = { line: '#ccc', muted: '#888', accent: '#06c', good: '#0a0' };
{
  const g = fakeCtx();
  const r = draw(g, 200, 100, [], [], css);
  ok(r.empty);
  ok(g.log.some(([n, t]) => n === 'fillText' && t === 'loss will appear here'));
  ok(!g.log.some(([n]) => n === 'lineTo'));
}
{
  const g = fakeCtx();
  draw(g, 200, 100, [[1, 3], [2, 2], [3, 1]], [[3, 1.2]], css);
  eq(g.log.filter(([n]) => n === 'lineTo').length, 2);
  eq(g.log.filter(([n]) => n === 'moveTo').length, 1);
  eq(g.log.filter(([n]) => n === 'fillRect').length, 1);          // the one validation dot
  ok(g.log.some(([n, t]) => n === 'fillText' && t === '3.000'));
  ok(g.log.some(([n, t]) => n === 'fillText' && t === 'step 3'));
}
{
  const g = fakeCtx();
  draw(g, 200, 100, [[1, 3]], [], css);
  eq(g.log.filter(([n]) => n === 'fillRect').length, 1);          // a single point is a dot, not a line
  ok(!g.log.some(([n]) => n === 'lineTo'));
}

// ---- series helpers
{
  const s = { train: [], val: [], lastStep: 0 };
  collect([
    { ev: { kind: 'step', step: 10, loss: 3 } },
    { t: 'plain line' },
    { ev: { kind: 'val', epoch: 1, loss: 2.5 } },
    { ev: { kind: 'step', step: 20, loss: 2 } },
  ], s);
  eq(s.train, [[10, 3], [20, 2]]);
  eq(s.val, [[10, 2.5]]);                                          // a validation is placed at the latest step seen
  eq(s.lastStep, 20);
  eq(collect([], s).train.length, 2);
}
eq(runSummary(null), '');
eq(runSummary({ step: 20, total: 100, tokens_per_s: 1234.6, eta: 65 }), 'step 20/100 · 1235 tokens/s · 1m 05s left');

console.log(`OK ${checks} checks`);
