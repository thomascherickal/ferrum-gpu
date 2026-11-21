// Node tests of the `train` argument builder: each option on its own, and the model shape.
// usage: node train_args.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { trainArgs } from '../web/tabs/train_args.js';

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
for (const q of ['f16', 'fp4', 'q2']) eq(one({ quant: q }), ['--quant', q]);
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


// ---- model shape and metrics file
const shapeBase = { data: 'a.txt', out: 'm.gguf' };
const with_ = (f) => trainArgs({ ...shapeBase, ...f });
eq(with_({ embed: '64', heads: '4', blocks: '3', hidden: '256', stride: '8' }).args.slice(5),
  ['--embed', '64', '--heads', '4', '--blocks', '3', '--hidden', '256', '--stride', '8']);
eq(with_({ stride: '0' }).args.slice(5), ['--stride', '0']);
ok(with_({ embed: '30', heads: '4' }).error.includes('divide evenly'));
ok(with_({ embed: '0' }).error.includes('Width'));
ok(with_({ heads: 'x' }).error.includes('Heads'));
ok(with_({ blocks: '1.5' }).error.includes('Blocks'));
ok(with_({ hidden: '-1' }).error.includes('Hidden'));
ok(with_({ stride: 'a' }).error.includes('stride'));
eq(with_({ metricsCsv: ' m.csv ' }).args.slice(-2), ['--metrics-csv', 'm.csv']);
ok(!with_({ metricsCsv: '' }).args.includes('--metrics-csv'));

summary();
