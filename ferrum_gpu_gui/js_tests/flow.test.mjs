// Node tests of the Fine-tune, Evaluate and Generate tabs' pure logic: argument builders and their
// refusals, the text of a generation, and the sentence for an eval result.
// usage: node flow.test.mjs
import assert from 'node:assert/strict';
import { finetuneArgs, evalArgs, generateArgs, outputText } from '../web/tabs/flow_args.js';
import { resultText } from '../web/tabs/evaluate.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

// ---- finetune-gguf
const ft = { model: 'm.gguf', data: 'new.txt', out: 'tuned.gguf' };
eq(finetuneArgs(ft), { args: ['--model', 'm.gguf', '--data', 'new.txt', '--out', 'tuned.gguf', '--verbose'] });
eq(finetuneArgs({ ...ft, length: '5', lengthKind: 'epochs', batch: '8', lr: '0.001' }).args.slice(7),
  ['--epochs', '5', '--batch', '8', '--lr', '0.001']);
// the options that fix a new model's shape and tokenizer are not passed on, even if the form has them
const stripped = finetuneArgs({ ...ft, preset: 'small', vocab: '400', tokenizer: 't.gguf', tie: true }).args;
for (const flag of ['--preset', '--vocab', '--tokenizer', '--tie']) ok(!stripped.includes(flag), flag);
eq(finetuneArgs({ ...ft, val: '0.1', patience: '2', qat: false, quant: 'int4', device: 'cpu' }).args.slice(7),
  ['--no-qat', '--quant', 'int4', '--val', '0.1', '--patience', '2', '--device', 'cpu']);
ok(/model to fine-tune/.test(finetuneArgs({ data: 'a', out: 'b' }).error));
ok(/model to fine-tune/.test(finetuneArgs({ model: ' ', data: 'a', out: 'b' }).error));
ok(/training text/.test(finetuneArgs({ model: 'm', out: 'b' }).error));          // the shared builder's refusals pass through
ok(/save the model/.test(finetuneArgs({ model: 'm', data: 'a' }).error));
ok(/Learning rate/.test(finetuneArgs({ ...ft, lr: '5' }).error));

// ---- eval
eq(evalArgs({ model: 'm.gguf', data: 'd.txt' }), { args: ['--model', 'm.gguf', '--data', 'd.txt'] });
eq(evalArgs({ model: 'm', data: 'd', ctx: '64', quant: 'int8', json: 'r.json' }).args, ['--model', 'm', '--data', 'd', '--ctx', '64', '--quant', 'int8', '--json', 'r.json']);
ok(/model to evaluate/.test(evalArgs({ data: 'd' }).error));
ok(/text to measure/.test(evalArgs({ model: 'm' }).error));
ok(/at least 2/.test(evalArgs({ model: 'm', data: 'd', ctx: '1' }).error));
ok(/at least 2/.test(evalArgs({ model: 'm', data: 'd', ctx: 'x' }).error));
ok(/Precision/.test(evalArgs({ model: 'm', data: 'd', quant: 'int2' }).error));
eq(evalArgs({ model: 'm', data: 'd', ctx: '', quant: '', json: ' ' }).args, ['--model', 'm', '--data', 'd']);

// ---- generate
const g = { model: 'm.gguf', prompt: 'the cat' };
eq(generateArgs(g), { args: ['--model', 'm.gguf', '--prompt', 'the cat'] });
eq(generateArgs({ ...g, prompt: '  keep my spaces ' }).args[3], '  keep my spaces ');   // the prompt is not trimmed
eq(generateArgs({ ...g, maxNew: '32', temperature: '0.8', topK: '40', seed: '7' }).args.slice(4),
  ['--max-new', '32', '--temperature', '0.8', '--top-k', '40', '--seed', '7']);
eq(generateArgs({ ...g, topP: '0.9', repetition: '1.2' }).args.slice(4), ['--top-p', '0.9', '--repetition-penalty', '1.2']);
eq(generateArgs({ ...g, stop: '.\n\nEND\\n\n' }).args.slice(4), ['--stop', '.', '--stop', 'END\n']);   // one per line, escapes applied
eq(generateArgs({ ...g, device: 'auto' }).args.length, 4);
eq(generateArgs({ ...g, device: 'gpu', adapter: 'MX330', gpuMem: '1500' }).args.slice(4), ['--device', 'gpu', '--gpu-adapter', 'MX330', '--gpu-mem-mb', '1500']);
ok(/model to generate/.test(generateArgs({ prompt: 'x' }).error));
ok(/prompt/.test(generateArgs({ model: 'm' }).error));
ok(/prompt/.test(generateArgs({ model: 'm', prompt: '   ' }).error));
for (const [extra, needle] of [
  [{ maxNew: '0' }, 'New tokens'], [{ maxNew: '5000' }, 'New tokens'], [{ temperature: '11' }, 'Temperature'],
  [{ temperature: '-1' }, 'Temperature'], [{ topK: '300' }, 'Top-k'], [{ topK: '1.5' }, 'Top-k'],
  [{ topP: '0' }, 'Top-p'], [{ topP: '1.5' }, 'Top-p'], [{ repetition: '0.5' }, 'repetition'], [{ repetition: '11' }, 'repetition'],
  [{ seed: '-2' }, 'Seed'], [{ device: 'tpu' }, 'Device'], [{ gpuMem: '0' }, 'memory budget'],
]) {
  const r = generateArgs({ ...g, ...extra });
  ok(r.error && r.error.includes(needle), `${JSON.stringify(extra)} -> ${r.error}`);
}

// ---- the generated text
const L = (t, e = false) => ({ t, e });
eq(outputText([L('device: cpu (cpu)'), L('the cat sat'), L('on the mat')]), 'the cat sat\non the mat');
eq(outputText([L('device: gpu'), L('note: no usable GPU'), L('preflight: GPU buffers 1 MiB'), L('model: vocab 300'), L('tokenizer: 300 tokens'), L('hello')]), 'hello');
eq(outputText([L('boom', true), L('text')]), 'text');                  // error lines are not part of the text
eq(outputText([]), '');
eq(outputText([L(''), L('first line after a blank')]), 'first line after a blank');
eq(outputText([L('a'), L(''), L('b')]), 'a\n\nb');                     // blank lines inside the text are kept

// ---- generate: chat template switch, token ids, GPU weights, threads
const g0 = { model: 'm.gguf', prompt: 'hi' };
eq(generateArgs({ ...g0, raw: true }), { args: ['--model', 'm.gguf', '--prompt', 'hi', '--raw'] });
eq(generateArgs({ ...g0, raw: false }), { args: ['--model', 'm.gguf', '--prompt', 'hi'] });
eq(generateArgs({ model: 'm.gguf', prompt: '', ids: ' 1  15043 ' }), { args: ['--model', 'm.gguf', '--ids', '1 15043'] });
eq(generateArgs({ ...g0, ids: '1 2' }), { args: ['--model', 'm.gguf', '--prompt', 'hi', '--ids', '1 2'] });
ok(generateArgs({ ...g0, ids: '1 x' }).error.includes('whole numbers'));
ok(generateArgs({ model: 'm.gguf', prompt: ' ' }).error.includes('or give token ids'));
eq(generateArgs({ ...g0, gpuWeights: 'q8_0', threads: '4' }), { args: ['--model', 'm.gguf', '--prompt', 'hi', '--gpu-weights', 'q8_0', '--threads', '4'] });
eq(generateArgs({ ...g0, gpuWeights: 'f32' }), { args: ['--model', 'm.gguf', '--prompt', 'hi'] });
ok(generateArgs({ ...g0, gpuWeights: 'q5' }).error.includes('GPU weights must be one of'));
ok(generateArgs({ ...g0, threads: '0' }).error.includes('Threads'));

// ---- the new precisions are accepted everywhere a precision is chosen
for (const q of ['f16', 'fp4', 'q2']) {
  eq(evalArgs({ model: 'm', data: 'd', quant: q }).args.slice(-2), ['--quant', q]);
  eq(finetuneArgs({ ...ft, quant: q }).args.slice(-2), ['--quant', q]);
}
for (const w of ['fp4', 'fp8', 'fp8e5m2', 'q2_k']) {
  eq(generateArgs({ model: 'm.gguf', prompt: 'hi', gpuWeights: w }).args.slice(-2), ['--gpu-weights', w]);
}

// ---- the eval sentence
eq(resultText(null), '');
eq(resultText({ tokens: 4096, loss: 2.34567, perplexity: 10.4421 }), 'perplexity 10.442 · loss 2.3457 · over 4096 tokens');

eq(resultText({ tokens: 10, loss: 1.5, perplexity: 4.4817, bits_per_byte: 0.8125 }), 'perplexity 4.482 · loss 1.5000 · over 10 tokens · 0.8125 bits per byte');
eq(resultText({ tokens: 10, loss: 1.5, perplexity: 4.4817, bits_per_byte: null }), 'perplexity 4.482 · loss 1.5000 · over 10 tokens');

console.log(`OK ${checks} checks`);
