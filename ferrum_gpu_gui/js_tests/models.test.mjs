// Node tests of the Models tab's text helpers.
// usage: node models.test.mjs
import assert from 'node:assert/strict';
import { dimsText, paramsText, timeText, summaryLine, kindsText, rowMatches } from '../web/tabs/models_view.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

eq(dimsText([4, 3]), '4 × 3');
eq(dimsText([7]), '7');
eq(dimsText([]), '');

eq(paramsText(0), '0'); eq(paramsText(76), '76'); eq(paramsText(999), '999');
eq(paramsText(1500), '1.5K'); eq(paramsText(163_000_000), '163M'); eq(paramsText(1_100_000_000), '1.1B');
eq(paramsText(12_300_000), '12.3M'); eq(paramsText(100_000), '100K');
eq(paramsText(-1), '–'); eq(paramsText(NaN), '–'); eq(paramsText(Infinity), '–');

eq(timeText(0), '1970-01-01 00:00');
eq(timeText(1_763_395_440), '2025-11-17 16:04');   // checked against: date -u -d @1763395440
eq(timeText(null), '–'); eq(timeText(NaN), '–');

const info = { architecture: 'gpt2', name: 'tiny test', parameters: 76, file_size: 1536, n_tensors: 2, n_kv: 5 };
eq(summaryLine(info), 'gpt2 · tiny test · 76 parameters · 1.5 KiB file · 2 tensors · 5 metadata entries');
eq(summaryLine({ ...info, architecture: null, name: null, n_tensors: 1, n_kv: 1 }),
  'unknown architecture · 76 parameters · 1.5 KiB file · 1 tensor · 1 metadata entry');

eq(kindsText([{ kind: 'Q8_0', tensors: 1, bytes: 68 }, { kind: 'F32', tensors: 1, bytes: 48 }]), 'Q8_0 59% · F32 41%');
eq(kindsText([{ kind: 'F32', tensors: 3, bytes: 100 }]), 'F32 100%');
eq(kindsText([{ kind: 'F32', tensors: 3, bytes: 0 }, { kind: 'Q4_0', tensors: 1, bytes: 0 }]), 'F32 ×3 · Q4_0 ×1');
eq(kindsText([]), 'no tensors');

const row = { name: 'blk.0.attn_q.weight', kind: 'Q8_0' };
ok(rowMatches(row, '')); ok(rowMatches(row, '  ')); ok(rowMatches(row, undefined));
ok(rowMatches(row, 'BLK.0')); ok(rowMatches(row, 'q8')); ok(rowMatches(row, ' attn_q '));
ok(!rowMatches(row, 'blk.1')); ok(!rowMatches(row, 'f32'));

console.log(`OK ${checks} checks`);
