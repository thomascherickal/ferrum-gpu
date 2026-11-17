// Node tests of the Capable tab's helpers.
// usage: node capable.test.mjs
import assert from 'node:assert/strict';
import { capableQuery, tpsText } from '../web/tabs/capable_view.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

eq(capableQuery({ memory: '16' }), { params: { memory_gb: '16', unified: '0' } });
eq(capableQuery({ memory: ' 24.5 ', unified: true }), { params: { memory_gb: '24.5', unified: '1' } });
eq(capableQuery({ memory: '24', bandwidth: '936', model: '7' }).params, { memory_gb: '24', unified: '0', bandwidth: '936', model_b: '7' });
eq(capableQuery({ memory: '8', bandwidth: '', model: '  ' }).params, { memory_gb: '8', unified: '0' });
for (const [f, needle] of [
  [{}, 'how much memory'],
  [{ memory: '  ' }, 'how much memory'],
  [{ memory: 'lots' }, 'above 0'],
  [{ memory: '0' }, 'above 0'],
  [{ memory: '-3' }, 'above 0'],
  [{ memory: '2000000' }, 'above 0'],
  [{ memory: '8', bandwidth: '100' }, 'both'],
  [{ memory: '8', model: '7' }, 'both'],
  [{ memory: '8', bandwidth: '0', model: '7' }, 'Bandwidth'],
  [{ memory: '8', bandwidth: 'fast', model: '7' }, 'Bandwidth'],
  [{ memory: '8', bandwidth: '100', model: '0' }, 'model size'],
  [{ memory: '8', bandwidth: '100', model: 'big' }, 'model size'],
]) {
  const r = capableQuery(f);
  ok(r.error && r.error.includes(needle), `${JSON.stringify(f)} -> ${r.error}`);
  eq(r.params, undefined);
}
eq(tpsText(237.7), '238'); eq(tpsText(100), '100'); eq(tpsText(23.84), '24'); eq(tpsText(2.44), '2.4'); eq(tpsText(0), '0.0');
eq(tpsText(-1), '–'); eq(tpsText(NaN), '–'); eq(tpsText(Infinity), '–');

console.log(`OK ${checks} checks`);
