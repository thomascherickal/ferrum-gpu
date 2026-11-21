// Node tests of the loss chart's layout and drawing (against a fake canvas that records its calls)
// and the series helpers.
// usage: node train_chart.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { plan, draw, lossLabel } from '../web/charts/loss.js';
import { collect, runSummary } from '../web/tabs/gtrain.js';

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

summary();
