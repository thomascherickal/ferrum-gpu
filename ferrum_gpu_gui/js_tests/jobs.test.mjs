// Node tests of the jobs client: start / stop, and the follow loop with a fake fetch and a manual timer.
// usage: node jobs.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { startJob, stopJob, follow, formatLine, statusText, FINISHED } from '../web/jobs/client.js';
import { fakeFetch } from './lib/fake_fetch.js';

// ---- jobs/client.js: a manual timer so the polling is deterministic
function manualTimers() {
  const q = [];
  let next = 1;
  return {
    setTimeout: (fn) => { const id = next++; q.push([id, fn]); return id; },
    clearTimeout: (id) => { const i = q.findIndex((e) => e[0] === id); if (i >= 0) q.splice(i, 1); },
    async run() { const e = q.shift(); if (e) await e[1](); return !!e; },
    size: () => q.length,
  };
}
{
  const f = fakeFetch({ 'POST /api/jobs': [200, { id: 7 }], 'POST /api/jobs/7/stop': [200, { stopping: true }] });
  eq(await startJob('train', ['--x'], f), 7);
  eq(JSON.parse(f.calls[0].body), { cmd: 'train', args: ['--x'] });
  eq(await stopJob(7, true, f), { stopping: true });
  eq(JSON.parse(f.calls[1].body), { force: true });
  eq(FINISHED, ['done', 'stopped', 'error']);
}
{
  // two polls then done; `since` advances with `next`
  const polls = [
    { state: 'running', lines: [{ i: 0, e: false, t: 'a' }], next: 1, step: null },
    { state: 'running', lines: [], next: 1, step: { step: 1, total: 2, loss: 1.5 } },
    { state: 'done', exit: 0, lines: [{ i: 1, e: true, t: 'b' }], next: 2 },
  ];
  let n = 0;
  const f = fakeFetch(Object.fromEntries([0, 1, 2].map((since) => [`GET /api/jobs/3?since=${since}`, () => [200, polls[n++]]])));
  // since values requested: 0, 1, 1 -> only 0..1 are registered, so add the repeat
  f.calls.length = 0;
  const t = manualTimers();
  const lines = []; const states = []; let done = null;
  const g = fakeFetch({
    'GET /api/jobs/3?since=0': () => [200, polls[0]],
    'GET /api/jobs/3?since=1': (() => { let c = 0; return () => [200, polls[1 + c++]]; })(),
  });
  follow(3, { onLines: (l) => lines.push(...l), onState: (p) => states.push(p.state), onDone: (p) => { done = p; } }, { fetch: g, ...t });
  while (await t.run()) { /* run timers until nothing is scheduled */ }
  eq(lines.map((l) => l.t), ['a', 'b']);
  eq(states, ['running', 'running', 'done']);
  eq(done.state, 'done');
  eq(g.calls.map((c) => c.key), ['GET /api/jobs/3?since=0', 'GET /api/jobs/3?since=1', 'GET /api/jobs/3?since=1']);
  eq(t.size(), 0);
}
{
  // polling gives up after the retry limit with a synthetic error
  const t = manualTimers();
  let done = null;
  follow(9, { retries: 2, onDone: (p) => { done = p; } }, { fetch: fakeFetch({}), ...t });
  while (await t.run()) { /* drain */ }
  eq(done.state, 'error'); ok(/no route/.test(done.error));
}
{
  // a failure followed by success resets the count; stop() halts polling without calling onDone
  const t = manualTimers();
  let calls = 0; let done = false;
  const f = async () => { calls++; if (calls === 1) throw new Error('blip'); return { ok: true, status: 200, headers: { get: () => 'application/json' }, json: async () => ({ state: 'running', lines: [], next: 0 }) }; };
  const h = follow(1, { retries: 3, onDone: () => { done = true; } }, { fetch: f, ...t });
  await t.run(); await t.run();
  eq(calls, 2);
  h.stop();
  eq(t.size(), 0); eq(done, false);
  await t.run();
  eq(calls, 2);
  h.stop();   // stopping twice is harmless
}
eq(formatLine({ e: false, t: 'x' }), 'x'); eq(formatLine({ e: true, t: 'x' }), '! x');
eq(statusText({ state: 'running' }), 'running…');
eq(statusText({ state: 'running', step: { step: 3, total: 9, loss: 1.23456 } }), 'running · step 3/9 · loss 1.235');
eq(statusText({ state: 'stopping' }), 'stopping…'); eq(statusText({ state: 'stopped' }), 'stopped'); eq(statusText({ state: 'done' }), 'done');
eq(statusText({ state: 'error', error: 'boom' }), 'error: boom'); eq(statusText({ state: 'error', exit: 3 }), 'failed (exit 3)');

summary();
