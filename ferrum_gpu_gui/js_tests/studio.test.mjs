// Node tests of the Studio's pure logic: API helpers, the shared store, the jobs follow loop (with a
// fake fetch and a manual timer), path helpers, argument builders and dataset warnings.
// usage: node studio.test.mjs
import assert from 'node:assert/strict';
import { esc, api, post, withQuery, humanBytes, humanTime } from '../web/shell/api.js';
import { createStore } from '../web/shell/state.js';
import { startJob, stopJob, follow, formatLine, statusText, FINISHED } from '../web/jobs/client.js';
import { separator, join, crumbs, baseName, matchesExt } from '../web/files/paths.js';
import { prepareArgs, tokenizeArgs, datasetWarnings, describeStat } from '../web/tabs/args.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

// a fake fetch: `routes` maps "METHOD path-with-query" to a handler returning [status, json]
function fakeFetch(routes) {
  const calls = [];
  const f = async (path, opts = {}) => {
    const key = `${opts.method || 'GET'} ${path}`;
    calls.push({ key, body: opts.body });
    const h = routes[key];
    if (!h) return { ok: false, status: 404, headers: { get: () => 'application/json' }, json: async () => ({ error: `no route ${key}` }) };
    const [status, body] = typeof h === 'function' ? h(opts) : h;
    return { ok: status < 400, status, headers: { get: () => 'application/json' }, json: async () => body };
  };
  f.calls = calls;
  return f;
}

// ---- api.js
eq(esc('<a href="x">&\'</a>'), '&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;');
eq(withQuery('/p', {}), '/p');
eq(withQuery('/p', { a: 1, b: 'x y', c: undefined, d: null }), '/p?a=1&b=x%20y');
eq(humanBytes(0), '0 B'); eq(humanBytes(1023), '1023 B'); eq(humanBytes(1536), '1.5 KiB');
eq(humanBytes(5 * 1024 * 1024), '5.0 MiB'); eq(humanBytes(3 * 1024 ** 4), '3.0 TiB'); eq(humanBytes(5000 * 1024 ** 4), '5000.0 TiB');
eq(humanBytes(-1), '–'); eq(humanBytes(NaN), '–');
eq(humanTime(5), '5s'); eq(humanTime(65), '1m 05s'); eq(humanTime(3725), '1h 02m'); eq(humanTime(-1), '–');
{
  const f = fakeFetch({ 'GET /ok': [200, { v: 1 }], 'GET /bad': [400, { error: 'nope' }], 'GET /plain': [500, {}] });
  eq(await api('/ok', {}, f), { v: 1 });
  await assert.rejects(api('/bad', {}, f), /nope/); checks++;
  await assert.rejects(api('/plain', {}, f), /HTTP 500|undefined/); checks++;
  const f2 = fakeFetch({ 'POST /x': [200, { got: true }] });
  eq(await post('/x', { a: 1 }, f2), { got: true });
  eq(f2.calls[0].body, '{"a":1}');
  // a non-JSON reply is returned as bytes
  const bin = async () => ({ ok: true, status: 200, headers: { get: () => 'application/octet-stream' }, arrayBuffer: async () => new ArrayBuffer(3) });
  eq((await api('/b', {}, bin)).byteLength, 3);
}

// ---- state.js
{
  const s = createStore({ a: 1 });
  const seen = [];
  const off = s.on('a', (v) => seen.push(v));
  s.set('a', 2); s.set('a', 2); s.set('a', 3);
  eq(seen, [2, 3]); eq(s.get('a'), 3); eq(s.get('zzz'), undefined);
  off(); s.set('a', 4); eq(seen, [2, 3]);
  s.set('b', 1);   // a key nobody listens to
}

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

// ---- files/paths.js
eq(separator('/a/b'), '/'); eq(separator('C:\\a\\b'), '\\'); eq(separator('a'), '/');
eq(join('/a', 'b'), '/a/b'); eq(join('/a/', 'b'), '/a/b'); eq(join('C:\\a', 'b'), 'C:\\a\\b'); eq(join('/', 'x'), '/x');
eq(crumbs('/a/b'), [['/', '/'], ['a', '/a'], ['b', '/a/b']]);
eq(crumbs('/'), [['/', '/']]);
eq(crumbs('rel/x'), [['rel', 'rel'], ['x', 'rel/x']]);
eq(baseName('/a/b.txt'), 'b.txt'); eq(baseName('C:\\x\\y.gguf'), 'y.gguf'); eq(baseName('/'), '/');
ok(matchesExt('M.GGUF', ['.gguf'])); ok(!matchesExt('m.txt', ['.gguf'])); ok(matchesExt('x', [])); ok(matchesExt('x'));

// ---- tabs/args.js: prepare
{
  const base = { inputs: 'a.txt\n\n  b/  \n', out: 'out' };
  eq(prepareArgs(base), { args: ['--input', 'a.txt', '--input', 'b/', '--out', 'out'] });
  eq(prepareArgs({ ...base, nfkc: true, lowercase: true, stripGutenberg: true, dedup: true, lossy: true,
    minChars: '5', maxChars: '100', seed: '7', valFrac: '0.1', testFrac: '0.2', docSep: '<|doc|>' }).args,
  ['--input', 'a.txt', '--input', 'b/', '--out', 'out', '--nfkc', '--lowercase', '--strip-gutenberg', '--dedup', '--lossy',
    '--min-chars', '5', '--max-chars', '100', '--seed', '7', '--val-frac', '0.1', '--test-frac', '0.2', '--doc-sep', '<|doc|>']);
  for (const [bad, needle] of [
    [{ inputs: '', out: 'o' }, 'at least one input'],
    [{ inputs: '  \n ', out: 'o' }, 'at least one input'],
    [{ inputs: 'a', out: ' ' }, 'output folder'],
    [{ ...base, minChars: 'x' }, 'Minimum characters'],
    [{ ...base, maxChars: '-1' }, 'Maximum characters'],
    [{ ...base, seed: '1.5' }, 'Seed'],
    [{ ...base, valFrac: '1' }, 'Validation share'],
    [{ ...base, testFrac: '-0.1' }, 'Test share'],
    [{ ...base, valFrac: 'abc' }, 'Validation share'],
    [{ ...base, valFrac: '0.6', testFrac: '0.5' }, 'together'],
    [{ ...base, valFrac: '0.97' }, 'together'],
    [{ ...base, minChars: '10', maxChars: '5' }, 'at least the minimum'],
  ]) {
    const r = prepareArgs(bad);
    ok(r.error && r.error.includes(needle), `${JSON.stringify(bad)} -> ${r.error}`);
  }
  eq(prepareArgs({ ...base, maxChars: '0', minChars: '3' }).args.includes('--max-chars'), true);   // zero = no limit is accepted
  eq(prepareArgs({ inputs: 'a', out: 'o', docSep: '' }).args, ['--input', 'a', '--out', 'o']);
}
// ---- tokenize
eq(tokenizeArgs({ data: 'd', tokenizer: 't.gguf', out: 'o.gguf' }), { args: ['--data', 'd', '--tokenizer', 't.gguf', '--out', 'o.gguf'] });
eq(tokenizeArgs({ data: ' d ', tokenizer: 't', out: 'o', docSep: 'SEP' }).args.slice(-2), ['--doc-sep', 'SEP']);
ok(/text file or folder/.test(tokenizeArgs({ data: '', tokenizer: 't', out: 'o' }).error));
ok(/tokenizer file/.test(tokenizeArgs({ data: 'd', tokenizer: '', out: 'o' }).error));
ok(/token file/.test(tokenizeArgs({ data: 'd', tokenizer: 't', out: '' }).error));
ok(/text file or folder/.test(tokenizeArgs({}).error));
// ---- dataset text
const hb = (n) => `${n}B`;
eq(describeStat({ kind: 'missing' }, null, hb), 'No such file or folder.');
eq(describeStat({ kind: 'dir', files: 1, bytes: 10, truncated: false }, null, hb), 'Folder · 1 file · 10B');
eq(describeStat({ kind: 'dir', files: 3, bytes: 30, truncated: true }, null, hb), 'Folder · at least 3 files · at least 30B');
eq(describeStat({ kind: 'file', bytes: 5 }, { lines: 1 }, hb), 'File · 5B · 1 line');
eq(describeStat({ kind: 'file', bytes: 5 }, { lines: 4 }, hb), 'File · 5B · 4 lines');
eq(describeStat({ kind: 'file', bytes: 5 }, null, hb), 'File · 5B');
eq(datasetWarnings({ kind: 'missing' }, null), ['This path does not exist.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 50000 }, { binary: false }), []);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 0 }, { binary: false }), ['The file is empty.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 500 }, { binary: false }), ['Very little text: a model trained on it will mostly memorise it.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 5000 }, { binary: true }), ['This looks like a binary file, not text.']);
eq(datasetWarnings({ kind: 'dir', files: 0, bytes: 0, truncated: false }, null), ['There are no files here.']);
eq(datasetWarnings({ kind: 'dir', files: 9, bytes: 99999, truncated: true }, null), ['The folder is large: the file count and size are lower bounds.']);

console.log(`OK ${checks} checks`);
