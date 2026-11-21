// Node tests of the Studio's shared helpers: API calls, the shared store and path helpers.
// usage: node studio.test.mjs
import assert from 'node:assert/strict';
import { eq, ok, counted, summary } from './lib/check.js';
import { esc, api, post, withQuery, humanBytes, humanTime } from '../web/shell/api.js';
import { createStore } from '../web/shell/state.js';
import { separator, join, crumbs, baseName, matchesExt } from '../web/files/paths.js';
import { fakeFetch } from './lib/fake_fetch.js';

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
  await assert.rejects(api('/bad', {}, f), /nope/); counted();
  await assert.rejects(api('/plain', {}, f), /HTTP 500|undefined/); counted();
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

// ---- files/paths.js
eq(separator('/a/b'), '/'); eq(separator('C:\\a\\b'), '\\'); eq(separator('a'), '/');
eq(join('/a', 'b'), '/a/b'); eq(join('/a/', 'b'), '/a/b'); eq(join('C:\\a', 'b'), 'C:\\a\\b'); eq(join('/', 'x'), '/x');
eq(crumbs('/a/b'), [['/', '/'], ['a', '/a'], ['b', '/a/b']]);
eq(crumbs('/'), [['/', '/']]);
eq(crumbs('rel/x'), [['rel', 'rel'], ['x', 'rel/x']]);
eq(baseName('/a/b.txt'), 'b.txt'); eq(baseName('C:\\x\\y.gguf'), 'y.gguf'); eq(baseName('/'), '/');
ok(matchesExt('M.GGUF', ['.gguf'])); ok(!matchesExt('m.txt', ['.gguf'])); ok(matchesExt('x', [])); ok(matchesExt('x'));

summary();
