// Node-level tests of the browser runtime's pure parts (no WebGPU needed): the GGUF reader, the
// device repack and the RoPE tables, compared bit for bit / to f32 rounding against reference
// values the Rust side produced from the same file bytes.
// usage: node reader.test.mjs <server-url>
import assert from 'node:assert/strict';
import { Gguf, packQ, f16, TYPE } from '../web/gguf.js';
import { groups1d, f32bits } from '../web/webgpu.js';
import { readConfig, ropeTables } from '../web/llama.js';

const base = process.argv[2];
const { cases } = await (await fetch(`${base}/api/selftest/vectors`)).json();
const buf = (b64) => Uint8Array.from(Buffer.from(b64, 'base64')).buffer;
let checks = 0;
const ok = (cond, msg) => { assert.ok(cond, msg); checks++; };

// f16 conversion: specials, subnormals, rounding-free exactness
assert.equal(f16(0x3c00), 1); assert.equal(f16(0xc000), -2); assert.equal(f16(0), 0);
assert.ok(Object.is(f16(0x8000), -0)); assert.equal(f16(0x7c00), Infinity); assert.ok(Number.isNaN(f16(0x7e00)));
assert.equal(f16(0x0001), 2 ** -24); assert.equal(f16(0x7bff), 65504); checks += 7;

// 1-D dispatch geometry (mirror of runtime::groups_1d)
assert.deepEqual(groups1d(1), [1, 1]); assert.deepEqual(groups1d(64), [1, 1]); assert.deepEqual(groups1d(65), [2, 1]);
assert.deepEqual(groups1d(65535 * 64), [65535, 1]); assert.deepEqual(groups1d(65535 * 64 + 1), [65535, 2]);
assert.throws(() => groups1d(2 ** 40), /1-D dispatch limit/); checks += 6;
assert.equal(f32bits(1), 0x3f800000); assert.equal(f32bits(-0), 0x80000000); checks += 2;

for (const c of cases) {
  const g = new Gguf(buf(c.gguf)), p = c.probe;
  // the tensor decodes to the Rust reader's values, exactly
  const got = g.f32(p.tensor);
  ok(got.length === p.elements, `${c.name}: element count`);
  p.decoded.forEach((want, i) => assert.equal(got[i], Math.fround(want), `${c.name}: ${p.tensor}[${i}]`));
  checks += p.decoded.length;
  // the device repack is the Rust repack, word for word
  if (p.packed) {
    const t = g.tensor(p.tensor), k = t.dims[0], n = t.dims[1];
    const words = packQ(t.type, g.raw(t), k, n);
    ok(words.length === p.packed_len, `${c.name}: packed length`);
    p.packed.forEach((w, i) => assert.equal(words[i], w, `${c.name}: packed[${i}]`));
    checks += p.packed.length;
  } else ok(g.tensor(p.tensor).type === TYPE.F32, `${c.name}: stored as f32`);
  // metadata becomes a valid config
  const cfg = readConfig(g);
  ok(cfg.vocab === c.vocab && cfg.ctx === 10 && cfg.dim === 32, `${c.name}: config`);
  // RoPE tables match the Rust ones (f32 rounding apart)
  const { cos, sin } = ropeTables(cfg, cfg.ctx);
  ok(cos.length === p.rope_cos.length, `${c.name}: rope table size`);
  p.rope_cos.forEach((w, i) => assert.ok(Math.abs(cos[i] - w) < 2e-7 && Math.abs(sin[i] - p.rope_sin[i]) < 2e-7, `${c.name}: rope[${i}] ${cos[i]} vs ${w}`));
  checks += p.rope_cos.length;
}

// hostile input is an error, never a hang or a wrong answer
const good = new Uint8Array(buf(cases[0].gguf));
const bad = (name, bytes, re) => { assert.throws(() => new Gguf(bytes.buffer ?? bytes), re, name); checks++; };
bad('empty', new ArrayBuffer(0), /truncated/);
bad('magic', Uint8Array.from([1, 2, 3, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), /bad magic/);
const v2 = good.slice(); v2[4] = 2; bad('version', v2, /version 2/);
for (const cut of [8, 24, 100, Math.floor(good.length / 2), good.length - 1]) bad(`cut at ${cut}`, good.slice(0, cut), /truncated|past the end|exceed/);
const huge = good.slice(); new DataView(huge.buffer).setBigUint64(8, 2n ** 60n, true); bad('absurd tensor count', huge, /too large|exceed/);
assert.throws(() => packQ(TYPE.Q8_0, new Uint8Array(10), 33, 1), /cannot pack/); checks++;
console.log(`OK ${checks} checks over ${cases.length} cases`);
