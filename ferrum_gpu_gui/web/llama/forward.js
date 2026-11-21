// The two forward passes of BrowserLlama, one GPU command stream each: a single decode step that
// keeps a KV cache, and a stateless causal pass over a whole sequence.
import { fail } from './util.js';

/** Consume one token at the next position; resolves to its next-token logits. */
export async function feedToken(m, token) {
  const { gpu, cfg: c, s, p } = m;
  if (m.pos >= c.ctx) fail(`context of ${c.ctx} tokens is full`);
  if (!Number.isInteger(token) || token < 0 || token >= c.vocab) fail(`token ${token} is out of range for vocab ${c.vocab}`);
  const d = c.dim, qd = c.heads * c.headDim, kvd = c.kvHeads * c.headDim;
  gpu.write(s.ids, new Uint32Array([token]));
  gpu.begin();
  gpu.dispatch1d(p.gather, [m.tok, s.ids, s.x], [1, d], d);
  for (const b of m.blocks) {
    m.rms(s.x, b.attnNorm, s.h, 1);
    m.lin(s.h, b.wq, b.bq, s.q, s.tmpq, 1);
    m.lin(s.h, b.wk, b.bk, s.k, s.tmpkv, 1);
    m.lin(s.h, b.wv, b.bv, s.v, s.tmpkv, 1);
    m.rope(s.q, 1, c.heads, m.pos);
    m.rope(s.k, 1, c.kvHeads, m.pos);
    gpu.copy(s.k, 0, b.kcache, m.pos * kvd, kvd);
    gpu.copy(s.v, 0, b.vcache, m.pos * kvd, kvd);
    const u = [c.heads, c.kvHeads, c.headDim, m.pos + 1, c.ctx];
    gpu.dispatch1d(p.attn_gqd_probs, [s.q, b.kcache, s.probs], u, c.heads);
    gpu.dispatch1d(p.attn_gqd_pv, [s.probs, b.vcache, s.attn], u, qd);
    m.lin(s.attn, b.wo, null, s.o, s.tmpq, 1);
    m.add(s.x, s.o, s.x2, d);
    m.rms(s.x2, b.ffnNorm, s.h, 1);
    m.lin(s.h, b.gate, null, s.gate, s.tmpq, 1);
    m.lin(s.h, b.up, null, s.up, s.tmpq, 1);
    gpu.dispatch1d(p.swiglu, [s.gate, s.up, s.act], [c.hidden], c.hidden);
    m.lin(s.act, b.down, null, s.f2, s.tmpq, 1);
    m.add(s.x2, s.f2, s.x, d);
  }
  m.rms(s.x, m.outNorm, s.h, 1);
  m.lin(s.h, m.head, null, s.logits, s.tmpq, 1);
  gpu.end();
  const logits = await gpu.read(s.logits, c.vocab);
  m.pos += 1;
  return logits;
}


/** Stateless causal forward pass over `ids`; resolves to every position's logits (t * vocab). */
export async function forwardAll(m, ids) {
  const { gpu, cfg: c, p } = m, t = ids.length;
  if (t === 0 || t > c.ctx) fail(`sequence length ${t} must be in 1..=${c.ctx}`);
  if (ids.some((i) => !Number.isInteger(i) || i < 0 || i >= c.vocab)) fail(`a token is out of range for vocab ${c.vocab}`);
  const d = c.dim, qd = c.heads * c.headDim, kvd = c.kvHeads * c.headDim, h = c.hidden, buf = (n, len) => gpu.buffer(n, len);
  const gi = gpu.buffer('ids', new Uint32Array(ids));
  const x = buf('x', t * d), x2 = buf('x2', t * d), hh = buf('h', t * d), o = buf('o', t * d), f2 = buf('f2', t * d);
  const q = buf('q', t * qd), a = buf('a', t * qd), k = buf('k', t * kvd), v = buf('v', t * kvd);
  const gate = buf('gate', t * h), up = buf('up', t * h), act = buf('act', t * h);
  const tmpq = buf('tmpq', t * qd), tmpkv = buf('tmpkv', t * kvd), probs = buf('probs', c.heads * t * t), logits = buf('logits', t * c.vocab);
  gpu.begin();
  gpu.dispatch1d(p.gather, [m.tok, gi, x], [t, d], t * d);
  for (const b of m.blocks) {
    m.rms(x, b.attnNorm, hh, t);
    m.lin(hh, b.wq, b.bq, q, tmpq, t);
    m.lin(hh, b.wk, b.bk, k, tmpkv, t);
    m.lin(hh, b.wv, b.bv, v, tmpkv, t);
    m.rope(q, t, c.heads, 0);
    m.rope(k, t, c.kvHeads, 0);
    const u = [1, t, c.heads, c.headDim, c.kvHeads];
    gpu.dispatch1d(p.attn_gqa_scores, [q, k, probs], u, c.heads * t * t);
    gpu.dispatch1d(p.attn_softmax, [probs], u.slice(0, 4), c.heads * t);
    gpu.dispatch1d(p.attn_gqa_pv, [probs, v, a], u, t * qd);
    m.lin(a, b.wo, null, o, tmpq, t);
    m.add(x, o, x2, t * d);
    m.rms(x2, b.ffnNorm, hh, t);
    m.lin(hh, b.gate, null, gate, tmpq, t);
    m.lin(hh, b.up, null, up, tmpq, t);
    gpu.dispatch1d(p.swiglu, [gate, up, act], [t * h], t * h);
    m.lin(act, b.down, null, f2, tmpq, t);
    m.add(x2, f2, x, t * d);
  }
  m.rms(x, m.outNorm, hh, t);
  m.lin(hh, m.head, null, logits, tmpq, t);
  gpu.end();
  const out = await gpu.read(logits, t * c.vocab);
  for (const b of [gi, x, x2, hh, o, f2, q, a, k, v, gate, up, act, tmpq, tmpkv, probs, logits]) b.destroy();
  return out;
}
