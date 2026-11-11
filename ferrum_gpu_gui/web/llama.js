// Llama / Qwen2 inference in the browser on WebGPU. This is the browser twin of
// ferrum_wgpu::models::llama::GpuLlama: the same native WGSL kernels (served by the Rust crate),
// the same dispatch sequence and uniform layouts. A parity test (selftest.js) keeps it honest
// against the CPU oracle. Matrices stay packed when the file stores them as Q8_0 / Q4_0.

import { Gguf, TYPE, packQ } from './gguf.js';
import { f32bits } from './webgpu.js';

const fail = (m) => { throw new Error(m); };

function transpose(a, rows, cols) {
  const out = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = a[r * cols + c];
  return out;
}

/** The model's hyperparameters from GGUF metadata (llama and qwen2), validated like the Rust loader. */
export function readConfig(g) {
  const arch = g.str('general.architecture');
  if (arch !== 'llama' && arch !== 'qwen2') fail(`architecture is '${arch}'; only 'llama' and 'qwen2' are supported`);
  const k = (s) => g.uint(`${arch}.${s}`), ko = (s) => g.optUint(`${arch}.${s}`);
  const dim = k('embedding_length'), heads = k('attention.head_count');
  const emb = g.tensor('token_embd.weight');
  if (emb.dims.length !== 2 || emb.dims[0] !== dim) fail('token_embd.weight must be [embedding_length, vocab]');
  const layers = k('block_count');
  if (layers > g.tensors.size) fail(`block_count ${layers} exceeds the file's ${g.tensors.size} tensors`);
  const headDim = ko('attention.key_length') ?? (heads > 0 ? Math.floor(dim / heads) : 0);
  const cfg = {
    arch, neox: arch === 'qwen2', dim, heads, layers, headDim,
    kvHeads: ko('attention.head_count_kv') ?? heads,
    vocab: emb.dims[1], ctx: ko('context_length') ?? 2048, hidden: k('feed_forward_length'),
    ropeBase: g.optFloat(`${arch}.rope.freq_base`) ?? 10000,
    eps: g.optFloat(`${arch}.attention.layer_norm_rms_epsilon`) ?? 1e-5,
  };
  cfg.ropeDim = ko('rope.dimension_count') ?? headDim;
  for (const [n, v] of Object.entries({ dim, heads, layers, headDim, kvHeads: cfg.kvHeads, vocab: cfg.vocab, ctx: cfg.ctx, hidden: cfg.hidden, ropeDim: cfg.ropeDim })) {
    if (!(v > 0)) fail(`every dimension must be positive (${n} is ${v})`);
  }
  if (heads % cfg.kvHeads) fail(`heads ${heads} is not a multiple of kv_heads ${cfg.kvHeads}`);
  if (cfg.ropeDim > headDim || cfg.ropeDim % 2) fail(`rope_dim ${cfg.ropeDim} must be even and at most head_dim ${headDim}`);
  if (!(cfg.eps > 0) || !(cfg.ropeBase > 0)) fail('eps and rope_base must be positive');
  // RoPE scaling: none / linear / rope_freqs.weight. Anything else would run silently wrong.
  const key = (s) => `${arch}.rope.${s}`;
  let factor = g.optFloat(key('scaling.factor')) ?? g.optFloat(key('scale_linear'));
  const type = g.has(key('scaling.type')) ? g.str(key('scaling.type')) : 'linear';
  factor = factor ? factor : 1;
  if (type === 'linear') cfg.linear = factor;
  else if (type === 'none' && factor === 1) cfg.linear = 1;
  else fail(`rope scaling type '${type}' (factor ${factor}) is not supported`);
  if ((g.optFloat(key('scaling.attn_factor')) ?? 1) !== 1) fail('rope scaling attn_factor is not supported');
  cfg.freqFactors = g.tensors.has('rope_freqs.weight') ? g.f32('rope_freqs.weight') : null;
  if (cfg.freqFactors && cfg.freqFactors.length !== cfg.ropeDim / 2) fail('rope_freqs.weight has the wrong length');
  return cfg;
}

/** cos / sin tables for positions 0..n, host-built exactly like ferrum_ref::llama::rope_tables. */
export function ropeTables(cfg, n) {
  const half = cfg.ropeDim / 2, cos = new Float32Array(n * half), sin = new Float32Array(n * half);
  const inv = Math.fround(1 / cfg.linear);
  for (let p = 0; p < n; p++) {
    for (let i = 0; i < half; i++) {
      const freq = Math.fround(Math.pow(cfg.ropeBase, (-2 * i) / cfg.ropeDim));
      const ff = cfg.freqFactors ? cfg.freqFactors[i] : 1;
      const theta = Math.fround(Math.fround(Math.fround(p * freq) / ff) * inv);
      cos[p * half + i] = Math.cos(theta);
      sin[p * half + i] = Math.sin(theta);
    }
  }
  return { cos, sin };
}

const KERNELS = ['llama/gather', 'llama/rmsnorm', 'llama/rope', 'llama/swiglu', 'llama/attn_gqd_probs', 'llama/attn_gqd_pv',
  'llama/attn_gqa_scores', 'llama/attn_gqa_pv', 'fwd/attn_softmax', 'fwd/add', 'fwd/bias_add', 'matmul/matmul_blocked4', 'quant/qlinear'];

export class BrowserLlama {
  /** Parse `arrayBuffer` (a GGUF file) and upload its weights to `gpu`. */
  static async load(gpu, arrayBuffer) {
    const g = new Gguf(arrayBuffer), cfg = readConfig(g), m = new BrowserLlama(gpu, cfg);
    const { dim: d, hidden: h } = cfg, qd = cfg.heads * cfg.headDim, kvd = cfg.kvHeads * cfg.headDim;
    const vec = (name, n) => { const t = g.tensor(name); if (t.elements !== n) fail(`tensor '${name}' has ${t.elements} elements, expected ${n}`); return gpu.buffer(name, g.f32(name)); };
    const bias = (name, n) => (g.tensors.has(name) ? vec(name, n) : null);
    const lin = (name, k, n) => {
      const t = g.tensor(name);
      if (t.dims.length !== 2 || t.dims[0] !== k || t.dims[1] !== n) fail(`tensor '${name}' has dims [${t.dims}], expected [${k},${n}] from the metadata`);
      if ((t.type === TYPE.Q8_0 || t.type === TYPE.Q4_0) && k % 32 === 0) return { kind: t.type, buf: gpu.buffer(name, packQ(t.type, g.raw(t), k, n)), k, n };
      return { kind: TYPE.F32, buf: gpu.buffer(name, transpose(g.f32(name), n, k)), k, n };
    };
    const tok = g.f32('token_embd.weight');
    m.tok = gpu.buffer('token_embd.weight', tok);
    m.blocks = [];
    for (let i = 0; i < cfg.layers; i++) {
      const n = (s) => `blk.${i}.${s}`;
      for (const stem of ['attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']) if (g.tensors.has(n(`${stem}.bias`))) fail(`tensor '${n(`${stem}.bias`)}' is not supported`);
      m.blocks.push({
        attnNorm: vec(n('attn_norm.weight'), d), ffnNorm: vec(n('ffn_norm.weight'), d),
        wq: lin(n('attn_q.weight'), d, qd), bq: bias(n('attn_q.bias'), qd),
        wk: lin(n('attn_k.weight'), d, kvd), bk: bias(n('attn_k.bias'), kvd),
        wv: lin(n('attn_v.weight'), d, kvd), bv: bias(n('attn_v.bias'), kvd),
        wo: lin(n('attn_output.weight'), qd, d),
        gate: lin(n('ffn_gate.weight'), d, h), up: lin(n('ffn_up.weight'), d, h), down: lin(n('ffn_down.weight'), h, d),
        kcache: gpu.buffer(`blk${i}.kcache`, cfg.ctx * kvd), vcache: gpu.buffer(`blk${i}.vcache`, cfg.ctx * kvd),
      });
    }
    m.outNorm = vec('output_norm.weight', d);
    if (g.tensors.has('output.bias')) fail("tensor 'output.bias' is not supported");
    // A missing output.weight means the head is tied to the embedding (transposed).
    m.head = g.tensors.has('output.weight')
      ? lin('output.weight', d, cfg.vocab)
      : { kind: TYPE.F32, buf: gpu.buffer('head (tied)', transpose(tok, cfg.vocab, d)), k: d, n: cfg.vocab };
    const { cos, sin } = ropeTables(cfg, cfg.ctx);
    m.cos = gpu.buffer('rope_cos', cos); m.sin = gpu.buffer('rope_sin', sin);
    m.s = {
      ids: gpu.buffer('ids', new Uint32Array(1)), x: gpu.buffer('x', d), x2: gpu.buffer('x2', d), h: gpu.buffer('h', d),
      q: gpu.buffer('q', qd), k: gpu.buffer('k', kvd), v: gpu.buffer('v', kvd), attn: gpu.buffer('attn', qd), o: gpu.buffer('o', d),
      gate: gpu.buffer('gate', h), up: gpu.buffer('up', h), act: gpu.buffer('act', h), f2: gpu.buffer('f2', d),
      probs: gpu.buffer('probs', cfg.heads * cfg.ctx), logits: gpu.buffer('logits', cfg.vocab),
      tmpq: gpu.buffer('tmpq', qd), tmpkv: gpu.buffer('tmpkv', kvd),
    };
    await gpu.warm(KERNELS);
    m.p = Object.fromEntries(KERNELS.map((k) => [k.split('/')[1], gpu.pipes.get(k)]));
    return m;
  }

  constructor(gpu, cfg) { this.gpu = gpu; this.cfg = cfg; this.pos = 0; }

  reset() { this.pos = 0; }

  /** y = x * W (+ bias) for `rows` rows. `tmp` holds the product when a bias must be added. */
  lin(x, W, bias, y, tmp, rows) {
    const { gpu, p } = this, out = bias ? tmp : y;
    if (W.kind === TYPE.F32) gpu.dispatch(p.matmul_blocked4, [x, W.buf, out], [rows, W.k, W.n], [Math.ceil(W.n / 64), Math.ceil(rows / 64)]);
    else gpu.dispatch1d(p.qlinear, [x, W.buf, out], [rows, W.k, W.n, W.kind === TYPE.Q4_0 ? 1 : 0], rows * W.n);
    if (bias) gpu.dispatch1d(p.bias_add, [tmp, bias, y], [rows * W.n, W.n], rows * W.n);
  }

  rope(x, rows, heads, offset) {
    const c = this.cfg;
    this.gpu.dispatch1d(this.p.rope, [x, this.cos, this.sin], [rows, heads, c.headDim, c.ropeDim, c.neox ? 1 : 0, offset], rows * heads * (c.ropeDim / 2));
  }

  rms(x, gain, y, rows) { this.gpu.dispatch1d(this.p.rmsnorm, [x, gain, y], [rows, this.cfg.dim, f32bits(this.cfg.eps)], rows); }
  add(a, b, c, n) { this.gpu.dispatch1d(this.p.add, [a, b, c], [n], n); }

  /** Consume one token at the next position; resolves to its next-token logits. */
  async feed(token) {
    const { gpu, cfg: c, s, p } = this;
    if (this.pos >= c.ctx) fail(`context of ${c.ctx} tokens is full`);
    if (!Number.isInteger(token) || token < 0 || token >= c.vocab) fail(`token ${token} is out of range for vocab ${c.vocab}`);
    const d = c.dim, qd = c.heads * c.headDim, kvd = c.kvHeads * c.headDim;
    gpu.write(s.ids, new Uint32Array([token]));
    gpu.begin();
    gpu.dispatch1d(p.gather, [this.tok, s.ids, s.x], [1, d], d);
    for (const b of this.blocks) {
      this.rms(s.x, b.attnNorm, s.h, 1);
      this.lin(s.h, b.wq, b.bq, s.q, s.tmpq, 1);
      this.lin(s.h, b.wk, b.bk, s.k, s.tmpkv, 1);
      this.lin(s.h, b.wv, b.bv, s.v, s.tmpkv, 1);
      this.rope(s.q, 1, c.heads, this.pos);
      this.rope(s.k, 1, c.kvHeads, this.pos);
      gpu.copy(s.k, 0, b.kcache, this.pos * kvd, kvd);
      gpu.copy(s.v, 0, b.vcache, this.pos * kvd, kvd);
      const u = [c.heads, c.kvHeads, c.headDim, this.pos + 1, c.ctx];
      gpu.dispatch1d(p.attn_gqd_probs, [s.q, b.kcache, s.probs], u, c.heads);
      gpu.dispatch1d(p.attn_gqd_pv, [s.probs, b.vcache, s.attn], u, qd);
      this.lin(s.attn, b.wo, null, s.o, s.tmpq, 1);
      this.add(s.x, s.o, s.x2, d);
      this.rms(s.x2, b.ffnNorm, s.h, 1);
      this.lin(s.h, b.gate, null, s.gate, s.tmpq, 1);
      this.lin(s.h, b.up, null, s.up, s.tmpq, 1);
      gpu.dispatch1d(p.swiglu, [s.gate, s.up, s.act], [c.hidden], c.hidden);
      this.lin(s.act, b.down, null, s.f2, s.tmpq, 1);
      this.add(s.x2, s.f2, s.x, d);
    }
    this.rms(s.x, this.outNorm, s.h, 1);
    this.lin(s.h, this.head, null, s.logits, s.tmpq, 1);
    gpu.end();
    const logits = await gpu.read(s.logits, c.vocab);
    this.pos += 1;
    return logits;
  }

  /** Stateless causal forward pass over `ids`; resolves to every position's logits (t * vocab). */
  async forwardAll(ids) {
    const { gpu, cfg: c, p } = this, t = ids.length;
    if (t === 0 || t > c.ctx) fail(`sequence length ${t} must be in 1..=${c.ctx}`);
    if (ids.some((i) => !Number.isInteger(i) || i < 0 || i >= c.vocab)) fail(`a token is out of range for vocab ${c.vocab}`);
    const d = c.dim, qd = c.heads * c.headDim, kvd = c.kvHeads * c.headDim, h = c.hidden, buf = (n, len) => gpu.buffer(n, len);
    const gi = gpu.buffer('ids', new Uint32Array(ids));
    const x = buf('x', t * d), x2 = buf('x2', t * d), hh = buf('h', t * d), o = buf('o', t * d), f2 = buf('f2', t * d);
    const q = buf('q', t * qd), a = buf('a', t * qd), k = buf('k', t * kvd), v = buf('v', t * kvd);
    const gate = buf('gate', t * h), up = buf('up', t * h), act = buf('act', t * h);
    const tmpq = buf('tmpq', t * qd), tmpkv = buf('tmpkv', t * kvd), probs = buf('probs', c.heads * t * t), logits = buf('logits', t * c.vocab);
    gpu.begin();
    gpu.dispatch1d(p.gather, [this.tok, gi, x], [t, d], t * d);
    for (const b of this.blocks) {
      this.rms(x, b.attnNorm, hh, t);
      this.lin(hh, b.wq, b.bq, q, tmpq, t);
      this.lin(hh, b.wk, b.bk, k, tmpkv, t);
      this.lin(hh, b.wv, b.bv, v, tmpkv, t);
      this.rope(q, t, c.heads, 0);
      this.rope(k, t, c.kvHeads, 0);
      const u = [1, t, c.heads, c.headDim, c.kvHeads];
      gpu.dispatch1d(p.attn_gqa_scores, [q, k, probs], u, c.heads * t * t);
      gpu.dispatch1d(p.attn_softmax, [probs], u.slice(0, 4), c.heads * t);
      gpu.dispatch1d(p.attn_gqa_pv, [probs, v, a], u, t * qd);
      this.lin(a, b.wo, null, o, tmpq, t);
      this.add(x, o, x2, t * d);
      this.rms(x2, b.ffnNorm, hh, t);
      this.lin(hh, b.gate, null, gate, tmpq, t);
      this.lin(hh, b.up, null, up, tmpq, t);
      gpu.dispatch1d(p.swiglu, [gate, up, act], [t * h], t * h);
      this.lin(act, b.down, null, f2, tmpq, t);
      this.add(x2, f2, x, t * d);
    }
    this.rms(x, this.outNorm, hh, t);
    this.lin(hh, this.head, null, logits, tmpq, t);
    gpu.end();
    const out = await gpu.read(logits, t * c.vocab);
    for (const b of [gi, x, x2, hh, o, f2, q, a, k, v, gate, up, act, tmpq, tmpkv, probs, logits]) b.destroy();
    return out;
  }
}

/** Sample from a logits row: greedy at temperature 0, else temperature (and top-k) with a seeded PRNG. */
export function sample(logits, temperature, topK, rand) {
  if (temperature <= 0) { let b = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[b]) b = i; return b; }
  let idx = Array.from(logits, (_, i) => i).sort((a, b) => logits[b] - logits[a] || a - b);
  if (topK > 0) idx = idx.slice(0, topK);
  const m = logits[idx[0]], w = idx.map((i) => Math.exp((logits[i] - m) / temperature)), sum = w.reduce((a, b) => a + b, 0);
  let r = rand() * sum;
  for (let j = 0; j < idx.length; j++) { r -= w[j]; if (r < 0) return idx[j]; }
  return idx[idx.length - 1];
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
