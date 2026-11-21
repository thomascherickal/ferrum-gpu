// Upload a GGUF file's weights to the GPU: norms, projections (packed when Q8_0 / Q4_0), the
// embedding and head, the RoPE tables, scratch buffers and the compiled kernels.
import { TYPE, packQ } from '../gguf.js';
import { fail, transpose } from './util.js';
import { ropeTables } from './config.js';

const KERNELS = ['llama/gather', 'llama/rmsnorm', 'llama/rope', 'llama/swiglu', 'llama/attn_gqd_probs', 'llama/attn_gqd_pv',
  'llama/attn_gqa_scores', 'llama/attn_gqa_pv', 'fwd/attn_softmax', 'fwd/add', 'fwd/bias_add', 'matmul/matmul_blocked4', 'quant/qlinear'];

/** Fill `m` (a BrowserLlama) with the GPU buffers for the tensors of `g` under `cfg`. */
export async function uploadWeights(m, g, cfg) {
  const gpu = m.gpu;
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
}
