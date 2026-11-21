// The model's hyperparameters and RoPE tables, read from GGUF metadata.
import { fail } from './util.js';

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
