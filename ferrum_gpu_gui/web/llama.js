// Llama / Qwen2 inference in the browser on WebGPU. This is the browser twin of
// ferrum_wgpu::models::llama::GpuLlama: the same native WGSL kernels (served by the Rust crate),
// the same dispatch sequence and uniform layouts. A parity test (selftest.js) keeps it honest
// against the CPU oracle. Matrices stay packed when the file stores them as Q8_0 / Q4_0.
import { Gguf, TYPE } from './gguf.js';
import { f32bits } from './webgpu.js';
import { readConfig } from './llama/config.js';
import { uploadWeights } from './llama/weights.js';
import { feedToken, forwardAll } from './llama/forward.js';

export { readConfig, ropeTables } from './llama/config.js';
export { sample, mulberry32 } from './llama/sampling.js';

export class BrowserLlama {
  /** Parse `arrayBuffer` (a GGUF file) and upload its weights to `gpu`. */
  static async load(gpu, arrayBuffer) {
    const g = new Gguf(arrayBuffer), cfg = readConfig(g), m = new BrowserLlama(gpu, cfg);
    await uploadWeights(m, g, cfg);
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
  feed(token) { return feedToken(this, token); }

  /** Stateless causal forward pass over `ids`; resolves to every position's logits (t * vocab). */
  forwardAll(ids) { return forwardAll(this, ids); }
}
