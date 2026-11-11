// WebGPU layer: device and limits, pipelines built from the native WGSL files (served from the
// Rust crate that owns them, with the same `gid` prelude), buffers with the binding-size check,
// and one command encoder per batch of dispatches.

const GROUPS_X = 65535; // the 1-D dispatch geometry the shaders' `gid()` assumes

export const f32bits = (x) => { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; };

/** Workgroup counts for a 1-D kernel over n elements (mirror of runtime::groups_1d). */
export function groups1d(n) {
  const g = Math.max(Math.ceil(n / 64), 1), x = Math.min(g, GROUPS_X), y = Math.ceil(g / x);
  if (y > GROUPS_X) throw new Error(`${n} elements exceed the 1-D dispatch limit`);
  return [x, y];
}

export class Gpu {
  /** Open an adapter. `options` is a GPURequestAdapterOptions ({powerPreference} or {forceFallbackAdapter}). */
  static async create(options = {}) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter(options);
    if (!adapter) throw new Error('WebGPU is available but no adapter was granted');
    // Ask for what the adapter offers for buffer sizes; everything else stays at the baseline.
    const l = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize },
    });
    const g = new Gpu(adapter, device);
    device.lost.then((info) => { g.lost = info.message || info.reason; });
    return g;
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.limits = device.limits;
    this.info = adapter.info || {};
    this.lost = null;
    this.pipes = new Map();
    this.prelude = null;
    this.encoder = null;
    this.pending = [];
  }

  describe() {
    const i = this.info;
    return { vendor: i.vendor || '', architecture: i.architecture || '', description: i.description || '', fallback: !!this.adapter.isFallbackAdapter, maxStorageBufferBindingSize: this.limits.maxStorageBufferBindingSize };
  }

  /** Build (once) the pipeline for the native kernel `group/name`. */
  async pipeline(path) {
    let p = this.pipes.get(path);
    if (!p) {
      if (this.prelude === null) this.prelude = await (await fetch('/shaders/prelude.wgsl')).text();
      const r = await fetch(`/shaders/${path}.wgsl`);
      if (!r.ok) throw new Error(`kernel ${path} is not served (${r.status})`);
      const module = this.device.createShaderModule({ code: `${this.prelude}\n${await r.text()}` });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`kernel ${path}: ${errors.map((m) => `${m.lineNum}: ${m.message}`).join('; ')}`);
      p = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipes.set(path, p);
    }
    return p;
  }

  async warm(paths) { for (const p of paths) await this.pipeline(p); }

  /** A storage buffer holding `data` (a typed array) or `len` zeroed f32s. */
  buffer(name, data) {
    const bytes = typeof data === 'number' ? Math.max(data, 1) * 4 : Math.max(data.byteLength, 4);
    if (bytes > this.limits.maxStorageBufferBindingSize) {
      throw new Error(`tensor ${name} needs ${bytes} bytes but this adapter binds at most ${this.limits.maxStorageBufferBindingSize}`);
    }
    const buf = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (typeof data !== 'number' && data.byteLength) this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
    return buf;
  }

  write(buf, data) { this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength); }

  begin() {
    if (this.encoder) throw new Error('a batch is already open');
    this.encoder = this.device.createCommandEncoder();
  }

  end() {
    const enc = this.encoder;
    this.encoder = null;
    this.device.queue.submit([enc.finish()]);
  }

  /** Record one dispatch: buffers bind at 0..n, then a uniform of u32 words (padded to 16 bytes). */
  dispatch(pipe, buffers, uniform, groups) {
    const words = new Uint32Array(Math.max(Math.ceil(uniform.length / 4), 1) * 4);
    uniform.forEach((w, i) => { words[i] = w; });
    const ubuf = this.device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(ubuf, 0, words);
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: buffers.length, resource: { buffer: ubuf } });
    const bg = this.device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups[0], groups[1] || 1); pass.end();
  }

  dispatch1d(pipe, buffers, uniform, n) { this.dispatch(pipe, buffers, uniform, groups1d(n)); }

  copy(src, srcOff, dst, dstOff, n) { this.encoder.copyBufferToBuffer(src, srcOff * 4, dst, dstOff * 4, n * 4); }

  /** Read `n` f32s from the start of `buf` (after all recorded work has been submitted). */
  async read(buf, n) {
    const rb = this.device.createBuffer({ size: Math.max(n * 4, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0, n * 4));
    rb.unmap(); rb.destroy();
    if (this.lost) throw new Error(`GPU device lost: ${this.lost}`);
    return out;
  }
}
