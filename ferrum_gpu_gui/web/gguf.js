// GGUF v3 reader for the browser: metadata, tensor directory, and decoding of F32 / F16 / Q8_0 /
// Q4_0 exactly as ferrum_gguf::quant::decode does (the scale is converted from f16 once, then
// value = scale * quant in f32). `packQ` repacks block-quantised rows into the device layout of
// ferrum_wgpu::ops::quant (one 4-byte-aligned record per block, block-major).

export const TYPE = { F32: 0, F16: 1, Q4_0: 2, Q8_0: 8 };
const QK = 32;
const BLOCK_BYTES = { [TYPE.Q8_0]: 2 + QK, [TYPE.Q4_0]: 2 + QK / 2 };
export const RECORD_WORDS = { [TYPE.Q8_0]: 1 + QK / 4, [TYPE.Q4_0]: 1 + QK / 8 };

/** Half-precision bits to f32 (exact: every half is representable in f64, then in f32). */
export function f16(h) {
  const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  let v;
  if (e === 0) v = m * 2 ** -24;
  else if (e === 31) v = m ? NaN : Infinity;
  else v = (1 + m / 1024) * 2 ** (e - 15);
  return Math.fround(s ? -v : v);
}

function fail(msg) { throw new Error(`GGUF: ${msg}`); }

export class Gguf {
  constructor(buffer) {
    this.bytes = new Uint8Array(buffer);
    this.view = new DataView(buffer);
    this.pos = 0;
    this.kv = new Map();
    this.tensors = new Map();
    if (this.u32() !== 0x46554747) fail('bad magic (not a GGUF file)');
    const version = this.u32();
    if (version !== 3) fail(`version ${version} is not supported (3 is)`);
    const nTensors = this.u64(), nKv = this.u64();
    if (nTensors > this.bytes.length || nKv > this.bytes.length) fail('counts exceed the file size');
    for (let i = 0; i < nKv; i++) { const k = this.string(); this.kv.set(k, this.value(this.u32())); }
    const infos = [];
    for (let i = 0; i < nTensors; i++) {
      const name = this.string(), nd = this.u32();
      if (nd < 1 || nd > 4) fail(`tensor ${name} has ${nd} dimensions`);
      const dims = [];
      for (let d = 0; d < nd; d++) dims.push(this.u64());
      infos.push({ name, dims, type: this.u32(), offset: this.u64() });
    }
    const align = this.kv.has('general.alignment') ? Number(this.kv.get('general.alignment')) : 32;
    this.dataStart = Math.ceil(this.pos / align) * align;
    for (const t of infos) {
      t.elements = t.dims.reduce((a, d) => a * d, 1);
      if (!Number.isSafeInteger(t.elements)) fail(`tensor ${t.name} is too large`);
      t.nbytes = this.nbytes(t.type, t.elements);
      if (this.dataStart + t.offset + t.nbytes > this.bytes.length) fail(`tensor ${t.name} runs past the end of the file`);
      this.tensors.set(t.name, t);
    }
  }

  take(n) {
    if (!Number.isSafeInteger(n) || n < 0 || this.pos + n > this.bytes.length) fail('truncated file');
    const at = this.pos; this.pos += n; return at;
  }
  u32() { const a = this.take(4); return this.view.getUint32(a, true); }
  u64() {
    const a = this.take(8), v = this.view.getBigUint64(a, true);
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) fail('a 64-bit value is too large');
    return Number(v);
  }
  string() { const n = this.u64(), a = this.take(n); return new TextDecoder('utf-8', { fatal: true }).decode(this.bytes.subarray(a, a + n)); }
  value(type) {
    const v = this.view;
    switch (type) {
      case 0: return v.getUint8(this.take(1));
      case 1: return v.getInt8(this.take(1));
      case 2: return v.getUint16(this.take(2), true);
      case 3: return v.getInt16(this.take(2), true);
      case 4: return v.getUint32(this.take(4), true);
      case 5: return v.getInt32(this.take(4), true);
      case 6: return v.getFloat32(this.take(4), true);
      case 7: return v.getUint8(this.take(1)) !== 0;
      case 8: return this.string();
      case 9: {
        const et = this.u32(), n = this.u64();
        if (n > this.bytes.length) fail('array longer than the file');
        const out = [];
        for (let i = 0; i < n; i++) out.push(this.value(et));
        return out;
      }
      case 10: return this.u64();
      case 11: return Number(v.getBigInt64(this.take(8), true));
      case 12: return v.getFloat64(this.take(8), true);
      default: fail(`unknown value type ${type}`);
    }
  }

  nbytes(type, n) {
    if (type === TYPE.F32) return n * 4;
    if (type === TYPE.F16) return n * 2;
    if (type in BLOCK_BYTES) {
      if (n % QK) fail(`${n} elements is not a multiple of the block size ${QK}`);
      return (n / QK) * BLOCK_BYTES[type];
    }
    fail(`unsupported tensor type ${type}`);
  }

  has(key) { return this.kv.has(key); }
  uint(key) {
    const v = this.kv.get(key);
    if (typeof v !== 'number' || v < 0 || !Number.isInteger(v)) fail(`'${key}' is missing or not a non-negative integer`);
    return v;
  }
  optUint(key) { return this.kv.has(key) ? this.uint(key) : undefined; }
  optFloat(key) {
    if (!this.kv.has(key)) return undefined;
    const v = this.kv.get(key);
    if (typeof v !== 'number') fail(`'${key}' is not a number`);
    return v;
  }
  str(key) { const v = this.kv.get(key); if (typeof v !== 'string') fail(`'${key}' is missing or not a string`); return v; }

  tensor(name) { const t = this.tensors.get(name); if (!t) fail(`missing tensor '${name}'`); return t; }
  raw(t) { return this.bytes.subarray(this.dataStart + t.offset, this.dataStart + t.offset + t.nbytes); }

  /** Decode a tensor to Float32Array (row-major, GGML dim 0 contiguous). */
  f32(name) {
    const t = this.tensor(name), b = this.raw(t), n = t.elements;
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const out = new Float32Array(n);
    if (t.type === TYPE.F32) for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
    else if (t.type === TYPE.F16) for (let i = 0; i < n; i++) out[i] = f16(dv.getUint16(i * 2, true));
    else if (t.type === TYPE.Q8_0) {
      for (let blk = 0; blk < n / QK; blk++) {
        const o = blk * (2 + QK), d = f16(dv.getUint16(o, true));
        for (let i = 0; i < QK; i++) out[blk * QK + i] = Math.fround(d * dv.getInt8(o + 2 + i));
      }
    } else {
      for (let blk = 0; blk < n / QK; blk++) {
        const o = blk * (2 + QK / 2), d = f16(dv.getUint16(o, true));
        for (let j = 0; j < QK / 2; j++) {
          const q = dv.getUint8(o + 2 + j);
          out[blk * QK + j] = Math.fround(d * ((q & 15) - 8));
          out[blk * QK + j + QK / 2] = Math.fround(d * ((q >> 4) - 8));
        }
      }
    }
    return out;
  }
}

/** Repack the block bytes of a `k x n` weight (n rows of k, blocks along k) into device records. */
export function packQ(type, bytes, k, n) {
  const nb = k / QK, fb = BLOCK_BYTES[type], w = RECORD_WORDS[type];
  if (!(type in BLOCK_BYTES) || k % QK || bytes.length !== nb * n * fb) fail('cannot pack this tensor');
  const out = new Uint32Array(nb * n * w), f = new Float32Array(1), fu = new Uint32Array(f.buffer);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let col = 0; col < n; col++) {
    for (let b = 0; b < nb; b++) {
      const src = (col * nb + b) * fb, rec = (b * n + col) * w;
      f[0] = f16(dv.getUint16(src, true));
      out[rec] = fu[0];
      for (let i = 0; i < w - 1; i++) out[rec + 1 + i] = dv.getUint32(src + 2 + 4 * i, true);
    }
  }
  return out;
}
