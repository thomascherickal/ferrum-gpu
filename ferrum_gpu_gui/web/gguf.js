// GGUF v3 reader for the browser: metadata, tensor directory, and decoding of F32 / F16 / Q8_0 /
// Q4_0 exactly as ferrum_gguf::quant::decode does (the scale is converted from f16 once, then
// value = scale * quant in f32).
import { fail } from './gguf/fail.js';
import { Reader } from './gguf/reader.js';
import { TYPE, tensorBytes, decodeF32 } from './gguf/quant.js';

export { TYPE, RECORD_WORDS, f16, packQ } from './gguf/quant.js';

export class Gguf extends Reader {
  constructor(buffer) {
    super(buffer);
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
      t.nbytes = tensorBytes(t.type, t.elements);
      if (this.dataStart + t.offset + t.nbytes > this.bytes.length) fail(`tensor ${t.name} runs past the end of the file`);
      this.tensors.set(t.name, t);
    }
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
  f32(name) { const t = this.tensor(name); return decodeF32(t, this.raw(t)); }
}
