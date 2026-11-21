// Tensor element types and their block layouts: f16 conversion, decoding to f32 exactly as
// ferrum_gguf::quant::decode does, and `packQ`, which repacks block-quantised rows into the device
// layout of ferrum_wgpu::ops::quant (one 4-byte-aligned record per block, block-major).
import { fail } from './fail.js';

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

/** The byte length of `n` elements of tensor `type`. */
export function tensorBytes(type, n) {
  if (type === TYPE.F32) return n * 4;
  if (type === TYPE.F16) return n * 2;
  if (type in BLOCK_BYTES) {
    if (n % QK) fail(`${n} elements is not a multiple of the block size ${QK}`);
    return (n / QK) * BLOCK_BYTES[type];
  }
  fail(`unsupported tensor type ${type}`);
}

/** Decode tensor `t` whose raw bytes are `b` to a Float32Array (row-major, GGML dim 0 contiguous). */
export function decodeF32(t, b) {
  const n = t.elements;
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

