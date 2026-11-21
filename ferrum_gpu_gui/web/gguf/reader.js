// The binary cursor of a GGUF file: little-endian integers, strings and metadata values.
import { fail } from './fail.js';

export class Reader {
  constructor(buffer) {
    this.bytes = new Uint8Array(buffer);
    this.view = new DataView(buffer);
    this.pos = 0;
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
}
