// The 1-D dispatch geometry the shaders' `gid()` assumes, and f32 -> u32 bit casts for uniforms.

const GROUPS_X = 65535; // the 1-D dispatch geometry the shaders' `gid()` assumes

export const f32bits = (x) => { const f = new Float32Array([x]); return new Uint32Array(f.buffer)[0]; };

/** Workgroup counts for a 1-D kernel over n elements (mirror of runtime::groups_1d). */
export function groups1d(n) {
  const g = Math.max(Math.ceil(n / 64), 1), x = Math.min(g, GROUPS_X), y = Math.ceil(g / x);
  if (y > GROUPS_X) throw new Error(`${n} elements exceed the 1-D dispatch limit`);
  return [x, y];
}
