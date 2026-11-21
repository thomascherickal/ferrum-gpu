// Token sampling: greedy or temperature / top-k, with a seeded PRNG.

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
