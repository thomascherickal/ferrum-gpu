// The Capable tab's pure helpers: the query for `/api/capable`, and how a speed is written.

const blank = (v) => String(v ?? '').trim() === '';
const num = (v) => Number(String(v).trim());

/** The query parameters for the memory the user typed (and optionally a bandwidth and a model size). */
export function capableQuery(f) {
  if (blank(f.memory)) return { error: 'Enter how much memory the model must fit in, in gigabytes.' };
  const m = num(f.memory);
  if (!Number.isFinite(m) || m <= 0 || m > 1e6) return { error: 'Memory must be a number above 0 gigabytes.' };
  const params = { memory_gb: String(m), unified: f.unified ? '1' : '0' };
  const b = blank(f.bandwidth), s = blank(f.model);
  if (b !== s) return { error: 'For a speed estimate give both the memory bandwidth and the model size, or neither.' };
  if (!b) {
    const bw = num(f.bandwidth), mb = num(f.model);
    if (!Number.isFinite(bw) || bw <= 0) return { error: 'Bandwidth must be a number above 0 gigabytes per second.' };
    if (!Number.isFinite(mb) || mb <= 0) return { error: 'The model size must be a number above 0 billion parameters.' };
    params.bandwidth = String(bw);
    params.model_b = String(mb);
  }
  return { params };
}

/** Tokens per second as `238`, `23.8` or `2.4`; whole numbers from 100 up. */
export function tpsText(v) {
  if (!Number.isFinite(v) || v < 0) return '–';
  if (v >= 100) return String(Math.round(v));
  if (v >= 10) return v.toFixed(0);
  return v.toFixed(1);
}
