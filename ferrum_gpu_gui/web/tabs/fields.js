// Form-field parsing shared by every argument builder: blank checks, whole numbers and ranges, and a
// table-driven way to append optional numeric flags. Pure, tested under node through the builders.

export const blank = (v) => String(v ?? '').trim() === '';
export const text = (v) => String(v ?? '').trim();

/** Parse a whole number in [min, max]; null on failure. */
export function whole(v, min, max = Number.MAX_SAFE_INTEGER) {
  const s = text(v);
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n >= min && n <= max ? n : null;
}

/** Parse a finite number for which `ok(n)` holds; null on failure. */
export function real(v, ok) {
  const s = text(v);
  if (s === '') return null;
  const n = Number(s);
  return Number.isFinite(n) && ok(n) ? n : null;
}

/**
 * Append `flag value` for each filled row `[value, flag, parse, message]`, in order. Returns the
 * first problem as `{ error: message }`, or null when every filled row parsed.
 */
export function numeric(args, rows) {
  for (const [v, flag, parse, message] of rows) {
    if (blank(v)) continue;
    const n = parse(v);
    if (n === null) return { error: message };
    args.push(flag, String(n));
  }
  return null;
}

/** The `--device`, `--gpu-adapter`, `--gpu-mem-mb` (and optionally `--gpu-weights`) and `--threads` flags. */
export function deviceFlags(f, args, { devices, deviceMessage, weights = null }) {
  if (!blank(f.device) && f.device !== 'auto') {
    if (!devices.includes(f.device)) return { error: deviceMessage };
    args.push('--device', f.device);
  }
  if (!blank(f.adapter)) args.push('--gpu-adapter', text(f.adapter));
  const mem = [f.gpuMem, '--gpu-mem-mb', (v) => whole(v, 1), 'The GPU memory budget must be a whole number of MiB, 1 or more.'];
  const threads = [f.threads, '--threads', (v) => whole(v, 1), 'Threads must be a whole number, 1 or more.'];
  const bad = numeric(args, [mem]);
  if (bad) return bad;
  if (weights && !blank(f.gpuWeights)) {
    if (!weights.includes(f.gpuWeights)) return { error: `GPU weights must be one of ${weights.join(', ')}.` };
    if (f.gpuWeights !== 'f32') args.push('--gpu-weights', f.gpuWeights);
  }
  return numeric(args, [threads]);
}
