// The GPU lab tab's pure helpers: the arguments of `ferrum-gpu lab`, the report a `--json` run prints,
// and the text of a memory meter. Tested under node.

const text = (v) => String(v ?? '').trim();

/** The arguments for `lab`: `{ args }` or `{ error }`. */
export function labArgs(f) {
  const args = [];
  if (text(f.adapter)) args.push('--gpu-adapter', text(f.adapter));
  const size = text(f.size) || 'small';
  if (!['small', 'medium'].includes(size)) return { error: 'The model size must be small or medium.' };
  args.push('--size', size);
  const reps = text(f.reps);
  if (reps) {
    if (!/^\d+$/.test(reps) || Number(reps) < 1 || Number(reps) > 200) {
      return { error: 'Repetitions must be a whole number from 1 to 200.' };
    }
    args.push('--reps', reps);
  }
  if (f.parityOnly) args.push('--parity-only');
  if (f.json) args.push('--json');
  return { args };
}

/** The lab report in a finished `--json` job's output lines, or `null` if there is none. */
export function reportFromLines(lines) {
  for (let i = lines.length - 1; i >= 0; i--) {
    const l = lines[i];
    if (l.e || !l.t.startsWith('{')) continue;
    try {
      const j = JSON.parse(l.t);
      if (j && j.format === 1 && Array.isArray(j.checks)) return j;
    } catch { /* not the report */ }
  }
  return null;
}

/** One sentence for the report: the verdict and what it covers. */
export function reportSummary(r) {
  const bad = r.checks.filter((c) => !c.ok).length;
  const speeds = r.speeds.length ? `, speed measured at ${r.speeds.length} precisions` : ', no speed measured';
  const who = r.adapter ? r.adapter.name : 'no adapter';
  return `${r.passed ? 'PASS' : 'FAIL'}: ${r.checks.length - bad} of ${r.checks.length} parity checks passed on ${who}${speeds}.`;
}

/** The text and the meter values of an adapter's memory meter. */
export function meterView(m) {
  if (!m) return { text: 'memory: not readable on this machine', value: 0, max: 0 };
  const free = m.free_mib;
  const what = free === null || free === undefined ? `${m.total_mib} MiB in total` : `${free} MiB free of ${m.total_mib} MiB`;
  return { text: `${what} (${m.source})`, value: free ?? 0, max: m.total_mib };
}
