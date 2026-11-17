// Building `ferrum-gpu` argument lists from form values, and refusing bad values before a job is
// started. Pure functions: each returns `{ args }` or `{ error }`, so they are tested under node.

const isCount = (v) => /^\d+$/.test(String(v).trim());

/** A non-negative integer field; blank means "leave the CLI default". */
function count(name, v, out, flag) {
  const s = String(v ?? '').trim();
  if (s === '') return null;
  if (!isCount(s)) return `${name} must be a whole number`;
  out.push(flag, s);
  return null;
}

/** A fraction in [0, 1); blank leaves the default. */
function fraction(name, v, out, flag) {
  const s = String(v ?? '').trim();
  if (s === '') return null;
  const n = Number(s);
  if (!Number.isFinite(n) || n < 0 || n >= 1) return `${name} must be at least 0 and below 1`;
  out.push(flag, s);
  return null;
}

/** The arguments of `prepare`. `inputs` is the text of a box, one path per line. */
export function prepareArgs(f) {
  const inputs = String(f.inputs ?? '').split('\n').map((s) => s.trim()).filter(Boolean);
  if (!inputs.length) return { error: 'Add at least one input file or folder.' };
  const out = String(f.out ?? '').trim();
  if (!out) return { error: 'Choose the output folder.' };
  const args = [];
  for (const p of inputs) args.push('--input', p);
  args.push('--out', out);
  for (const [on, flag] of [[f.nfkc, '--nfkc'], [f.lowercase, '--lowercase'], [f.stripGutenberg, '--strip-gutenberg'], [f.dedup, '--dedup'], [f.lossy, '--lossy']]) {
    if (on) args.push(flag);
  }
  const errs = [
    count('Minimum characters', f.minChars, args, '--min-chars'),
    count('Maximum characters', f.maxChars, args, '--max-chars'),
    count('Seed', f.seed, args, '--seed'),
    fraction('Validation share', f.valFrac, args, '--val-frac'),
    fraction('Test share', f.testFrac, args, '--test-frac'),
  ].filter(Boolean);
  if (errs.length) return { error: errs[0] };
  const total = Number(f.valFrac || 0.05) + Number(f.testFrac || 0.05);
  if (total >= 1) return { error: 'The validation and test shares together must be below 1.' };
  const min = Number(f.minChars || 1);
  const max = Number(f.maxChars || 0);
  if (max !== 0 && max < min) return { error: 'Maximum characters must be 0 (no limit) or at least the minimum.' };
  if (String(f.docSep ?? '') !== '') args.push('--doc-sep', String(f.docSep));
  return { args };
}

/** The arguments of `tokenize`. */
export function tokenizeArgs(f) {
  const data = String(f.data ?? '').trim();
  const tokenizer = String(f.tokenizer ?? '').trim();
  const out = String(f.out ?? '').trim();
  if (!data) return { error: 'Choose the text file or folder to tokenize.' };
  if (!tokenizer) return { error: 'Choose the tokenizer file (the .gguf a training run saved).' };
  if (!out) return { error: 'Choose where to write the token file.' };
  const args = ['--data', data, '--tokenizer', tokenizer, '--out', out];
  if (String(f.docSep ?? '') !== '') args.push('--doc-sep', String(f.docSep));
  return { args };
}

/** One sentence about a dataset: what it is, how big, how many lines. `fmt` formats bytes. */
export function describeStat(stat, preview, fmt) {
  if (stat.kind === 'missing') return 'No such file or folder.';
  const lower = stat.truncated ? 'at least ' : '';
  if (stat.kind === 'dir') return `Folder · ${lower}${stat.files} file${stat.files === 1 ? '' : 's'} · ${lower}${fmt(stat.bytes)}`;
  const lines = preview ? ` · ${preview.lines} line${preview.lines === 1 ? '' : 's'}` : '';
  return `File · ${fmt(stat.bytes)}${lines}`;
}

/** Warnings about a dataset from its stat and preview; an empty list means nothing to flag. */
export function datasetWarnings(stat, preview) {
  const w = [];
  if (stat.kind === 'missing') return ['This path does not exist.'];
  if (stat.files === 0) w.push('There are no files here.');
  if (stat.truncated) w.push('The folder is large: the file count and size are lower bounds.');
  if (stat.kind === 'file' && stat.bytes === 0) w.push('The file is empty.');
  if (preview?.binary) w.push('This looks like a binary file, not text.');
  if (stat.bytes > 0 && stat.bytes < 1000) w.push('Very little text: a model trained on it will mostly memorise it.');
  return w;
}
