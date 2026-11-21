// Node tests of the Prepare and Tokenize argument builders and the dataset warnings.
// usage: node args.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { prepareArgs, tokenizeArgs, datasetWarnings, describeStat } from '../web/tabs/args.js';

// ---- tabs/args.js: prepare
{
  const base = { inputs: 'a.txt\n\n  b/  \n', out: 'out' };
  eq(prepareArgs(base), { args: ['--input', 'a.txt', '--input', 'b/', '--out', 'out'] });
  eq(prepareArgs({ ...base, nfkc: true, lowercase: true, stripGutenberg: true, dedup: true, lossy: true,
    minChars: '5', maxChars: '100', seed: '7', valFrac: '0.1', testFrac: '0.2', docSep: '<|doc|>' }).args,
  ['--input', 'a.txt', '--input', 'b/', '--out', 'out', '--nfkc', '--lowercase', '--strip-gutenberg', '--dedup', '--lossy',
    '--min-chars', '5', '--max-chars', '100', '--seed', '7', '--val-frac', '0.1', '--test-frac', '0.2', '--doc-sep', '<|doc|>']);
  for (const [bad, needle] of [
    [{ inputs: '', out: 'o' }, 'at least one input'],
    [{ inputs: '  \n ', out: 'o' }, 'at least one input'],
    [{ inputs: 'a', out: ' ' }, 'output folder'],
    [{ ...base, minChars: 'x' }, 'Minimum characters'],
    [{ ...base, maxChars: '-1' }, 'Maximum characters'],
    [{ ...base, seed: '1.5' }, 'Seed'],
    [{ ...base, valFrac: '1' }, 'Validation share'],
    [{ ...base, testFrac: '-0.1' }, 'Test share'],
    [{ ...base, valFrac: 'abc' }, 'Validation share'],
    [{ ...base, valFrac: '0.6', testFrac: '0.5' }, 'together'],
    [{ ...base, valFrac: '0.97' }, 'together'],
    [{ ...base, minChars: '10', maxChars: '5' }, 'at least the minimum'],
  ]) {
    const r = prepareArgs(bad);
    ok(r.error && r.error.includes(needle), `${JSON.stringify(bad)} -> ${r.error}`);
  }
  eq(prepareArgs({ ...base, maxChars: '0', minChars: '3' }).args.includes('--max-chars'), true);   // zero = no limit is accepted
  eq(prepareArgs({ inputs: 'a', out: 'o', docSep: '' }).args, ['--input', 'a', '--out', 'o']);
}
// ---- tokenize
eq(tokenizeArgs({ data: 'd', tokenizer: 't.gguf', out: 'o.gguf' }), { args: ['--data', 'd', '--tokenizer', 't.gguf', '--out', 'o.gguf'] });
eq(tokenizeArgs({ data: ' d ', tokenizer: 't', out: 'o', docSep: 'SEP' }).args.slice(-2), ['--doc-sep', 'SEP']);
ok(/text file or folder/.test(tokenizeArgs({ data: '', tokenizer: 't', out: 'o' }).error));
ok(/tokenizer file/.test(tokenizeArgs({ data: 'd', tokenizer: '', out: 'o' }).error));
ok(/token file/.test(tokenizeArgs({ data: 'd', tokenizer: 't', out: '' }).error));
ok(/text file or folder/.test(tokenizeArgs({}).error));
// ---- dataset text
const hb = (n) => `${n}B`;
eq(describeStat({ kind: 'missing' }, null, hb), 'No such file or folder.');
eq(describeStat({ kind: 'dir', files: 1, bytes: 10, truncated: false }, null, hb), 'Folder · 1 file · 10B');
eq(describeStat({ kind: 'dir', files: 3, bytes: 30, truncated: true }, null, hb), 'Folder · at least 3 files · at least 30B');
eq(describeStat({ kind: 'file', bytes: 5 }, { lines: 1 }, hb), 'File · 5B · 1 line');
eq(describeStat({ kind: 'file', bytes: 5 }, { lines: 4 }, hb), 'File · 5B · 4 lines');
eq(describeStat({ kind: 'file', bytes: 5 }, null, hb), 'File · 5B');
eq(datasetWarnings({ kind: 'missing' }, null), ['This path does not exist.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 50000 }, { binary: false }), []);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 0 }, { binary: false }), ['The file is empty.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 500 }, { binary: false }), ['Very little text: a model trained on it will mostly memorise it.']);
eq(datasetWarnings({ kind: 'file', files: 1, bytes: 5000 }, { binary: true }), ['This looks like a binary file, not text.']);
eq(datasetWarnings({ kind: 'dir', files: 0, bytes: 0, truncated: false }, null), ['There are no files here.']);
eq(datasetWarnings({ kind: 'dir', files: 9, bytes: 99999, truncated: true }, null), ['The folder is large: the file count and size are lower bounds.']);

summary();
