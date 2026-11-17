// The Datasets tab: look at a text file or folder before training on it, and hand it to the Train tab.
import { api, withQuery, humanBytes } from '../shell/api.js';
import { store } from '../shell/state.js';
import { datasetWarnings, describeStat } from './args.js';

export function mountDatasets(doc = document, f) {
  const $ = (id) => doc.getElementById(id);
  const path = $('ds-path');
  const summary = $('ds-summary');
  const warnings = $('ds-warnings');
  const preview = $('ds-preview');
  const use = $('ds-use');
  use.disabled = true;

  async function inspect() {
    const p = path.value.trim();
    summary.className = 'muted';
    warnings.textContent = '';
    preview.textContent = '';
    use.disabled = true;
    if (!p) { summary.textContent = 'Choose a file or folder first.'; return; }
    try {
      const stat = await api(withQuery('/api/fs/stat', { path: p }), {}, f);
      let pv = null;
      if (stat.kind === 'file') pv = await api(withQuery('/api/fs/preview', { path: p, bytes: 2000 }), {}, f);
      summary.textContent = describeStat(stat, pv, humanBytes);
      for (const w of datasetWarnings(stat, pv)) {
        const li = doc.createElement('li');
        li.textContent = w;
        warnings.append(li);
      }
      if (pv && !pv.binary) preview.textContent = pv.text + (pv.truncated ? '\n…' : '');
      use.disabled = stat.kind === 'missing' || stat.files === 0;
    } catch (e) {
      summary.textContent = e.message;
      summary.className = 'bad';
    }
  }

  $('ds-inspect').addEventListener('click', inspect);
  path.addEventListener('change', inspect);
  use.addEventListener('click', () => {
    store.set('dataPath', path.value.trim());
    summary.textContent += ' · chosen as the training data';
  });
  return { inspect };
}
