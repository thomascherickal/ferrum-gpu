// The Prepare tab: clean and split raw text (`prepare`), and encode it with a tokenizer (`tokenize`).
import { mountJobPanel } from '../jobs/panel.js';
import { prepareArgs, tokenizeArgs } from './args.js';
import { store } from '../shell/state.js';

export function mountPrepare(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const prep = mountJobPanel($('pr-job'), doc, env);
  const tok = mountJobPanel($('tk-job'), doc, env);

  function fail(panel, msg) {
    panel.elements.status.textContent = msg;
    panel.elements.status.className = 'bad';
  }

  $('pr-add-btn').addEventListener('click', () => {
    const v = $('pr-add').value.trim();
    if (!v) return;
    const lines = $('pr-inputs').value.split('\n').map((s) => s.trim()).filter(Boolean);
    if (!lines.includes(v)) lines.push(v);
    $('pr-inputs').value = lines.join('\n');
    $('pr-add').value = '';
  });

  $('pr-run').addEventListener('click', async () => {
    const r = prepareArgs({
      inputs: $('pr-inputs').value, out: $('pr-out').value,
      nfkc: $('pr-nfkc').checked, lowercase: $('pr-lowercase').checked,
      stripGutenberg: $('pr-gutenberg').checked, dedup: $('pr-dedup').checked, lossy: $('pr-lossy').checked,
      minChars: $('pr-minchars').value, maxChars: $('pr-maxchars').value,
      valFrac: $('pr-valfrac').value, testFrac: $('pr-testfrac').value,
      seed: $('pr-seed').value, docSep: $('pr-docsep').value,
    });
    if (r.error) return fail(prep, r.error);
    try {
      await prep.run('prepare', r.args, {
        onDone(p) { if (p.state === 'done') store.set('preparedDir', $('pr-out').value.trim()); },
      });
    } catch { /* the panel shows the reason */ }
  });

  $('tk-run').addEventListener('click', async () => {
    const r = tokenizeArgs({
      data: $('tk-data').value, tokenizer: $('tk-tokenizer').value,
      out: $('tk-out').value, docSep: $('tk-docsep').value,
    });
    if (r.error) return fail(tok, r.error);
    try {
      await tok.run('tokenize', r.args, {
        onDone(p) { if (p.state === 'done') store.set('tokenFile', $('tk-out').value.trim()); },
      });
    } catch { /* the panel shows the reason */ }
  });

  // a dataset chosen in the Datasets tab is the default input of both forms
  store.on('dataPath', (p) => {
    if (!$('tk-data').value) $('tk-data').value = p;
    if (!$('pr-inputs').value.trim()) $('pr-inputs').value = p;
  });
  return { prep, tok };
}
