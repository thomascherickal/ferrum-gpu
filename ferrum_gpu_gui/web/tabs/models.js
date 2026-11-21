// The Models tab: list the GGUF files in a folder, look inside one (header, metadata, tensors), and
// choose it as the model the other tabs use. Everything is read on the server, from the start of the file.
import { api, withQuery } from '../shell/api.js';
import { store } from '../shell/state.js';
import { summaryLine, kindsText, tokenizerLine } from './models_view.js';
import { renderTensors, renderMeta, renderModelList } from './models_dom.js';

export function mountModels(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  let current = null;   // the inspected model's path
  let info = null;

  function showTensors() { renderTensors(doc, $('md-tensors'), info, $('md-filter').value); }

  async function inspect(path) {
    const summary = $('md-summary');
    summary.className = 'muted';
    summary.textContent = 'reading…';
    $('md-use').disabled = true;
    try {
      info = await api(withQuery('/api/gguf/inspect', { path }), {}, env.fetch);
    } catch (e) {
      info = null;
      summary.textContent = e.message;
      summary.className = 'bad';
      $('md-kinds').textContent = '';
      $('md-tokenizer').textContent = '';
      $('md-meta').textContent = '';
      showTensors();
      return;
    }
    current = path;
    summary.textContent = summaryLine(info);
    $('md-kinds').textContent = kindsText(info.kinds);
    $('md-tokenizer').textContent = tokenizerLine(info);
    renderMeta(doc, $('md-meta'), info);
    $('md-more').textContent = info.truncated ? `The model has ${info.n_tensors} tensors; the first ${info.tensors.length} are listed.` : '';
    showTensors();
    $('md-use').disabled = false;
  }

  async function scan() {
    const list = $('md-list');
    list.textContent = '';
    const note = $('md-scan-note');
    note.className = 'muted';
    note.textContent = 'scanning…';
    try {
      const r = await api(withQuery('/api/gguf/models', { dir: $('md-dir').value.trim() || undefined }), {}, env.fetch);
      note.textContent = `${r.files.length} model${r.files.length === 1 ? '' : 's'} under ${r.dir}${r.truncated ? ' (the list was cut)' : ''}`;
      renderModelList(doc, list, r.files, (f) => { $('md-file').value = f.path; inspect(f.path); });
    } catch (e) {
      note.textContent = e.message;
      note.className = 'bad';
    }
  }

  $('md-scan').addEventListener('click', scan);
  $('md-inspect').addEventListener('click', () => {
    const p = $('md-file').value.trim();
    if (p) inspect(p);
    else { $('md-summary').textContent = 'Choose a model file first.'; $('md-summary').className = 'bad'; }
  });
  $('md-filter').addEventListener('input', showTensors);
  $('md-use').addEventListener('click', () => {
    if (current) {
      store.set('modelPath', current);
      $('md-summary').textContent += ' · chosen as the model for Fine-tune, Evaluate and Generate';
    }
  });
  $('md-use').disabled = true;
  store.on('modelPath', (p) => { if (!$('md-file').value) $('md-file').value = p; });
  return { scan, inspect };
}
