// The Export tab: re-write a model in another precision (`export-gguf`), register it with Ollama
// (`export-ollama`) or import it into LM Studio (`export-lmstudio`). Each is a job; the page says
// which of the two tools are installed.
import { api } from '../shell/api.js';
import { mountJobPanel } from '../jobs/panel.js';
import { exportGgufArgs, exportOllamaArgs, exportLmstudioArgs, toolText } from './runtime_args.js';
import { store } from '../shell/state.js';

export function mountExport(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const gguf = mountJobPanel($('exg-job'), doc, env);
  const ollama = mountJobPanel($('exo-job'), doc, env);
  const lms = mountJobPanel($('exl-job'), doc, env);

  function launch(panel, cmd, built) {
    if (built.error) {
      panel.elements.status.textContent = built.error;
      panel.elements.status.className = 'bad';
      return;
    }
    panel.run(cmd, built.args).catch(() => { /* the panel shows the reason */ });
  }
  const model = () => $('ex-model').value;

  $('exg-run').addEventListener('click', () => launch(gguf, 'export-gguf', exportGgufArgs({
    model: model(), out: $('exg-out').value, quant: $('exg-quant').value,
  })));
  $('exo-run').addEventListener('click', () => launch(ollama, 'export-ollama', exportOllamaArgs({
    model: model(), name: $('exo-name').value, modelfile: $('exo-modelfile').value, bin: $('exo-bin').value,
  })));
  $('exl-run').addEventListener('click', () => launch(lms, 'export-lmstudio', exportLmstudioArgs({
    model: model(), repo: $('exl-repo').value, bin: $('exl-bin').value,
  })));

  async function tools() {
    const note = $('ex-tools');
    try {
      const r = await api('/api/tools', {}, env.fetch);
      note.textContent = r.tools.map(toolText).join(' · ');
    } catch (e) {
      note.textContent = e.message;
    }
  }
  tools();
  store.on('modelPath', (p) => { if (!$('ex-model').value) $('ex-model').value = p; });
  return { tools };
}
