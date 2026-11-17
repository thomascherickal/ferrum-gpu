// The Generate tab: continue a prompt with a GGUF model (`generate`), as a job; the generated text is
// shown on its own, apart from the commands' status lines.
import { mountJobPanel } from '../jobs/panel.js';
import { generateArgs, outputText } from './flow_args.js';
import { store } from '../shell/state.js';
import { fillAdapters } from './gtrain.js';

export function mountGenerate(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('gg-job'), doc, env);
  const out = $('gg-output');
  let all = [];

  $('gg-start').addEventListener('click', async () => {
    const r = generateArgs({
      model: $('gg-model').value, prompt: $('gg-prompt').value, maxNew: $('gg-maxnew').value,
      temperature: $('gg-temp').value, topK: $('gg-topk').value, topP: $('gg-topp').value,
      repetition: $('gg-rep').value, stop: $('gg-stop').value, seed: $('gg-seed').value,
      device: $('gg-device').value, adapter: $('gg-adapter').value,
    });
    if (r.error) {
      panel.elements.status.textContent = r.error;
      panel.elements.status.className = 'bad';
      return;
    }
    all = [];
    out.textContent = '';
    try {
      await panel.run('generate', r.args, {
        onLines(lines) { all = all.concat(lines); out.textContent = outputText(all); },
      });
    } catch { /* the panel shows the reason */ }
  });

  store.on('modelPath', (p) => { if (!$('gg-model').value) $('gg-model').value = p; });
  fillAdapters($('gg-adapter'), env);
  return { panel };
}
