// The Fine-tune tab: continue training a GGUF model on new text (`finetune-gguf`), as a job, with a
// live loss chart. The model keeps its own tokenizer and shape.
import { mountJobPanel } from '../jobs/panel.js';
import { finetuneArgs } from './flow_args.js';
import { mountLossChart } from '../charts/livechart.js';
import { store } from '../shell/state.js';
import { runSummary, fillAdapters } from './gtrain.js';

export function mountFinetune(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('ft-job'), doc, env);
  const chart = mountLossChart(doc, $('ft-chart'));
  const stats = $('ft-stats');

  const form = () => ({
    model: $('ft-model').value, data: $('ft-data').value, out: $('ft-out').value,
    lengthKind: $('ft-lengthkind').value, length: $('ft-length').value,
    batch: $('ft-batch').value, lr: $('ft-lr').value,
    val: $('ft-val').value, patience: $('ft-patience').value, maxSeconds: $('ft-maxsec').value,
    qat: $('ft-qat').checked, quant: $('ft-quant').value,
    device: $('ft-device').value, adapter: $('ft-adapter').value, gpuMem: $('ft-gpumem').value,
  });

  $('ft-start').addEventListener('click', async () => {
    const r = finetuneArgs(form());
    if (r.error) {
      panel.elements.status.textContent = r.error;
      panel.elements.status.className = 'bad';
      return;
    }
    chart.reset();
    stats.textContent = '';
    try {
      await panel.run('finetune-gguf', r.args, {
        onLines: chart.add,
        onState(p) { stats.textContent = runSummary(p.step); },
        onDone(p) {
          if (p.state === 'done' || p.state === 'stopped') store.set('modelPath', $('ft-out').value.trim());
        },
      });
    } catch { /* the panel shows the reason */ }
  });

  // the model trained in the Train tab is the default model to fine-tune, and the dataset the default text
  store.on('modelPath', (p) => { if (!$('ft-model').value) $('ft-model').value = p; });
  store.on('dataPath', (p) => { if (!$('ft-data').value) $('ft-data').value = p; });
  fillAdapters($('ft-adapter'), env);
  return { panel, chart };
}
