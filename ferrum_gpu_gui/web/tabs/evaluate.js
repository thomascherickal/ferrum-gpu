// The Evaluate tab: the loss and perplexity of a text under a model (`eval`, always on the CPU).
import { mountJobPanel } from '../jobs/panel.js';
import { evalArgs } from './flow_args.js';
import { store } from '../shell/state.js';

/** The sentence for an eval result event: `{tokens, loss, perplexity, bits_per_byte?}`. */
export function resultText(ev) {
  if (!ev) return '';
  const bpb = Number.isFinite(ev.bits_per_byte) ? ` · ${ev.bits_per_byte.toFixed(4)} bits per byte` : '';
  return `perplexity ${ev.perplexity.toFixed(3)} · loss ${ev.loss.toFixed(4)} · over ${ev.tokens} tokens${bpb}`;
}

export function mountEvaluate(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('ev-job'), doc, env);
  const result = $('ev-result');

  $('ev-start').addEventListener('click', async () => {
    const r = evalArgs({
      model: $('ev-model').value, data: $('ev-data').value, ctx: $('ev-ctx').value,
      quant: $('ev-quant').value, json: $('ev-json').value,
    });
    if (r.error) {
      panel.elements.status.textContent = r.error;
      panel.elements.status.className = 'bad';
      return;
    }
    result.textContent = '';
    try {
      await panel.run('eval', r.args, {
        onState(p) { if (p.eval) result.textContent = resultText(p.eval); },
      });
    } catch { /* the panel shows the reason */ }
  });

  store.on('modelPath', (p) => { if (!$('ev-model').value) $('ev-model').value = p; });
  store.on('dataPath', (p) => { if (!$('ev-data').value) $('ev-data').value = p; });
  return { panel };
}
