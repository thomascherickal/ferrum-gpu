// The Train tab: train a GPT-2 style model (BPE tokenizer, tied head, dropout, quantisation-aware
// training, validation with early stopping) as a job, with a live loss chart.
import { mountJobPanel } from '../jobs/panel.js';
import { trainArgs } from './train_args.js';
import { mountLossChart, collect } from '../charts/livechart.js';
import { store } from '../shell/state.js';
import { api, humanTime } from '../shell/api.js';

export { collect };

/** The status sentence of a running training: speed and time left. */
export function runSummary(step) {
  if (!step) return '';
  return `step ${step.step}/${step.total} · ${Math.round(step.tokens_per_s)} tokens/s · ${humanTime(step.eta)} left`;
}

/** The adapters this machine reports, as `<option>`s for the device choice of any tab. */
export async function fillAdapters(select, env = {}) {
  try {
    const info = await api('/api/info', {}, env.fetch);
    select.add(new Option('automatic', ''));
    for (const a of info.adapters) select.add(new Option(`${a.name} (${a.kind})`, a.name));
  } catch { /* the Device tab shows the error */ }
}

export function mountGtrain(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('gt-job'), doc, env);
  const chart = mountLossChart(doc, $('gt-chart'));
  const stats = $('gt-stats');

  const form = () => ({
    data: $('gt-data').value, out: $('gt-out').value, preset: $('gt-preset').value,
    lengthKind: $('gt-lengthkind').value, length: $('gt-length').value,
    batch: $('gt-batch').value, context: $('gt-context').value, lr: $('gt-lr').value, seed: $('gt-seed').value,
    vocab: $('gt-vocab').value, tokenizer: $('gt-tokenizer').value,
    weightDecay: $('gt-wd').value, clip: $('gt-clip').value, cosine: $('gt-cosine').checked, warmup: $('gt-warmup').value,
    tie: $('gt-tie').checked, dropout: $('gt-dropout').value, qat: $('gt-qat').checked, quant: $('gt-quant').value,
    val: $('gt-val').value, patience: $('gt-patience').value, maxSeconds: $('gt-maxsec').value,
    checkpoint: $('gt-checkpoint').value, checkpointEvery: $('gt-ckevery').value, resume: $('gt-resume').value,
    sample: $('gt-sample').checked, metricsCsv: $('gt-metrics').value,
    embed: $('gt-embed').value, heads: $('gt-heads').value, blocks: $('gt-blocks').value, hidden: $('gt-hidden').value, stride: $('gt-stride').value,
    device: $('gt-device').value, adapter: $('gt-adapter').value, gpuMem: $('gt-gpumem').value, threads: $('gt-threads').value,
  });

  $('gt-start').addEventListener('click', async () => {
    const r = trainArgs(form());
    if (r.error) {
      panel.elements.status.textContent = r.error;
      panel.elements.status.className = 'bad';
      return;
    }
    chart.reset();
    stats.textContent = '';
    try {
      await panel.run('train', r.args, {
        onLines: chart.add,
        onState(p) { stats.textContent = runSummary(p.step); },
        onDone(p) {
          if (p.state === 'done' || p.state === 'stopped') store.set('modelPath', $('gt-out').value.trim());
        },
      });
    } catch { /* the panel shows the reason */ }
  });

  // a dataset chosen in the Datasets tab, or a token file made in Prepare, is the default training data
  store.on('dataPath', (p) => { if (!$('gt-data').value) $('gt-data').value = p; });
  store.on('tokenFile', (p) => { $('gt-data').value = p; });
  fillAdapters($('gt-adapter'), env);
  return { panel, chart };
}
