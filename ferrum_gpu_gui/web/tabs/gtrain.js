// The Train tab: train a GPT-2 style model (BPE tokenizer, tied head, dropout, quantisation-aware
// training, validation with early stopping) as a job, with a live loss chart.
import { mountJobPanel } from '../jobs/panel.js';
import { trainArgs } from './train_args.js';
import { draw } from '../charts/loss.js';
import { store } from '../shell/state.js';
import { api, humanTime } from '../shell/api.js';

/** Fold output lines into the chart series: steps become `[step, loss]`, validations `[lastStep, loss]`. */
export function collect(lines, series) {
  for (const l of lines) {
    const ev = l.ev;
    if (!ev) continue;
    if (ev.kind === 'step') {
      series.train.push([ev.step, ev.loss]);
      series.lastStep = ev.step;
    } else if (ev.kind === 'val') {
      series.val.push([series.lastStep, ev.loss]);
    }
  }
  return series;
}

/** The status sentence of a running training: speed and time left. */
export function runSummary(step) {
  if (!step) return '';
  return `step ${step.step}/${step.total} · ${Math.round(step.tokens_per_s)} tokens/s · ${humanTime(step.eta)} left`;
}

export function mountGtrain(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('gt-job'), doc, env);
  const series = { train: [], val: [], lastStep: 0 };
  const canvas = $('gt-chart');
  const stats = $('gt-stats');

  function css() {
    const s = doc.defaultView.getComputedStyle(doc.documentElement);
    const v = (k) => s.getPropertyValue(k).trim();
    return { line: v('--line'), muted: v('--muted'), accent: v('--accent'), good: v('--good') };
  }
  function paint() {
    const ratio = doc.defaultView.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    const h = Math.round((w * 220) / 600);
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    const g = canvas.getContext('2d');
    g.scale(ratio, ratio);
    draw(g, w, h, series.train, series.val, css());
  }
  paint();
  doc.defaultView.addEventListener('resize', paint);

  const form = () => ({
    data: $('gt-data').value, out: $('gt-out').value, preset: $('gt-preset').value,
    lengthKind: $('gt-lengthkind').value, length: $('gt-length').value,
    batch: $('gt-batch').value, context: $('gt-context').value, lr: $('gt-lr').value, seed: $('gt-seed').value,
    vocab: $('gt-vocab').value, tokenizer: $('gt-tokenizer').value,
    weightDecay: $('gt-wd').value, clip: $('gt-clip').value, cosine: $('gt-cosine').checked, warmup: $('gt-warmup').value,
    tie: $('gt-tie').checked, dropout: $('gt-dropout').value, qat: $('gt-qat').checked, quant: $('gt-quant').value,
    val: $('gt-val').value, patience: $('gt-patience').value, maxSeconds: $('gt-maxsec').value,
    checkpoint: $('gt-checkpoint').value, checkpointEvery: $('gt-ckevery').value, resume: $('gt-resume').value,
    sample: $('gt-sample').checked,
    device: $('gt-device').value, adapter: $('gt-adapter').value, gpuMem: $('gt-gpumem').value, threads: $('gt-threads').value,
  });

  $('gt-start').addEventListener('click', async () => {
    const r = trainArgs(form());
    if (r.error) {
      panel.elements.status.textContent = r.error;
      panel.elements.status.className = 'bad';
      return;
    }
    series.train.length = 0;
    series.val.length = 0;
    series.lastStep = 0;
    stats.textContent = '';
    paint();
    try {
      await panel.run('train', r.args, {
        onLines(lines) { collect(lines, series); paint(); },
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

  // the adapters this machine reports, for the device choice
  async function fillAdapters() {
    try {
      const info = await api('/api/info', {}, env.fetch);
      const sel = $('gt-adapter');
      sel.add(new Option('automatic', ''));
      for (const a of info.adapters) sel.add(new Option(`${a.name} (${a.kind})`, a.name));
    } catch { /* the Device tab shows the error */ }
  }
  fillAdapters();
  return { panel, series, paint };
}
