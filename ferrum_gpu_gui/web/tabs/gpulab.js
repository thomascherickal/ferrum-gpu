// The GPU lab tab: pick an adapter, run `ferrum-gpu lab` (GPU against the CPU reference, speed per
// precision) as a job, make the JSON report to send back, and watch each adapter's memory.
import { api } from '../shell/api.js';
import { mountJobPanel } from '../jobs/panel.js';
import { labArgs, reportFromLines, reportSummary, meterView } from './lab_args.js';

export function mountGpuLab(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const panel = mountJobPanel($('gl-job'), doc, env);
  let lines = [];

  function form(json) {
    return {
      adapter: $('gl-adapter').value, size: $('gl-size').value, reps: $('gl-reps').value,
      parityOnly: $('gl-parity').checked, json,
    };
  }

  function launch(json) {
    const built = labArgs(form(json));
    if (built.error) {
      panel.elements.status.textContent = built.error;
      panel.elements.status.className = 'bad';
      return;
    }
    lines = [];
    $('gl-report').value = '';
    $('gl-summary').textContent = '';
    panel.run('lab', built.args, {
      onLines: (l) => lines.push(...l),
      onDone: () => {
        const r = json ? reportFromLines(lines) : null;
        if (r) {
          $('gl-report').value = JSON.stringify(r, null, 2);
          $('gl-summary').textContent = reportSummary(r);
        }
      },
    }).catch(() => { /* the panel shows the reason */ });
  }
  $('gl-run').addEventListener('click', () => launch(false));
  $('gl-json').addEventListener('click', () => launch(true));

  async function adapters() {
    try {
      const info = await api('/api/info', {}, env.fetch);
      const sel = $('gl-adapter');
      for (const a of info.adapters) {
        const o = doc.createElement('option');
        o.value = a.name;
        o.textContent = `${a.name} (${a.kind}, ${a.backend})`;
        sel.append(o);
      }
    } catch { /* the Device tab shows the reason */ }
  }

  async function meters() {
    const body = $('gl-meters');
    try {
      const s = await api('/api/system', {}, env.fetch);
      body.textContent = '';
      for (const a of s.adapters) {
        const v = meterView(a.meter);
        const tr = doc.createElement('tr');
        const name = doc.createElement('td');
        name.textContent = `${a.name} (${a.kind})`;
        const bar = doc.createElement('td');
        const m = doc.createElement('meter');
        m.min = 0;
        m.max = v.max || 1;
        m.value = v.value;
        m.setAttribute('aria-label', `Free memory of ${a.name}`);
        const t = doc.createElement('span');
        t.textContent = ` ${v.text}`;
        bar.append(m, t);
        const budget = doc.createElement('td');
        budget.textContent = a.budget_mib === null ? 'unknown' : `${a.budget_mib} MiB`;
        budget.title = a.note;
        tr.append(name, bar, budget);
        body.append(tr);
      }
      if (!s.adapters.length) {
        const tr = doc.createElement('tr');
        const td = doc.createElement('td');
        td.colSpan = 3;
        td.textContent = 'No GPU adapters found.';
        tr.append(td);
        body.append(tr);
      }
    } catch (e) {
      body.textContent = e.message;
    }
  }
  $('gl-refresh').addEventListener('click', meters);
  adapters();
  meters();
  return { meters, adapters };
}
