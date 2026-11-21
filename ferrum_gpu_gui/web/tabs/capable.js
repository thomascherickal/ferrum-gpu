// The Capable tab: how large a model a given amount of memory can hold, per precision and for a full
// fine-tune; the decode-speed ceiling for a model size and a memory bandwidth; and what this machine's
// own adapters can hold. All figures are estimates (hardware-caps.md has the formulas and the caveats).
import { api, withQuery, humanBytes } from '../shell/api.js';
import { capableQuery, tpsText } from './capable_view.js';

export function mountCapable(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  const cell = (tag, text) => {
    const e = doc.createElement(tag);
    e.textContent = text;
    return e;
  };

  async function ask() {
    const note = $('cap-note');
    const q = capableQuery({
      memory: $('cap-memory').value, unified: $('cap-unified').checked,
      bandwidth: $('cap-bandwidth').value, model: $('cap-model').value,
    });
    if (q.error) {
      note.textContent = q.error;
      note.className = 'bad';
      return;
    }
    try {
      const r = await api(withQuery('/api/capable', q.params), {}, env.fetch);
      note.className = 'muted';
      note.textContent = `${r.row.memory_gb} GB, ${r.row.usable_gb.toFixed(1)} GB usable. Estimates: the memory formula is within about -24% to +37% of measured plans.`;
      const body = $('cap-result');
      body.textContent = '';
      for (const i of r.row.infer) {
        const tr = doc.createElement('tr');
        tr.append(cell('td', `Inference, ${i.precision}`), cell('td', i.text));
        body.append(tr);
      }
      const tr = doc.createElement('tr');
      tr.append(cell('td', 'Full fine-tune (f32, AdamW)'), cell('td', r.row.train.text));
      body.append(tr);
      $('cap-ceiling').textContent = r.ceiling
        ? `Decode speed ceiling for ${$('cap-model').value} billion weights at ${$('cap-bandwidth').value} GB/s: ${r.ceiling.map((c) => `${c.precision} ${tpsText(c.tokens_per_s)} tokens/s`).join(' · ')}. A ceiling, not what ferrum-gpu reaches.`
        : '';
    } catch (e) {
      note.textContent = e.message;
      note.className = 'bad';
    }
  }
  $('cap-run').addEventListener('click', ask);

  async function machine() {
    const body = $('cap-machine');
    try {
      const s = await api('/api/system', {}, env.fetch);
      $('cap-system').textContent = `${s.os} ${s.arch} · ${s.cpu_threads} CPU threads · ${s.ram_total ? humanBytes(s.ram_total) + ' RAM' : 'RAM unknown'}${s.ram_available ? ` (${humanBytes(s.ram_available)} available)` : ''}`;
      body.textContent = '';
      for (const a of s.adapters) {
        const tr = doc.createElement('tr');
        const f = a.fits;
        tr.append(
          cell('td', `${a.name} (${a.kind})`),
          cell('td', a.budget_mib === null ? 'unknown' : `${a.budget_mib} MiB`),
          cell('td', f ? f.infer.map((i) => `${i.precision} ${i.text}`).join(' · ') : '–'),
          cell('td', f ? f.train : '–'),
        );
        tr.title = a.note;
        body.append(tr);
      }
      if (!s.adapters.length) {
        const tr = doc.createElement('tr');
        const td = cell('td', 'No GPU adapters found; the CPU path is used.');
        td.colSpan = 4;
        tr.append(td);
        body.append(tr);
      }
    } catch (e) {
      $('cap-system').textContent = e.message;
    }
  }
  machine();
  return { ask, machine };
}
