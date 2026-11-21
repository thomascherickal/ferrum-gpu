// The device and model panels: the adapter table, the model summary, create / upload, and the
// WebGPU availability line.
import { esc, api, post } from '/web/shell/api.js';
import { $ } from './dom.js';

const adapterRow = (a) => `<tr><td>${esc(a.name)}</td><td>${esc(a.backend)}</td><td>${esc(a.kind)}</td><td>${a.max_binding_mib} MiB</td><td>${a.features.length ? esc(a.features.join(', ')) : '<span class="muted">none reported</span>'}</td><td title="${esc(a.memory_note)}">${a.memory_budget_mib == null ? '<span class="muted">unknown</span>' : a.memory_budget_mib + ' MiB'}</td></tr>`;

export async function refreshInfo() {
  const info = await api('/api/info');
  const d = info.device;
  $('device-badge').textContent = `${d.active.toUpperCase()} · ${d.name}`;
  $('device-note').textContent = d.note || `Requested: ${d.requested}. Running on: ${d.active}.`;
  $('adapters').innerHTML = info.adapters.length
    ? info.adapters.map(adapterRow).join('')
    : '<tr><td colspan="6" class="muted">No GPU adapters found; the CPU path is used.</td></tr>';
  const sel = $('preset');
  if (!sel.options.length) {
    for (const p of info.presets) sel.add(new Option(`${p.name} · ${(p.params / 1e3).toFixed(0)}K params · context ${p.ctx}`, p.name));
  }
  const m = info.model;
  $('model-summary').textContent = m
    ? `${m.preset || 'custom'} · ${(m.params / 1e3).toFixed(0)}K parameters · ${m.layers} layers · dim ${m.dim} · ${m.heads}/${m.kv_heads} heads · context ${m.ctx} · ${m.rope.toUpperCase()} RoPE${m.byte_tokens ? '' : ' · NOT a byte-level model (text in and out unavailable)'}`
    : 'No model yet. Create one, or load a GGUF.';
  return info;
}

const modelMsg = (text, cls) => { $('model-msg').textContent = text; $('model-msg').className = cls; };

export function mountModelPanel() {
  $('model-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      await post('/api/model', { preset: $('preset').value, seed: Number($('seed').value), tied: $('tied').checked });
      modelMsg('Created a new model.', 'good');
      await refreshInfo();
    } catch (err) { modelMsg(err.message, 'bad'); }
  });
  $('quant').addEventListener('change', () => { $('download').href = `/api/model.gguf?quant=${$('quant').value}`; });
  $('upload').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      await api('/api/model.gguf', { method: 'PUT', headers: { 'content-type': 'application/octet-stream' }, body: await f.arrayBuffer() });
      modelMsg(`Loaded ${f.name}.`, 'good');
      await refreshInfo();
    } catch (err) { modelMsg(err.message, 'bad'); }
    e.target.value = '';
  });
}

export async function checkWebGpu() {
  const el = $('webgpu-status');
  if (!navigator.gpu) { el.textContent = 'Not available in this browser.'; return; }
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) { el.textContent = 'navigator.gpu exists but no adapter was granted.'; return; }
    const i = a.info || {};
    el.textContent = `Available: ${[i.vendor, i.architecture, i.description].filter(Boolean).join(' · ') || 'adapter granted'}${a.isFallbackAdapter ? ' (software fallback)' : ''}.`;
  } catch (err) { el.textContent = `Error: ${err.message}`; }
}
