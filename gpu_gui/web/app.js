// ferrum-gpu GUI: talks to the local server's JSON API. No external requests.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const type = r.headers.get('content-type') || '';
  const data = type.includes('json') ? await r.json() : await r.arrayBuffer();
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

// ---- tabs
const tabs = [...document.querySelectorAll('[role="tab"]')];
function showTab(tab) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute('aria-selected', on);
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute('aria-controls')).hidden = !on;
  }
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => showTab(t));
  t.addEventListener('keydown', (e) => {
    const n = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!n) return;
    const next = tabs[(i + n + tabs.length) % tabs.length];
    next.focus();
    showTab(next);
  });
});
showTab(tabs[0]);

// ---- device + model panels
let info = null;
async function refreshInfo() {
  info = await api('/api/info');
  const d = info.device;
  $('device-badge').textContent = `${d.active.toUpperCase()} · ${d.name}`;
  $('device-note').textContent = d.note || `Requested: ${d.requested}. Running on: ${d.active}.`;
  $('adapters').innerHTML = info.adapters.length
    ? info.adapters.map((a) => `<tr><td>${esc(a.name)}</td><td>${esc(a.backend)}</td><td>${esc(a.kind)}</td><td>${a.max_binding_mib} MiB</td></tr>`).join('')
    : '<tr><td colspan="4" class="muted">No GPU adapters found; the CPU path is used.</td></tr>';
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

$('model-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await post('/api/model', { preset: $('preset').value, seed: Number($('seed').value) });
    $('model-msg').textContent = 'Created a new model.';
    $('model-msg').className = 'good';
    await refreshInfo();
  } catch (err) { $('model-msg').textContent = err.message; $('model-msg').className = 'bad'; }
});
$('quant').addEventListener('change', () => { $('download').href = `/api/model.gguf?quant=${$('quant').value}`; });
$('upload').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    await api('/api/model.gguf', { method: 'PUT', headers: { 'content-type': 'application/octet-stream' }, body: await f.arrayBuffer() });
    $('model-msg').textContent = `Loaded ${f.name}.`;
    $('model-msg').className = 'good';
    await refreshInfo();
  } catch (err) { $('model-msg').textContent = err.message; $('model-msg').className = 'bad'; }
  e.target.value = '';
});

// ---- training
function drawChart(losses) {
  const c = $('chart'), ratio = window.devicePixelRatio || 1;
  const w = c.clientWidth || 600, h = Math.round(w * 220 / 600);
  c.width = w * ratio; c.height = h * ratio;
  const g = c.getContext('2d');
  g.scale(ratio, ratio);
  const css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = css.getPropertyValue('--line'); g.fillStyle = css.getPropertyValue('--muted');
  g.font = '12px system-ui'; g.lineWidth = 1;
  g.strokeRect(40.5, 8.5, w - 52, h - 30);
  if (losses.length < 2) { g.fillText('loss will appear here', 50, h / 2); return; }
  const max = Math.max(...losses), min = Math.min(...losses), span = max - min || 1;
  g.fillText(max.toFixed(2), 4, 20); g.fillText(min.toFixed(2), 4, h - 22);
  g.fillText('step ' + losses.length, w - 70, h - 6);
  g.strokeStyle = css.getPropertyValue('--accent'); g.lineWidth = 2; g.beginPath();
  losses.forEach((l, i) => {
    const x = 41 + (i / (losses.length - 1)) * (w - 54), y = 9 + (1 - (l - min) / span) * (h - 32);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}
let poll = null;
async function pollTrain() {
  try {
    const s = await api('/api/train/status');
    $('progress').max = s.total || 1; $('progress').value = s.step;
    const last = s.losses.length ? s.losses[s.losses.length - 1].toFixed(3) : '–';
    $('train-status').textContent = s.state === 'error' ? `error: ${s.error}` : `${s.state} · step ${s.step}/${s.total || '?'} · loss ${last}`;
    $('train-status').className = s.state === 'error' ? 'bad' : s.state === 'done' ? 'good' : 'muted';
    drawChart(s.losses);
    const running = s.state === 'running';
    $('train-start').disabled = running; $('train-stop').disabled = !running;
    if (!running && poll) { clearInterval(poll); poll = null; refreshInfo(); }
  } catch (err) { $('train-status').textContent = err.message; }
}
$('train-start').addEventListener('click', async () => {
  try {
    await post('/api/train', {
      text: $('text').value, steps: Number($('steps').value), lr: Number($('lr').value),
      batch: Number($('batch').value), context: Number($('context').value),
    });
    if (!poll) poll = setInterval(pollTrain, 400);
    pollTrain();
  } catch (err) { $('train-status').textContent = err.message; $('train-status').className = 'bad'; }
});
$('train-stop').addEventListener('click', () => post('/api/train/stop', {}).catch(() => {}));

// ---- generation
async function generate() {
  $('gen-msg').textContent = ''; $('gen-msg').className = 'muted';
  const req = {
    prompt: $('prompt').value, max_new: Number($('max_new').value),
    temperature: Number($('temperature').value), top_k: Number($('top_k').value), seed: 1,
  };
  try {
    if ($('runner').value === 'browser') {
      if (!window.ferrumBrowser) throw new Error('the in-browser runtime is not loaded');
      const t0 = performance.now();
      const text = await window.ferrumBrowser.generate(req);
      $('output').textContent = req.prompt + text;
      $('gen-msg').textContent = `in this browser (WebGPU) · ${(performance.now() - t0).toFixed(0)} ms`;
    } else {
      const t0 = performance.now();
      const r = await post('/api/generate', req);
      $('output').textContent = req.prompt + r.text;
      $('gen-msg').textContent = `on the server ${r.device.toUpperCase()} · ${(performance.now() - t0).toFixed(0)} ms`;
    }
  } catch (err) { $('gen-msg').textContent = err.message; $('gen-msg').className = 'bad'; }
}
$('generate').addEventListener('click', generate);

// ---- WebGPU availability (Phase 5b fills in the runtime)
async function checkWebGpu() {
  const el = $('webgpu-status');
  if (!navigator.gpu) { el.textContent = 'Not available in this browser.'; return; }
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) { el.textContent = 'navigator.gpu exists but no adapter was granted.'; return; }
    const i = a.info || {};
    el.textContent = `Available: ${[i.vendor, i.architecture, i.description].filter(Boolean).join(' · ') || 'adapter granted'}${a.isFallbackAdapter ? ' (software fallback)' : ''}.`;
  } catch (err) { el.textContent = `Error: ${err.message}`; }
}

refreshInfo().then(checkWebGpu).catch((err) => { $('device-badge').textContent = err.message; });
pollTrain();
window.addEventListener('resize', pollTrain);
