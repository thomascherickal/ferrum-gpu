// The Serve tab: run the JSON server for a model inside this application (the same server as
// `ferrum-gpu-server`), see its address and status lines, and stop it.
import { api, post } from '../shell/api.js';
import { serveArgs } from './runtime_args.js';
import { fillAdapters } from './gtrain.js';
import { store } from '../shell/state.js';
import { humanTime } from '../shell/api.js';

/** The sentence for the server's state. */
export function statusSentence(s) {
  if (!s.running) return 'not running';
  const auth = s.has_token ? ' · bearer token required' : '';
  return `running at http://${s.addr}/ · up ${humanTime(s.seconds)}${auth}`;
}

export function mountServe(doc = document, env = {}) {
  const $ = (id) => doc.getElementById(id);
  let timer = null;

  function show(s) {
    $('sv-status').textContent = statusSentence(s);
    $('sv-status').className = s.running ? 'good' : 'muted';
    $('sv-log').textContent = s.lines.join('\n');
    $('sv-start').disabled = s.running;
    $('sv-stop').disabled = !s.running;
    if (s.running && !timer) timer = setInterval(refresh, 2000);
    if (!s.running && timer) { clearInterval(timer); timer = null; }
  }

  async function refresh() {
    try { show(await api('/api/serve/status', {}, env.fetch)); } catch (e) { $('sv-status').textContent = e.message; }
  }

  $('sv-start').addEventListener('click', async () => {
    const r = serveArgs({
      model: $('sv-model').value, host: $('sv-host').value, port: $('sv-port').value, token: $('sv-token').value,
      device: $('sv-device').value, adapter: $('sv-adapter').value, gpuMem: $('sv-gpumem').value, threads: $('sv-threads').value,
    });
    if (r.error) {
      $('sv-status').textContent = r.error;
      $('sv-status').className = 'bad';
      return;
    }
    $('sv-start').disabled = true;
    try {
      show(await post('/api/serve/start', { args: r.args }, env.fetch));
    } catch (e) {
      $('sv-status').textContent = e.message;
      $('sv-status').className = 'bad';
      await refresh();
      $('sv-status').textContent = e.message;
      $('sv-status').className = 'bad';
      $('sv-start').disabled = false;
    }
  });
  $('sv-stop').addEventListener('click', async () => {
    try { await post('/api/serve/stop', {}, env.fetch); } catch { /* shown by the refresh */ }
    await refresh();
  });

  store.on('modelPath', (p) => { if (!$('sv-model').value) $('sv-model').value = p; });
  fillAdapters($('sv-adapter'), env);
  $('sv-stop').disabled = true;
  refresh();
  return { refresh };
}
