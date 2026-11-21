// The first-page generation panel: the same prompt through the server or this browser's WebGPU.
import { post } from '/web/shell/api.js';
import { stopStrings, cutAtStop } from '/web/stop.js';
import { $ } from './dom.js';

async function generate() {
  $('gen-msg').textContent = ''; $('gen-msg').className = 'muted';
  const req = {
    prompt: $('prompt').value, max_new: Number($('max_new').value),
    temperature: Number($('temperature').value), top_k: Number($('top_k').value), seed: 1,
    stop: stopStrings($('stop').value),
  };
  try {
    if ($('runner').value === 'browser') {
      if (!window.ferrumBrowser) throw new Error('the in-browser runtime is not loaded');
      const t0 = performance.now();
      const text = cutAtStop(await window.ferrumBrowser.generate(req), req.stop);
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

export function mountGenerate() {
  $('generate').addEventListener('click', generate);
  // The "this browser" runner loads the WebGPU runtime only when it is used.
  window.ferrumBrowser = { generate: async (req) => (await import('/web/browser.js')).generate(req) };
}
