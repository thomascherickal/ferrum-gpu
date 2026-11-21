// The scripted `?autotest=ui` run: drives the real first-page controls, then reports.
import { api } from '/web/shell/api.js';
import { showTab } from '/web/studio.js';
import { $ } from './dom.js';
import { refreshInfo } from './device.js';
import { until, reportSelfTest } from './report.js';

export async function uiAutotest() {
  const steps = [];
  try {
    await refreshInfo();
    showTab($('tab-model')); $('preset').value = 'tiny'; $('model-form').requestSubmit();
    await until(() => $('model-msg').textContent.startsWith('Created')); steps.push('model');
    showTab($('tab-train'));
    $('text').value = 'the quick brown fox jumps over the lazy dog. '.repeat(10);
    $('steps').value = 40; $('batch').value = 4; $('context').value = 24; $('val').value = 20; $('train-start').click();
    await until(() => /^done/.test($('train-status').textContent)); steps.push('train');
    const s = await api('/api/train/status');
    const improved = s.losses.length === 40 && s.losses[39] < 0.7 * s.losses[0];
    // the validation share was held out: its loss was measured before and after, and fell
    const validated = s.val_before != null && s.val_after != null && s.val_after < s.val_before
      && /validation/.test($('train-status').textContent);
    showTab($('tab-generate')); $('prompt').value = 'the quick brown '; $('max_new').value = 12; $('generate').click();
    await until(() => $('output').textContent.length > 16); steps.push('generate');
    const out = $('output').textContent;
    // a stop string ends the text before its first match
    $('stop').value = ' '; $('output').textContent = ''; $('generate').click();
    await until(() => $('output').textContent.length > 0); steps.push('stop');
    const stopped = $('output').textContent === 'the quick brown fox';
    $('stop').value = '';
    // the same prompt through the in-browser WebGPU runner must give the server's greedy text
    let browser = { skipped: true };
    if (navigator.gpu && await navigator.gpu.requestAdapter()) {
      $('runner').value = 'browser'; $('output').textContent = ''; $('generate').click();
      await until(() => $('output').textContent.length > 16 || $('gen-msg').className === 'bad'); steps.push('browser');
      browser = { text: $('output').textContent, error: $('gen-msg').className === 'bad' ? $('gen-msg').textContent : null, note: $('gen-msg').textContent };
      browser.matches = browser.text === out;
    }
    const browserOk = browser.skipped || browser.matches === true;
    await reportSelfTest({ kind: 'ui', pass: improved && validated && stopped && out.startsWith('the quick brown ') && out.includes('fox') && browserOk, steps, first: s.losses[0], last: s.losses[39], validated, stopped, output: out, browser, device: $('device-badge').textContent });
  } catch (err) { await reportSelfTest({ kind: 'ui', pass: false, steps, error: String(err) }); }
}
