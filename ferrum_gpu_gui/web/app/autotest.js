// Scripted runs for the headless-browser tests: report uncaught errors, pick the run named by
// `?autotest=`, and hand its result to the server.
import { showTab } from '/web/studio.js';
import { $ } from './dom.js';
import { uiAutotest } from './ui_autotest.js';
import { reportSelfTest } from './report.js';

const params = new URLSearchParams(location.search);

/** In a scripted run any uncaught error is reported to the server so the test can show it. */
export function reportUncaughtErrors() {
  if (!params.get('autotest')) return;
  const send = (m) => fetch('/api/selftest/result', { method: 'POST', body: JSON.stringify({ kind: 'error', pass: false, error: String(m) }) });
  window.addEventListener('error', (e) => send(`${e.message} at ${e.filename}:${e.lineno}`));
  window.addEventListener('unhandledrejection', (e) => send(`unhandled: ${e.reason}`));
}

// the autotest name -> the module that runs it and the function to call
const MODULE_RUNS = {
  studio: ['/web/autotest/studio.js', (m) => m.run(params)],
  webgpu: ['/web/selftest.js', (m) => m.runSelfTest()],
  bench: ['/web/bench.js', (m) => m.runBench()],
};

/** Open the tab named by `?tab=`, then start the `?autotest=` run if there is one. */
export function startAutotest() {
  if (params.get('tab')) { const t = $('tab-' + params.get('tab')); if (t) showTab(t); }
  const kind = params.get('autotest');
  if (kind === 'ui') uiAutotest();
  const run = MODULE_RUNS[kind];
  if (run) {
    import(run[0]).then(run[1]).then(reportSelfTest)
      .catch((err) => reportSelfTest({ kind, pass: false, error: String(err) }));
  }
}
