// The job panel: a status line, a progress bar, the log and a Stop button for one job at a time.
// Put `<div class="job" id="…"></div>` in a tab and call `mountJobPanel`.
import { startJob, stopJob, follow, formatLine, statusText } from './client.js';

const MAX_LOG_CHARS = 400_000;

export function mountJobPanel(root, doc = document, env = {}) {
  const status = doc.createElement('p');
  status.className = 'muted';
  status.setAttribute('role', 'status');
  status.textContent = 'idle';
  const bar = doc.createElement('progress');
  bar.max = 1;
  bar.value = 0;
  bar.hidden = true;
  const stop = doc.createElement('button');
  stop.type = 'button';
  stop.textContent = 'Stop';
  stop.disabled = true;
  const log = doc.createElement('pre');
  log.className = 'joblog';
  log.setAttribute('role', 'log');
  log.tabIndex = 0;
  const row = doc.createElement('div');
  row.className = 'row';
  row.append(stop, bar, status);
  root.append(row, log);

  let current = null;
  let tracker = null;

  function append(lines) {
    log.textContent += lines.map(formatLine).join('\n') + '\n';
    if (log.textContent.length > MAX_LOG_CHARS) log.textContent = log.textContent.slice(-MAX_LOG_CHARS);
    log.scrollTop = log.scrollHeight;
  }

  stop.addEventListener('click', async () => {
    if (current === null) return;
    try { await stopJob(current, false, env.fetch); } catch (e) { status.textContent = e.message; }
  });

  /** Start `cmd args`; `hooks.onState(poll)` and `hooks.onDone(poll)` are optional. Returns the job id. */
  async function run(cmd, args, hooks = {}) {
    if (current !== null) throw new Error('a job is already running in this panel');
    log.textContent = '';
    status.className = 'muted';
    status.textContent = 'starting…';
    stop.disabled = false;
    let id;
    try {
      id = await startJob(cmd, args, env.fetch);
    } catch (e) {
      status.textContent = e.message;
      status.className = 'bad';
      stop.disabled = true;
      throw e;
    }
    current = id;
    tracker = follow(id, {
      onLines(lines) {
        append(lines);
        hooks.onLines?.(lines);
      },
      onState(p) {
        status.textContent = statusText(p);
        if (p.step) { bar.hidden = false; bar.max = p.step.total || 1; bar.value = p.step.step; }
        hooks.onState?.(p);
      },
      onDone(p) {
        current = null;
        stop.disabled = true;
        status.className = p.state === 'done' ? 'good' : p.state === 'error' ? 'bad' : 'muted';
        status.textContent = statusText(p);
        hooks.onDone?.(p);
      },
    }, env);
    return id;
  }

  return { run, busy: () => current !== null, stop: () => tracker?.stop(), elements: { status, bar, stop, log } };
}
