// The jobs client: start a `ferrum-gpu` subcommand on the server, follow its output, stop it.
// `fetch` and the timer are parameters so the polling logic is tested under node.
import { api, post, withQuery } from '../shell/api.js';

export const FINISHED = ['done', 'stopped', 'error'];

/** Start a job; resolves to its id. */
export async function startJob(cmd, args = [], f) {
  const r = await post('/api/jobs', { cmd, args }, f);
  return r.id;
}

/** Ask a job to stop (a clean stop file when it has one), or kill it with `force`. */
export const stopJob = (id, force = false, f) => post(`/api/jobs/${id}/stop`, { force }, f);

/**
 * Follow job `id`: poll for new lines until it finishes.
 *  - `onLines(lines)` gets each new batch (`{i, e, t, ev?}`),
 *  - `onState(poll)` gets every poll (state, last step, last validation),
 *  - `onDone(poll)` gets the final poll once.
 * After `retries` failed polls in a row it gives up with a synthetic error poll.
 * Returns `{ stop() }`, which stops polling (it does not stop the job).
 */
export function follow(id, { onLines = () => {}, onState = () => {}, onDone = () => {}, intervalMs = 300, retries = 3 } = {}, env = {}) {
  const f = env.fetch;
  const wait = env.setTimeout ?? globalThis.setTimeout;
  const unwait = env.clearTimeout ?? globalThis.clearTimeout;
  let since = 0;
  let failures = 0;
  let timer = null;
  let live = true;

  async function tick() {
    timer = null;
    if (!live) return;
    try {
      const p = await api(withQuery(`/api/jobs/${id}`, { since }), {}, f);
      failures = 0;
      if (p.lines.length) onLines(p.lines);
      since = p.next;
      onState(p);
      if (FINISHED.includes(p.state)) {
        live = false;
        onDone(p);
        return;
      }
    } catch (e) {
      failures += 1;
      if (failures >= retries) {
        live = false;
        onDone({ id, state: 'error', exit: null, error: String(e.message || e), lines: [], next: since });
        return;
      }
    }
    if (live) timer = wait(tick, intervalMs);
  }
  timer = wait(tick, 0);
  return {
    stop() {
      live = false;
      if (timer !== null) unwait(timer);
    },
  };
}

/** One line of the log as text: stderr lines are marked so they stand out. */
export function formatLine(l) {
  return l.e ? `! ${l.t}` : l.t;
}

/** The status sentence for a poll. */
export function statusText(p) {
  switch (p.state) {
    case 'running': return p.step ? `running · step ${p.step.step}/${p.step.total} · loss ${p.step.loss.toFixed(3)}` : 'running…';
    case 'stopping': return 'stopping…';
    case 'stopped': return 'stopped';
    case 'done': return 'done';
    default: return p.error ? `error: ${p.error}` : `failed (exit ${p.exit})`;
  }
}
