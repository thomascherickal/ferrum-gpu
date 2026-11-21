// The helpers every Studio autotest stage shares, bundled as one context object.

/** The context handed to each stage: `params` from the query string and the step log. */
export function makeKit(params) {
  const $ = (id) => document.getElementById(id);
  const fail = (m) => { throw new Error(m); };
  const status = (id) => $(id).querySelector('[role="status"]').textContent;
  return {
    $, params, steps: [],
    check: (cond, m) => { if (!cond) fail(m); },
    until: async (fn, what, ms = 90000) => {
      const t0 = performance.now();
      while (!fn()) {
        if (performance.now() - t0 > ms) fail(`timed out waiting for ${what}`);
        await new Promise((r) => setTimeout(r, 40));
      }
    },
    key: (el, k) => el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true })),
    change: (el, v) => { el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); },
    selected: () => document.querySelector('[role="tab"][aria-selected="true"]').id,
    tip: () => $('ferrum-tooltip'),
    st: status,
    finished: (id) => /^(done|error|failed|stopped)/.test(status(id)),
  };
}
