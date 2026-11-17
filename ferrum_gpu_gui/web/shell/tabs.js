// The tab bar: which tabs a mode shows, how the keyboard moves between them, and the DOM wiring.
// The decisions are pure functions (tested under node); `mountTabs` is the thin DOM layer.

/** Tabs visible in `mode`: a tab marked advanced is hidden from beginners. */
export function visibleTabs(tabs, mode) {
  return tabs.filter((t) => !(t.adv && mode === 'beginner'));
}

/**
 * The id the keyboard moves to from `current` among the visible `ids` (in order), or null when
 * the key is not a tab-bar key. Arrows wrap; Home and End jump to the ends.
 */
export function moveTo(ids, current, key) {
  if (!ids.length) return null;
  const at = ids.indexOf(current);
  switch (key) {
    case 'ArrowRight': return ids[(at + 1) % ids.length];
    case 'ArrowLeft': return ids[(at - 1 + ids.length) % ids.length];
    case 'Home': return ids[0];
    case 'End': return ids[ids.length - 1];
    default: return null;
  }
}

/** The tab to fall back to when the selected one is hidden by a mode change. */
export function fallback(ids, selected) {
  return ids.includes(selected) ? selected : (ids[0] ?? null);
}

/** Read the tab list from the DOM: `[role=tab]` buttons, `data-adv` marks the advanced ones. */
export function readTabs(root) {
  return [...root.querySelectorAll('[role="tab"]')].map((el) => ({
    id: el.id, el, adv: el.dataset.adv === 'true', panel: el.getAttribute('aria-controls'),
  }));
}

/**
 * Wire the tab bar. Returns `{ show(tabEl), applyMode(mode), selected() }`. The selected tab is
 * remembered in `onChange` so a reload can come back to it.
 */
export function mountTabs(root, doc, { mode = 'expert', onChange = () => {} } = {}) {
  const tabs = readTabs(root);
  const byId = Object.fromEntries(tabs.map((t) => [t.id, t]));
  let current = tabs[0]?.id ?? null;
  let currentMode = mode;
  const ids = () => visibleTabs(tabs, currentMode).map((t) => t.id);

  function show(el) {
    const target = typeof el === 'string' ? byId[el] : tabs.find((t) => t.el === el);
    if (!target) return;
    current = target.id;
    for (const t of tabs) {
      const on = t.id === current;
      t.el.setAttribute('aria-selected', String(on));
      t.el.tabIndex = on ? 0 : -1;
      const panel = doc.getElementById(t.panel);
      if (panel) panel.hidden = !on;
    }
    onChange(current);
  }

  function applyMode(m) {
    currentMode = m;
    const shown = new Set(ids());
    for (const t of tabs) t.el.hidden = !shown.has(t.id);
    const keep = fallback([...shown], current);
    if (keep) show(keep);
  }

  for (const t of tabs) {
    t.el.addEventListener('click', () => show(t.el));
    t.el.addEventListener('keydown', (e) => {
      const to = moveTo(ids(), t.id, e.key);
      if (!to) return;
      e.preventDefault();
      byId[to].el.focus();
      show(to);
    });
  }
  applyMode(mode);
  return { show, applyMode, selected: () => current };
}
