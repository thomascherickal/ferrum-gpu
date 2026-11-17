// Tool-tips: one shared bubble, shown on hover (after a short delay) and on keyboard focus, hidden by
// Escape, scrolling or leaving. Every tipped control also gets `aria-describedby` pointing at a
// visually hidden copy of the text, so a screen reader announces it when focus arrives.
import { helpFor, HELP } from './help.js';

export const HOVER_MS = 350;

/**
 * Where to put a bubble of `tip` ({w, h}) for a target rectangle inside a `view` ({w, h}): below
 * the target when it fits, otherwise above, always clamped inside the view.
 */
export function placeTip(target, tip, view, gap = 8) {
  const below = target.bottom + gap;
  const fitsBelow = below + tip.h <= view.h;
  const top = fitsBelow ? below : Math.max(0, target.top - gap - tip.h);
  const left = Math.min(Math.max(0, target.left), Math.max(0, view.w - tip.w));
  return { left, top: Math.min(top, Math.max(0, view.h - tip.h)) };
}

/** The key under which `el` is documented: its own id, or the control a wrapping label holds. */
export function keyFor(el) {
  if (!el || !el.closest) return null;
  if (el.id && HELP[el.id]) return el.id;
  const label = el.closest('label');
  const inner = label && label.querySelector('[id]');
  return inner && HELP[inner.id] ? inner.id : null;
}

export function mountTooltips(doc, getMode) {
  const bubble = doc.createElement('div');
  bubble.id = 'ferrum-tooltip';
  bubble.setAttribute('role', 'tooltip');
  bubble.className = 'tip';
  bubble.hidden = true;
  doc.body.appendChild(bubble);
  const store = doc.createElement('div');
  store.className = 'sr-only';
  doc.body.appendChild(store);

  let timer = null;
  let suppressed = null;

  function describe() {
    store.textContent = '';
    for (const id of Object.keys(HELP)) {
      const el = doc.getElementById(id);
      if (!el) continue;
      const span = doc.createElement('span');
      span.id = `help-${id}`;
      span.textContent = helpFor(id, getMode());
      store.appendChild(span);
      el.setAttribute('aria-describedby', span.id);
    }
  }

  function hide() {
    clearTimeout(timer);
    bubble.hidden = true;
  }

  function show(el) {
    const key = keyFor(el);
    if (!key || suppressed === el) return;
    bubble.textContent = helpFor(key, getMode());
    bubble.hidden = false;
    const r = el.getBoundingClientRect();
    const p = placeTip(r, { w: bubble.offsetWidth, h: bubble.offsetHeight },
      { w: doc.documentElement.clientWidth, h: doc.documentElement.clientHeight });
    bubble.style.left = `${p.left + (doc.defaultView?.scrollX || 0)}px`;
    bubble.style.top = `${p.top + (doc.defaultView?.scrollY || 0)}px`;
  }

  doc.addEventListener('mouseover', (e) => {
    const el = e.target;
    if (!keyFor(el)) return;
    clearTimeout(timer);
    timer = setTimeout(() => show(el), HOVER_MS);
  });
  doc.addEventListener('mouseout', (e) => {
    if (e.target === suppressed) suppressed = null;
    hide();
  });
  doc.addEventListener('focusin', (e) => { if (keyFor(e.target)) show(e.target); });
  doc.addEventListener('focusout', () => { suppressed = null; hide(); });
  doc.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !bubble.hidden) { suppressed = doc.activeElement; hide(); }
  });
  doc.addEventListener('scroll', hide, true);
  doc.defaultView?.addEventListener('resize', hide);

  describe();
  return { describe, hide, bubble };
}
