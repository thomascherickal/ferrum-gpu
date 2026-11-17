// Persistent interface settings (mode, theme, last tab). Storage can be missing or throw (a private
// window, blocked site data), so every access is guarded and the page works without it.

export const MODES = ['beginner', 'expert'];
export const THEMES = ['auto', 'light', 'dark'];
const KEY = 'ferrum-gpu.';

/** `v` if it is one of `allowed`, else `fallback`. */
export function pick(v, allowed, fallback) {
  return allowed.includes(v) ? v : fallback;
}

export function load(storage, name, allowed, fallback) {
  try {
    return pick(storage?.getItem(KEY + name), allowed, fallback);
  } catch {
    return fallback;
  }
}

/** True when the value was stored. */
export function save(storage, name, value) {
  try {
    storage.setItem(KEY + name, value);
    return true;
  } catch {
    return false;
  }
}

/** Apply a theme: `auto` follows the system, the others pin `data-theme` on the root element. */
export function applyTheme(root, theme) {
  if (theme === 'light' || theme === 'dark') root.dataset.theme = theme;
  else delete root.dataset.theme;
}
