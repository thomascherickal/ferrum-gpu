// The Studio shell: tabs, Beginner/Expert mode, theme and tool-tips, wired to the page.
import { mountTabs } from './tabs.js';
import { MODES, THEMES, load, save, pick, applyTheme } from './settings.js';
import { mountTooltips } from './tooltip.js';

function storage() {
  try { return window.localStorage; } catch { return null; }
}

const store = storage();
let mode = load(store, 'mode', MODES, 'expert');
let theme = load(store, 'theme', THEMES, 'auto');
const tabIds = [...document.querySelectorAll('nav[role="tablist"] [role="tab"]')].map((t) => t.id);
const lastTab = load(store, 'tab', tabIds, null);

const tabs = mountTabs(document.querySelector('nav[role="tablist"]'), document, {
  mode,
  onChange: (id) => save(store, 'tab', id),
});
if (lastTab) tabs.show(lastTab);
const tips = mountTooltips(document, () => mode);

const modeSel = document.getElementById('ui-mode');
const themeSel = document.getElementById('ui-theme');
modeSel.value = mode;
themeSel.value = theme;
applyTheme(document.documentElement, theme);
document.body.dataset.mode = mode;

modeSel.addEventListener('change', () => {
  mode = pick(modeSel.value, MODES, 'expert');
  save(store, 'mode', mode);
  document.body.dataset.mode = mode;
  tabs.applyMode(mode);
  tips.describe();
});
themeSel.addEventListener('change', () => {
  theme = pick(themeSel.value, THEMES, 'auto');
  save(store, 'theme', theme);
  applyTheme(document.documentElement, theme);
});

export const showTab = (tab) => tabs.show(tab);
export const currentMode = () => mode;
