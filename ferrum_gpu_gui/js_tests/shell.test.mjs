// Node tests of the Studio shell: tab visibility and keyboard movement, settings storage that may
// throw, and tool-tip placement and lookup.
// usage: node shell.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { visibleTabs, moveTo, fallback } from '../web/shell/tabs.js';
import { pick, load, save, applyTheme, MODES, THEMES } from '../web/shell/settings.js';
import { placeTip, keyFor, HOVER_MS } from '../web/shell/tooltip.js';
import { HELP, helpFor } from '../web/shell/help.js';

// ---- visibleTabs / moveTo / fallback
const tabs = [{ id: 'a' }, { id: 'b', adv: true }, { id: 'c' }];
eq(visibleTabs(tabs, 'expert').map((t) => t.id), ['a', 'b', 'c']);
eq(visibleTabs(tabs, 'beginner').map((t) => t.id), ['a', 'c']);
eq(visibleTabs([], 'beginner'), []);
const ids = ['a', 'b', 'c'];
eq(moveTo(ids, 'a', 'ArrowRight'), 'b');
eq(moveTo(ids, 'c', 'ArrowRight'), 'a');       // wraps
eq(moveTo(ids, 'a', 'ArrowLeft'), 'c');        // wraps
eq(moveTo(ids, 'b', 'Home'), 'a');
eq(moveTo(ids, 'b', 'End'), 'c');
eq(moveTo(ids, 'b', 'x'), null);
eq(moveTo([], 'a', 'ArrowRight'), null);
eq(moveTo(['a'], 'a', 'ArrowRight'), 'a');
eq(fallback(ids, 'b'), 'b');
eq(fallback(['a', 'c'], 'b'), 'a');
eq(fallback([], 'b'), null);

// ---- settings
eq(pick('expert', MODES, 'x'), 'expert');
eq(pick('nope', MODES, 'beginner'), 'beginner');
eq(pick(undefined, THEMES, 'auto'), 'auto');
const mem = new Map();
const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v) };
eq(save(storage, 'mode', 'beginner'), true);
eq(load(storage, 'mode', MODES, 'expert'), 'beginner');
mem.set('ferrum-gpu.mode', 'garbage');
eq(load(storage, 'mode', MODES, 'expert'), 'expert');   // a stored value that is not allowed is ignored
const broken = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
eq(load(broken, 'mode', MODES, 'expert'), 'expert');
eq(save(broken, 'mode', 'beginner'), false);
eq(load(null, 'mode', MODES, 'expert'), 'expert');
const root = { dataset: {} };
applyTheme(root, 'dark'); eq(root.dataset.theme, 'dark');
applyTheme(root, 'light'); eq(root.dataset.theme, 'light');
applyTheme(root, 'auto'); eq('theme' in root.dataset, false);

// ---- tool-tip placement
const view = { w: 800, h: 600 };
eq(placeTip({ left: 100, top: 100, bottom: 130 }, { w: 300, h: 60 }, view), { left: 100, top: 138 });
eq(placeTip({ left: 100, top: 560, bottom: 590 }, { w: 300, h: 60 }, view), { left: 100, top: 492 }); // above
eq(placeTip({ left: 700, top: 100, bottom: 130 }, { w: 300, h: 60 }, view), { left: 500, top: 138 });  // clamped right
eq(placeTip({ left: -50, top: 100, bottom: 130 }, { w: 300, h: 60 }, view), { left: 0, top: 138 });    // clamped left
eq(placeTip({ left: 0, top: 5, bottom: 590 }, { w: 300, h: 60 }, view).top >= 0, true);
ok(HOVER_MS > 0 && HOVER_MS < 1000);

// ---- help lookup
eq(helpFor('seed', 'expert') === HELP.seed[0], true);
eq(helpFor('seed', 'beginner') === HELP.seed[1], true);
eq(helpFor('seed', 'anything else') === HELP.seed[0], true);
eq(helpFor('no-such-control', 'expert'), null);
const fakeEl = (id, inner) => ({ id, closest: (s) => (s === 'label' && inner !== undefined ? { querySelector: () => inner } : null) });
eq(keyFor(fakeEl('seed')), 'seed');
eq(keyFor(fakeEl('', { id: 'steps' })), 'steps');
eq(keyFor(fakeEl('', { id: 'not-documented' })), null);
eq(keyFor(fakeEl('not-documented')), null);
eq(keyFor(null), null);
eq(keyFor({}), null);

summary();
