// Node tests of the Studio shell: tab visibility and keyboard movement, settings storage that may
// throw, tool-tip placement and lookup, and the tab bar wired to a small fake DOM.
// usage: node shell.test.mjs
import assert from 'node:assert/strict';
import { visibleTabs, moveTo, fallback, mountTabs, readTabs } from '../web/shell/tabs.js';
import { pick, load, save, applyTheme, MODES, THEMES } from '../web/shell/settings.js';
import { placeTip, keyFor, HOVER_MS } from '../web/shell/tooltip.js';
import { HELP, helpFor } from '../web/shell/help.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

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

// ---- the tab bar on a fake DOM
function fakeTab(id, adv) {
  const attrs = { 'aria-controls': `panel-${id}` };
  const handlers = {};
  return {
    id, dataset: { adv: String(!!adv) }, hidden: false, tabIndex: -1, focused: false,
    getAttribute: (k) => attrs[k], setAttribute: (k, v) => { attrs[k] = v; }, attrs,
    addEventListener: (ev, fn) => { handlers[ev] = fn; }, handlers,
    focus() { this.focused = true; },
  };
}
const els = [fakeTab('t1'), fakeTab('t2', true), fakeTab('t3')];
const panels = Object.fromEntries(els.map((e) => [`panel-${e.id}`, { hidden: true }]));
const navRoot = { querySelectorAll: () => els };
const docStub = { getElementById: (id) => panels[id] };
eq(readTabs(navRoot).map((t) => [t.id, t.adv]), [['t1', false], ['t2', true], ['t3', false]]);
const seen = [];
const bar = mountTabs(navRoot, docStub, { mode: 'expert', onChange: (id) => seen.push(id) });
eq(bar.selected(), 't1');
eq(panels['panel-t1'].hidden, false);
eq(panels['panel-t2'].hidden, true);
eq(els[0].attrs['aria-selected'], 'true');
eq(els[0].tabIndex, 0);
eq(els[1].tabIndex, -1);
els[0].handlers.click();                                   // a click selects
bar.show(els[2]);
eq(bar.selected(), 't3');
eq(panels['panel-t3'].hidden, false);
eq(panels['panel-t1'].hidden, true);
let prevented = false;
els[2].handlers.keydown({ key: 'ArrowRight', preventDefault() { prevented = true; } });
eq(bar.selected(), 't1'); eq(prevented, true); eq(els[0].focused, true);   // wraps to the first
prevented = false;
els[0].handlers.keydown({ key: 'ArrowRight', preventDefault() { prevented = true; } });
eq(bar.selected(), 't2');                                  // expert: the advanced tab is reachable
els[1].handlers.keydown({ key: 'q', preventDefault() { prevented = true; } });   // other keys do nothing
eq(bar.selected(), 't2');
bar.applyMode('beginner');                                 // the selected advanced tab is hidden: fall back
eq(els[1].hidden, true);
eq(bar.selected(), 't1');
eq(els[2].hidden, false);
els[0].handlers.keydown({ key: 'ArrowRight', preventDefault() {} });
eq(bar.selected(), 't3');                                  // beginner skips the hidden tab
bar.applyMode('expert');
eq(els[1].hidden, false);
ok(seen.length > 3, 'onChange was called');
bar.show('t2'); eq(bar.selected(), 't2');                  // by id
bar.show('nope'); eq(bar.selected(), 't2');                // an unknown id is ignored
eq(mountTabs({ querySelectorAll: () => [] }, docStub).selected(), null);   // no tabs at all

console.log(`OK ${checks} checks`);
