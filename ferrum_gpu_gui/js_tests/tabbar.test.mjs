// Node tests of the tab bar wired to a small fake DOM.
// usage: node tabbar.test.mjs
import { eq, ok, summary } from './lib/check.js';
import { mountTabs, readTabs } from '../web/shell/tabs.js';

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

summary();
