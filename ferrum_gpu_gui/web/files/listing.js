// The DOM of the picker panel: breadcrumbs, shortcut places, the "use this folder" button and the
// folder listing. Pure rendering; the picker passes in what clicking does.
import { humanBytes } from '../shell/api.js';
import { crumbs, join, matchesExt } from './paths.js';

/** A `<tag class=cls>` holding `text` (either may be omitted). */
export function el(doc, tag, cls, text) {
  const e = doc.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** A link-styled button labelled `label` that calls `onClick`. */
export function linkButton(doc, label, onClick) {
  const b = el(doc, 'button', 'link', label);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

const row = (doc, ...children) => { const li = el(doc, 'li'); li.append(...children); return li; };

/** The nodes that fill the panel for the listing `l`; `kind` and `exts` say what may be chosen. */
export function listingNodes(doc, l, { kind, exts, show, choose }) {
  const bar = el(doc, 'div', 'crumbs');
  for (const [label, path] of crumbs(l.dir)) bar.append(linkButton(doc, label, () => show(path)));
  const places = el(doc, 'div', 'places');
  for (const p of l.places) places.append(linkButton(doc, p.name, () => show(p.path)));
  const nodes = [bar, places];
  if (kind !== 'file') {
    const use = el(doc, 'button', null, 'Use this folder');
    use.type = 'button';
    use.addEventListener('click', () => choose(l.dir));
    nodes.push(use);
  }
  const list = el(doc, 'ul', 'listing');
  if (l.parent) list.append(row(doc, linkButton(doc, '..', () => show(l.parent))));
  for (const e of l.entries) {
    if (e.hidden) continue;
    if (e.kind === 'dir') list.append(row(doc, linkButton(doc, `${e.name}/`, () => show(join(l.dir, e.name)))));
    else if (kind !== 'dir' && e.kind === 'file' && matchesExt(e.name, exts)) {
      list.append(row(doc, linkButton(doc, e.name, () => choose(join(l.dir, e.name))), el(doc, 'span', 'muted', ` ${humanBytes(e.size)}`)));
    } else list.append(row(doc, el(doc, 'span', 'muted', e.name)));
  }
  if (l.truncated) list.append(el(doc, 'li', 'muted', 'The folder has more entries than are shown.'));
  nodes.push(list);
  return nodes;
}
