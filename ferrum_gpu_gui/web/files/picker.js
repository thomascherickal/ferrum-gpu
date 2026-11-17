// The server-side file picker: a Browse button beside a path box that opens a listing of the
// server's folders (read-only, from /api/fs/list). A folder is entered by clicking it; a file is
// chosen by clicking it. Markup:
//   <div class="picker" data-kind="file|dir|any" data-ext=".gguf,.txt"><label>…<input id="x"></label>
//     <button type="button" class="browse">Browse…</button><div class="picker-panel" hidden></div></div>
import { api, withQuery, humanBytes } from '../shell/api.js';
import { crumbs, join, matchesExt } from './paths.js';

export function mountPicker(root, doc = document, f) {
  const input = root.querySelector('input');
  const button = root.querySelector('.browse');
  const panel = root.querySelector('.picker-panel');
  const kind = root.dataset.kind || 'any';
  const exts = (root.dataset.ext || '').split(',').map((s) => s.trim()).filter(Boolean);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-label', `Browse for ${input.closest('label')?.firstChild?.textContent?.trim() || 'a path'}`);

  const el = (tag, cls, text) => {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function choose(path) {
    input.value = path;
    input.dispatchEvent(new doc.defaultView.Event('input', { bubbles: true }));
    input.dispatchEvent(new doc.defaultView.Event('change', { bubbles: true }));
    close();
  }

  function close() {
    panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  }

  async function show(dir) {
    panel.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    panel.textContent = 'loading…';
    let l;
    try {
      l = await api(withQuery('/api/fs/list', { dir }), {}, f);
    } catch (e) {
      panel.textContent = '';
      panel.append(el('p', 'bad', e.message));
      return;
    }
    panel.textContent = '';
    const bar = el('div', 'crumbs');
    for (const [label, path] of crumbs(l.dir)) {
      const b = el('button', 'link', label);
      b.type = 'button';
      b.addEventListener('click', () => show(path));
      bar.append(b);
    }
    panel.append(bar);
    const places = el('div', 'places');
    for (const p of l.places) {
      const b = el('button', 'link', p.name);
      b.type = 'button';
      b.addEventListener('click', () => show(p.path));
      places.append(b);
    }
    panel.append(places);
    if (kind !== 'file') {
      const use = el('button', null, 'Use this folder');
      use.type = 'button';
      use.addEventListener('click', () => choose(l.dir));
      panel.append(use);
    }
    const list = el('ul', 'listing');
    if (l.parent) {
      const li = el('li');
      const b = el('button', 'link', '..');
      b.type = 'button';
      b.addEventListener('click', () => show(l.parent));
      li.append(b);
      list.append(li);
    }
    for (const e of l.entries) {
      if (e.hidden) continue;
      const li = el('li');
      if (e.kind === 'dir') {
        const b = el('button', 'link', `${e.name}/`);
        b.type = 'button';
        b.addEventListener('click', () => show(join(l.dir, e.name)));
        li.append(b);
      } else if (kind !== 'dir' && e.kind === 'file' && matchesExt(e.name, exts)) {
        const b = el('button', 'link', e.name);
        b.type = 'button';
        b.addEventListener('click', () => choose(join(l.dir, e.name)));
        li.append(b, el('span', 'muted', ` ${humanBytes(e.size)}`));
      } else {
        li.append(el('span', 'muted', e.name));
      }
      list.append(li);
    }
    if (l.truncated) list.append(el('li', 'muted', 'The folder has more entries than are shown.'));
    panel.append(list);
  }

  button.addEventListener('click', () => {
    if (!panel.hidden) return close();
    const start = input.value.trim();
    const dir = kind === 'dir' || !start ? start : start.replace(/[\\/][^\\/]*$/, '') || start;
    show(dir || undefined);
  });
  return { show, close, choose };
}

export function mountPickers(doc = document, f) {
  return [...doc.querySelectorAll('.picker')].map((r) => mountPicker(r, doc, f));
}
