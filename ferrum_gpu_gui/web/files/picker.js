// The server-side file picker: a Browse button beside a path box that opens a listing of the
// server's folders (read-only, from /api/fs/list). A folder is entered by clicking it; a file is
// chosen by clicking it. Markup:
//   <div class="picker" data-kind="file|dir|any" data-ext=".gguf,.txt"><label>…<input id="x"></label>
//     <button type="button" class="browse">Browse…</button><div class="picker-panel" hidden></div></div>
import { api, withQuery } from '../shell/api.js';
import { el, listingNodes } from './listing.js';

export function mountPicker(root, doc = document, f) {
  const input = root.querySelector('input');
  const button = root.querySelector('.browse');
  const panel = root.querySelector('.picker-panel');
  const kind = root.dataset.kind || 'any';
  const exts = (root.dataset.ext || '').split(',').map((s) => s.trim()).filter(Boolean);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-label', `Browse for ${input.closest('label')?.firstChild?.textContent?.trim() || 'a path'}`);

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
      panel.append(el(doc, 'p', 'bad', e.message));
      return;
    }
    panel.textContent = '';
    panel.append(...listingNodes(doc, l, { kind, exts, show, choose }));
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
