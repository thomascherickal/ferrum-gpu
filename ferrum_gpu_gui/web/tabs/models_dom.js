// The DOM of the Models tab: the tensor and metadata tables and the list of model files.
import { humanBytes } from '../shell/api.js';
import { dimsText, rowMatches, timeText } from './models_view.js';

/** A `<tag class=cls>` holding `text`. */
const cell = (doc, tag, text, cls) => {
  const e = doc.createElement(tag);
  e.textContent = text;
  if (cls) e.className = cls;
  return e;
};

const appendRow = (doc, body, ...cells) => {
  const tr = doc.createElement('tr');
  tr.append(...cells.map((c) => cell(doc, 'td', c)));
  body.append(tr);
};

/** The tensors of `info` whose rows match the filter text `q`, one table row each. */
export function renderTensors(doc, body, info, q) {
  body.textContent = '';
  if (!info) return;
  for (const t of info.tensors.filter((r) => rowMatches(r, q))) {
    appendRow(doc, body, t.name, dimsText(t.dims), t.kind, t.bytes === null ? '–' : humanBytes(t.bytes));
  }
}

/** The metadata key / value rows of `info`. */
export function renderMeta(doc, body, info) {
  body.textContent = '';
  for (const m of info.metadata) appendRow(doc, body, m.key, m.value);
}

/** One list item per model file: a button that calls `onPick(file)`, then its size and age. */
export function renderModelList(doc, list, files, onPick) {
  for (const f of files) {
    const li = doc.createElement('li');
    const b = cell(doc, 'button', f.name, 'link');
    b.type = 'button';
    b.addEventListener('click', () => onPick(f));
    li.append(b, cell(doc, 'span', ` ${humanBytes(f.size)} · ${timeText(f.modified)}`, 'muted'));
    list.append(li);
  }
}
