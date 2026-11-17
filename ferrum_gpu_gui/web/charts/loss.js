// The loss chart of a training run: the training loss as a line and the validation loss as dots.
// `plan` computes where everything goes (pure, tested under node); `draw` paints it on a canvas context.

/** Index of the first non-finite value, or -1. */
const finite = (xs) => xs.filter((v) => Number.isFinite(v));

/**
 * Lay out a chart of `train` (one loss per reported step, as `[step, loss]`) and `val` (`[step, loss]`
 * at each validation) inside a `w` by `h` area with the given margins. Returns the data range, the
 * pixel position of every point, and the axis labels. An empty or single-point series has no line.
 */
export function plan(train, val, w, h, m = { left: 44, right: 12, top: 10, bottom: 24 }) {
  const all = [...train, ...val].filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const box = { x: m.left, y: m.top, w: Math.max(1, w - m.left - m.right), h: Math.max(1, h - m.top - m.bottom) };
  if (!all.length) return { empty: true, box, train: [], val: [], range: null };
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const xspan = x1 - x0 || 1;
  const yspan = y1 - y0 || 1;
  const px = (x) => box.x + ((x - x0) / xspan) * box.w;
  const py = (y) => box.y + (1 - (y - y0) / yspan) * box.h;
  const place = (series) => series.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)).map(([x, y]) => [px(x), py(y)]);
  return {
    empty: false,
    box,
    range: { x0, x1, y0, y1 },
    train: place(train),
    val: place(val),
  };
}

/** The label for a loss value: three decimals, or a dash. */
export const lossLabel = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–');

/** Paint `plan(...)` on a 2D context using the CSS colours in `css` (`line`, `muted`, `accent`, `good`). */
export function draw(g, w, h, train, val, css) {
  const p = plan(train, val, w, h);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = css.line;
  g.lineWidth = 1;
  g.strokeRect(p.box.x + 0.5, p.box.y + 0.5, p.box.w, p.box.h);
  g.fillStyle = css.muted;
  g.font = '12px system-ui';
  if (p.empty) {
    g.fillText('loss will appear here', p.box.x + 10, p.box.y + p.box.h / 2);
    return p;
  }
  g.fillText(lossLabel(p.range.y1), 4, p.box.y + 10);
  g.fillText(lossLabel(p.range.y0), 4, p.box.y + p.box.h);
  g.fillText(`step ${p.range.x0}`, p.box.x, h - 6);
  g.fillText(`step ${p.range.x1}`, p.box.x + p.box.w - 54, h - 6);
  if (p.train.length > 1) {
    g.strokeStyle = css.accent;
    g.lineWidth = 2;
    g.beginPath();
    p.train.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.stroke();
  } else if (p.train.length === 1) {
    g.fillStyle = css.accent;
    g.fillRect(p.train[0][0] - 2, p.train[0][1] - 2, 4, 4);
  }
  g.fillStyle = css.good;
  for (const [x, y] of p.val) g.fillRect(x - 3, y - 3, 6, 6);
  return p;
}

export { finite };
