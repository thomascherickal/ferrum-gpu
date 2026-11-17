// A live loss chart bound to a canvas: it keeps the series, repaints on resize, and takes its colours
// from the page's CSS variables so it follows the theme. Used by the Train and Fine-tune tabs.
import { draw } from './loss.js';

/** Fold output lines into the series: steps become `[step, loss]`, validations `[lastStep, loss]`. */
export function collect(lines, series) {
  for (const l of lines) {
    const ev = l.ev;
    if (!ev) continue;
    if (ev.kind === 'step') {
      series.train.push([ev.step, ev.loss]);
      series.lastStep = ev.step;
    } else if (ev.kind === 'val') {
      series.val.push([series.lastStep, ev.loss]);
    }
  }
  return series;
}

export function mountLossChart(doc, canvas) {
  const series = { train: [], val: [], lastStep: 0 };

  function css() {
    const s = doc.defaultView.getComputedStyle(doc.documentElement);
    const v = (k) => s.getPropertyValue(k).trim();
    return { line: v('--line'), muted: v('--muted'), accent: v('--accent'), good: v('--good') };
  }

  function paint() {
    const ratio = doc.defaultView.devicePixelRatio || 1;
    const w = canvas.clientWidth || 600;
    const h = Math.round((w * 220) / 600);
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    const g = canvas.getContext('2d');
    g.scale(ratio, ratio);
    draw(g, w, h, series.train, series.val, css());
  }

  function reset() {
    series.train.length = 0;
    series.val.length = 0;
    series.lastStep = 0;
    paint();
  }

  paint();
  doc.defaultView.addEventListener('resize', paint);
  return { series, paint, reset, add: (lines) => { collect(lines, series); paint(); } };
}
