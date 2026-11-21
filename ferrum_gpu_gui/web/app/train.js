// The first-page training panel: start / stop, the status line, the progress bar and the loss chart.
import { api, post } from '/web/shell/api.js';
import { $ } from './dom.js';
import { refreshInfo } from './device.js';

function drawChart(losses) {
  const c = $('chart'), ratio = window.devicePixelRatio || 1;
  const w = c.clientWidth || 600, h = Math.round(w * 220 / 600);
  c.width = w * ratio; c.height = h * ratio;
  const g = c.getContext('2d');
  g.scale(ratio, ratio);
  const css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, w, h);
  g.strokeStyle = css.getPropertyValue('--line'); g.fillStyle = css.getPropertyValue('--muted');
  g.font = '12px system-ui'; g.lineWidth = 1;
  g.strokeRect(40.5, 8.5, w - 52, h - 30);
  if (losses.length < 2) { g.fillText('loss will appear here', 50, h / 2); return; }
  const max = Math.max(...losses), min = Math.min(...losses), span = max - min || 1;
  g.fillText(max.toFixed(2), 4, 20); g.fillText(min.toFixed(2), 4, h - 22);
  g.fillText('step ' + losses.length, w - 70, h - 6);
  g.strokeStyle = css.getPropertyValue('--accent'); g.lineWidth = 2; g.beginPath();
  losses.forEach((l, i) => {
    const x = 41 + (i / (losses.length - 1)) * (w - 54), y = 9 + (1 - (l - min) / span) * (h - 32);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  });
  g.stroke();
}

let poll = null;
export async function pollTrain() {
  try {
    const s = await api('/api/train/status');
    $('progress').max = s.total || 1; $('progress').value = s.step;
    const last = s.losses.length ? s.losses[s.losses.length - 1].toFixed(3) : '–';
    const val = s.val_before == null ? '' : ` · validation ${s.val_before.toFixed(3)}${s.val_after == null ? '' : ' → ' + s.val_after.toFixed(3)}`;
    $('train-status').textContent = s.state === 'error' ? `error: ${s.error}` : `${s.state} · step ${s.step}/${s.total || '?'} · loss ${last}${val}`;
    $('train-status').className = s.state === 'error' ? 'bad' : s.state === 'done' ? 'good' : 'muted';
    drawChart(s.losses);
    const running = s.state === 'running';
    $('train-start').disabled = running; $('train-stop').disabled = !running;
    if (!running && poll) { clearInterval(poll); poll = null; refreshInfo(); }
  } catch (err) { $('train-status').textContent = err.message; }
}

export function mountTrain() {
  $('train-start').addEventListener('click', async () => {
    try {
      await post('/api/train', {
        text: $('text').value, steps: Number($('steps').value), lr: Number($('lr').value),
        batch: Number($('batch').value), context: Number($('context').value),
        val: Number($('val').value) / 100,
      });
      if (!poll) poll = setInterval(pollTrain, 400);
      pollTrain();
    } catch (err) { $('train-status').textContent = err.message; $('train-status').className = 'bad'; }
  });
  $('train-stop').addEventListener('click', () => post('/api/train/stop', {}).catch(() => {}));
}
