// In-browser decode speed on each distinct WebGPU adapter the browser offers: ms per token for the
// benchmark shapes of the native benchmark (random weights), f32 / Q8_0 / Q4_0.

import { Gpu } from './webgpu.js';
import { BrowserLlama } from './llama.js';
import { REQUESTS } from './selftest.js';

const SHAPES = ['small', 'medium', 'large'], QUANTS = ['f32', 'q8_0', 'q4_0'], PROMPT = 4, MEASURED = 48;

export async function runBench() {
  const report = { kind: 'bench', pass: true, adapters: [] };
  if (!navigator.gpu) return { ...report, pass: false, skipped: true, reason: 'WebGPU is not available in this browser' };
  const seen = new Set();
  for (const opts of REQUESTS) {
    let gpu;
    try { gpu = await Gpu.create(opts); } catch (err) { continue; }
    const d = gpu.describe(), id = `${d.vendor}/${d.architecture}`;
    if (seen.has(id)) continue;
    seen.add(id);
    const rows = [];
    for (const shape of SHAPES) {
      for (const quant of QUANTS) {
        const bytes = await (await fetch(`/api/selftest/bench-model.gguf?shape=${shape}&quant=${quant}`)).arrayBuffer();
        const m = await BrowserLlama.load(gpu, bytes), toks = Array.from({ length: PROMPT + MEASURED }, (_, i) => (i * 7 + 3) % m.cfg.vocab);
        for (const t of toks.slice(0, PROMPT)) await m.feed(t); // warm up and fill the prompt
        const t0 = performance.now();
        for (const t of toks.slice(PROMPT)) await m.feed(t);
        const ms = (performance.now() - t0) / MEASURED;
        rows.push({ shape, quant, msPerToken: ms });
      }
    }
    report.adapters.push({ adapter: d, rows });
  }
  if (!report.adapters.length) return { ...report, pass: false, skipped: true, reason: 'no WebGPU adapter was granted' };
  return report;
}
