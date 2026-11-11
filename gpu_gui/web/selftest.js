// Parity self-test: the in-browser runtime against the CPU oracle's logits, produced by the Rust
// server from the same GGUF bytes. Resolves to a report; `skipped` when WebGPU is not usable.

import { Gpu } from './webgpu.js';
import { BrowserLlama } from './llama.js';

const fromBase64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer;
const relErr = (got, want) => {
  let worst = 0;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]) / Math.max(Math.abs(want[i]), 1);
    if (!(e <= worst)) worst = Number.isNaN(e) ? Infinity : e; // NaN never passes
  }
  return got.length === want.length ? worst : Infinity;
};

export async function runSelfTest() {
  const report = { kind: 'webgpu', pass: false, cases: [] };
  let gpu;
  try { gpu = await Gpu.create(); } catch (err) { return { ...report, skipped: true, reason: String(err.message || err) }; }
  report.adapter = gpu.describe();
  try {
    const { cases } = await (await fetch('/api/selftest/vectors')).json();
    for (const c of cases) {
      const model = await BrowserLlama.load(gpu, fromBase64(c.gguf));
      const want = Float32Array.from(c.logits), v = c.vocab;
      const fed = new Float32Array(want.length);
      for (let i = 0; i < c.ids.length; i++) fed.set(await model.feed(c.ids[i]), i * v);
      const all = await model.forwardAll(c.ids);
      const feedErr = relErr(fed, want), allErr = relErr(all, want);
      report.cases.push({ name: c.name, quant: c.quant, feedErr, allErr, ok: feedErr < c.tolerance && allErr < c.tolerance });
    }
    report.pass = report.cases.length > 0 && report.cases.every((c) => c.ok);
  } catch (err) { report.error = String(err.stack || err); }
  return report;
}
