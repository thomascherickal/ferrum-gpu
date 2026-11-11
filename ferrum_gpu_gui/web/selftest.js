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

async function runCases(gpu, cases) {
  const out = [];
  for (const c of cases) {
    const model = await BrowserLlama.load(gpu, fromBase64(c.gguf));
    const want = Float32Array.from(c.logits), v = c.vocab;
    const fed = new Float32Array(want.length);
    for (let i = 0; i < c.ids.length; i++) fed.set(await model.feed(c.ids[i]), i * v);
    const all = await model.forwardAll(c.ids);
    const feedErr = relErr(fed, want), allErr = relErr(all, want);
    out.push({ name: c.name, quant: c.quant, feedErr, allErr, ok: feedErr < c.tolerance && allErr < c.tolerance });
  }
  return out;
}

/** The adapter requests worth trying: the browser's idea of a fast GPU, a low-power GPU, the software fallback. */
export const REQUESTS = [{ powerPreference: 'high-performance' }, { powerPreference: 'low-power' }, { forceFallbackAdapter: true }];

export async function runSelfTest() {
  const report = { kind: 'webgpu', pass: false, adapters: [] };
  if (!navigator.gpu) return { ...report, skipped: true, reason: 'WebGPU is not available in this browser' };
  let cases;
  try { cases = (await (await fetch('/api/selftest/vectors')).json()).cases; } catch (err) { return { ...report, error: String(err) }; }
  const seen = new Set();
  for (const opts of REQUESTS) {
    let gpu;
    try { gpu = await Gpu.create(opts); } catch (err) { continue; }
    const d = gpu.describe(), id = `${d.vendor}/${d.architecture}/${d.description}/${d.fallback}`;
    if (seen.has(id)) continue; // the same adapter answered a different request
    seen.add(id);
    const entry = { request: opts, adapter: d, cases: [], pass: false };
    try { entry.cases = await runCases(gpu, cases); entry.pass = entry.cases.length > 0 && entry.cases.every((c) => c.ok); } catch (err) { entry.error = String(err.stack || err); }
    report.adapters.push(entry);
  }
  if (!report.adapters.length) return { ...report, skipped: true, reason: 'no WebGPU adapter was granted' };
  report.pass = report.adapters.every((a) => a.pass);
  return report;
}
