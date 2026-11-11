// "Run on: this browser": download the current model as GGUF, keep it on this tab's GPU, and
// decode here. Greedy output matches the server's; with a temperature the sampler is a seeded
// browser-side one (not bit-identical to the server's).

import { Gpu } from './webgpu.js';
import { BrowserLlama, sample, mulberry32 } from './llama.js';

let gpu = null, model = null, loadedVersion = null;

async function ensureModel() {
  gpu ??= await Gpu.create();
  const info = await (await fetch('/api/info')).json();
  if (!info.model) throw new Error('create or load a model first');
  if (!model || loadedVersion !== info.model.version) {
    const r = await fetch('/api/model.gguf');
    if (!r.ok) throw new Error(`could not download the model (${r.status})`);
    model = await BrowserLlama.load(gpu, await r.arrayBuffer());
    loadedVersion = info.model.version;
  }
  if (model.cfg.vocab !== 256) throw new Error("this model's vocabulary is not bytes; text in and out needs a byte-level model");
  return model;
}

/** Generate `req.max_new` tokens after `req.prompt` on this browser's GPU; resolves to the new text. */
export async function generate(req) {
  const m = await ensureModel();
  const prompt = new TextEncoder().encode(req.prompt);
  if (prompt.length === 0) throw new Error('the prompt is empty');
  if (prompt.length >= m.cfg.ctx) throw new Error(`the prompt has ${prompt.length} tokens but the context is ${m.cfg.ctx}`);
  m.reset();
  let logits;
  for (const t of prompt) logits = await m.feed(t);
  const rand = mulberry32(req.seed ?? 1), out = [];
  while (out.length < req.max_new && prompt.length + out.length < m.cfg.ctx) {
    const next = sample(logits, req.temperature, req.top_k, rand);
    out.push(next);
    if (prompt.length + out.length < m.cfg.ctx && out.length < req.max_new) logits = await m.feed(next);
  }
  return new TextDecoder().decode(new Uint8Array(out));
}

/** Where this tab's WebGPU runs, for the report. */
export async function describe() { gpu ??= await Gpu.create(); return gpu.describe(); }
