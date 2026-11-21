// Scripted run of the Studio tabs for the headless-browser test: drives the real controls (tabs,
// mode, tool-tips, Datasets, the file picker, Prepare, Tokenize, Train ... GPU lab) and returns what
// happened. Parameters come from the query string: file, dir, inputs, out, tok, tokout and more.
import { makeKit } from './kit.js';
import { tabs, modeAndTips } from './shell_stages.js';
import { inspect, picker, prepare, tokenize } from './data_stages.js';
import { trainAndStop, finetune, evaluate, generate } from './train_stages.js';
import { models, exportTools } from './model_stages.js';
import { serve, capable, gpuLab } from './service_stages.js';

const STAGES = [tabs, modeAndTips, inspect, picker, prepare, tokenize, trainAndStop, finetune,
  evaluate, generate, models, exportTools, serve, capable, gpuLab];

export async function run(params) {
  const kit = makeKit(params);
  try {
    for (const stage of STAGES) await stage(kit);
    return { kind: 'studio', pass: true, steps: kit.steps };
  } catch (e) {
    return { kind: 'studio', pass: false, steps: kit.steps, error: String(e.message || e) };
  }
}
