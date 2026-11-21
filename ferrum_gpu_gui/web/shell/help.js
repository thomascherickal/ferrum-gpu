// Tool-tip text for every control. Each entry is `[expert, beginner]` on ONE line, with no apostrophes
// or double quotes inside the text (a Rust test reads this file line by line and counts the words).
// The expert text says what the control is and what it does to the engine; the beginner text says
// what it is for. Both must run to 25 words or more.
import { DATA_AND_TRAINING } from './help/data_and_training.js';
import { MODELS_AND_SERVING } from './help/models_and_serving.js';
import { LAB_AND_PAGE } from './help/lab_and_page.js';

export const HELP = { ...DATA_AND_TRAINING, ...MODELS_AND_SERVING, ...LAB_AND_PAGE };

/** The tip for control `id` in `mode` ('beginner' or 'expert'), or null if there is none. */
export function helpFor(id, mode) {
  const e = HELP[id];
  if (!e) return null;
  return mode === 'beginner' ? e[1] : e[0];
}
