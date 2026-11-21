// Building the `train` command's arguments from the Train tab's form, and refusing bad values before a
// job is started. Pure: returns `{ args }` or `{ error }` (the first problem found), tested under node.
// Flag spellings and ranges follow ferrum_gpu_core::commands (docs/user-guide.md section 3).
import { blank, text } from './fields.js';
import { TRAIN_SECTIONS } from './train_sections.js';

export { LENGTH_KINDS, PRESETS, QUANTS, DEVICES, GPU_WEIGHTS } from './options.js';

/** The arguments of `train` (GPT-2, the default architecture). */
export function trainArgs(f) {
  if (blank(f.data)) return { error: 'Choose the training text (a file, a folder, or a token file).' };
  if (blank(f.out)) return { error: 'Choose where to save the model (a path ending in .gguf).' };
  const args = ['--data', text(f.data), '--out', text(f.out), '--verbose'];
  for (const section of TRAIN_SECTIONS) {
    const bad = section(f, args);
    if (bad) return bad;
  }
  return { args };
}
