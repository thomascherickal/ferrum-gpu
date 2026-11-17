// The Studio entry point: the shell (tabs, mode, theme, tool-tips), then each tab's behaviour.
// Every tab is one module under /web/tabs/; add a tab by adding its module here.
import { showTab, currentMode } from './shell/index.js';
import { mountPickers } from './files/picker.js';
import { mountDatasets } from './tabs/datasets.js';
import { mountPrepare } from './tabs/prepare.js';
import { mountGtrain } from './tabs/gtrain.js';
import { mountFinetune } from './tabs/finetune.js';
import { mountEvaluate } from './tabs/evaluate.js';
import { mountGenerate } from './tabs/generate.js';

mountPickers(document);
mountDatasets(document);
mountPrepare(document);
mountGtrain(document);
mountFinetune(document);
mountEvaluate(document);
mountGenerate(document);

export { showTab, currentMode };
