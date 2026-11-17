// The Studio entry point: the shell (tabs, mode, theme, tool-tips), then each tab's behaviour.
// Every tab is one module under /web/tabs/; add a tab by adding its module here.
import { showTab, currentMode } from './shell/index.js';
import { mountPickers } from './files/picker.js';
import { mountDatasets } from './tabs/datasets.js';
import { mountPrepare } from './tabs/prepare.js';
import { mountGtrain } from './tabs/gtrain.js';

mountPickers(document);
mountDatasets(document);
mountPrepare(document);
mountGtrain(document);

export { showTab, currentMode };
