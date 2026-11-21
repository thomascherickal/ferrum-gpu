// ferrum-gpu GUI: talks to the local server's JSON API. No external requests.
import { reportUncaughtErrors, startAutotest } from '/web/app/autotest.js';
import { refreshInfo, mountModelPanel, checkWebGpu } from '/web/app/device.js';
import { mountTrain, pollTrain } from '/web/app/train.js';
import { mountGenerate } from '/web/app/generate.js';
import { $ } from '/web/app/dom.js';

reportUncaughtErrors();
mountModelPanel();
mountTrain();
mountGenerate();
refreshInfo().then(checkWebGpu).catch((err) => { $('device-badge').textContent = err.message; });
pollTrain();
window.addEventListener('resize', pollTrain);
startAutotest();
