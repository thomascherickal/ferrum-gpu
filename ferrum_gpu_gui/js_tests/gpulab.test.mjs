// Node tests of the GPU lab tab's helpers.
// usage: node gpulab.test.mjs
import assert from 'node:assert/strict';
import { labArgs, reportFromLines, reportSummary, meterView } from '../web/tabs/lab_args.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };

eq(labArgs({}), { args: ['--size', 'small'] });
eq(labArgs({ adapter: ' Xe ', size: 'medium', reps: '3', parityOnly: true, json: true }),
  { args: ['--gpu-adapter', 'Xe', '--size', 'medium', '--reps', '3', '--parity-only', '--json'] });
for (const [f, needle] of [
  [{ size: 'huge' }, 'small or medium'],
  [{ reps: '0' }, '1 to 200'],
  [{ reps: '201' }, '1 to 200'],
  [{ reps: '2.5' }, '1 to 200'],
  [{ reps: 'x' }, '1 to 200'],
]) {
  const r = labArgs(f);
  ok(r.error && r.error.includes(needle), JSON.stringify(f));
  eq(r.args, undefined);
}

const report = { format: 1, passed: true, adapter: { name: 'Fake GPU' }, checks: [{ ok: true }, { ok: true }, { ok: false }], speeds: [{}, {}, {}] };
const lines = [{ t: 'noise' }, { t: '! warning', e: true }, { t: JSON.stringify(report) }, { t: 'trailing text' }];
eq(reportFromLines(lines), report);
eq(reportFromLines([{ t: '{not json' }, { t: '{"format":2,"checks":[]}' }, { t: '{"format":1}' }]), null);
eq(reportFromLines([]), null);
eq(reportFromLines([{ t: JSON.stringify(report), e: true }]), null);

eq(reportSummary({ ...report, passed: false }), 'FAIL: 2 of 3 parity checks passed on Fake GPU, speed measured at 3 precisions.');
eq(reportSummary({ ...report, adapter: null, speeds: [] }), 'PASS: 2 of 3 parity checks passed on no adapter, no speed measured.');

eq(meterView({ source: 'nvidia-smi', total_mib: 2048, free_mib: 1794 }), { text: '1794 MiB free of 2048 MiB (nvidia-smi)', value: 1794, max: 2048 });
eq(meterView({ source: 'sysfs', total_mib: 4096, free_mib: null }), { text: '4096 MiB in total (sysfs)', value: 0, max: 4096 });
eq(meterView({ source: 'ram', total_mib: 8000 }), { text: '8000 MiB in total (ram)', value: 0, max: 8000 });
eq(meterView(null), { text: 'memory: not readable on this machine', value: 0, max: 0 });

console.log(`OK ${checks} checks`);
