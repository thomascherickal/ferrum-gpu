// Studio autotest stages: Serve, Capable and the GPU lab. Each stage drives the real controls and throws on a failed check.

export async function serve({ $, check, until, steps, params }) {
  // Serve: start from the page, see it running, refuse an open address without a token, stop
  $('tab-serve').click();
  $('sv-model').value = params.get('ftout');
  $('sv-port').value = 0; $('sv-device').value = 'cpu'; $('sv-threads').value = 1;
  $('sv-start').click();
  await until(() => /^running at http:\/\/127\.0\.0\.1:\d+\//.test($('sv-status').textContent), `the server to run (${$('sv-status').textContent})`);
  check(/ferrum-gpu server: http:\/\//.test($('sv-log').textContent), `log: ${$('sv-log').textContent}`);
  check($('sv-start').disabled === true && $('sv-stop').disabled === false, 'the buttons follow the state');
  $('sv-stop').click();
  await until(() => $('sv-status').textContent === 'not running', 'the server to stop');
  check($('sv-start').disabled === false, 'it can be started again');
  $('sv-host').value = '0.0.0.0';
  $('sv-start').click();
  check(/needs a token/.test($('sv-status').textContent), `an open address without a token is refused: ${$('sv-status').textContent}`);
  $('sv-host').value = '';
  steps.push('serve');
}

export async function capable({ $, check, until, steps }) {
  // Capable: the documented answer for 16 GB, a speed ceiling, a refusal, and this machine's table
  $('tab-capable').click();
  $('cap-memory').value = 16; $('cap-bandwidth').value = 936; $('cap-model').value = 7;
  $('cap-run').click();
  await until(() => $('cap-result').querySelectorAll('tr').length === 4, 'the capacity table');
  check($('cap-result').textContent.includes('22.8B') && $('cap-result').textContent.includes('816M'), `capacity: ${$('cap-result').textContent}`);
  await until(() => /Q4_0 238 tokens\/s/.test($('cap-ceiling').textContent), `the ceiling (${$('cap-ceiling').textContent})`);
  $('cap-bandwidth').value = '';
  $('cap-run').click();
  check(/both/.test($('cap-note').textContent) && $('cap-note').className === 'bad', `a half-filled speed is refused: ${$('cap-note').textContent}`);
  await until(() => /CPU threads/.test($('cap-system').textContent), 'the machine line');
  check($('cap-machine').querySelectorAll('tr').length >= 1, 'the adapter table has rows');
  steps.push('capable');
}

export async function gpuLab({ $, check, until, steps }) {
  // GPU lab: a parity-only JSON report on the CPU-or-GPU the machine has, and the memory table
  $('tab-gpulab').click();
  $('gl-parity').checked = true; $('gl-reps').value = 1;
  $('gl-json').click();
  await until(() => /PASS|FAIL/.test($('gl-summary').textContent) || /error|failed/.test(document.querySelector('#gl-job [role=status]').textContent), 'the lab report', 120000);
  const labState = document.querySelector('#gl-job [role=status]').textContent;
  if (/no usable|needs a GPU/.test(labState)) {
    steps.push('gpulab:no-adapter');
  } else {
    // a virtual or software adapter may have looser numerics (docs/devices.md): its verdict is data, a real GPU's is strict
    const virtual = /paravirtual|virtual|llvmpipe|lavapipe|swiftshader|software|basic render/i.test($('gl-summary').textContent);
    const verdict = /(PASS|FAIL): \d+ of 10 parity checks/.test($('gl-summary').textContent);
    check(virtual ? verdict : /PASS: 10 of 10 parity checks passed/.test($('gl-summary').textContent), `lab summary: ${$('gl-summary').textContent} / ${labState}`);
    check(JSON.parse($('gl-report').value).format === 1, 'the JSON report is in the box');
    steps.push('gpulab');
  }
  await until(() => $('gl-meters').querySelectorAll('tr').length >= 1, 'the memory table');
}
