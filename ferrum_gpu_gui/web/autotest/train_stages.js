// Studio autotest stages: Train, Fine-tune, Evaluate and Generate. Each stage drives the real controls and throws on a failed check.

export async function trainAndStop({ $, check, until, steps, params, change, selected }) {
  // Train: a bad form is refused, beginners do not see the expert options, a short run completes with
  // a chart and a saved model, and a long one is stopped from the page
  $('tab-gtrain').click();
  check(selected() === 'tab-gtrain', 'the Train tab opens');
  $('gt-data').value = '';
  $('gt-start').click();
  check(/training text/.test($('gt-job').querySelector('[role="status"]').textContent), 'a form with no data is refused');
  change($('ui-mode'), 'beginner');
  check(getComputedStyle(document.querySelector('#panel-gtrain .adv')).display === 'none', 'the expert options are hidden for beginners');
  change($('ui-mode'), 'expert');
  check(getComputedStyle(document.querySelector('#panel-gtrain .adv')).display !== 'none', 'and shown for experts');
  const fill = (out, kind, n) => {
    $('gt-data').value = params.get('file');
    $('gt-out').value = out;
    $('gt-preset').value = 'tiny';
    $('gt-lengthkind').value = kind; $('gt-length').value = n;
    $('gt-batch').value = 4; $('gt-vocab').value = 300; $('gt-device').value = 'cpu';
    $('gt-val').value = '0.2';
  };
  const status = () => $('gt-job').querySelector('[role="status"]').textContent;
  fill(params.get('gtout'), 'steps', 30);
  $('gt-start').click();
  await until(() => /^(done|error|failed)/.test(status()), 'the training to finish');
  check(status() === 'done', `train: ${status()} ${$('gt-job').textContent.slice(-300)}`);
  check(/step 30\/30/.test($('gt-job').querySelector('pre').textContent), 'the step lines were streamed');
  check(/validation|val loss/.test($('gt-job').querySelector('pre').textContent), 'validation lines were streamed');
  const px = $('gt-chart').getContext('2d').getImageData(0, 0, $('gt-chart').width, $('gt-chart').height).data;
  let drawn = 0;
  for (let i = 3; i < px.length; i += 4) if (px[i] !== 0) drawn++;
  check(drawn > 200, `the loss chart was painted (${drawn} pixels)`);
  steps.push('train');

  fill(params.get('gtstop'), 'steps', 100000);
  $('gt-start').click();
  await until(() => /step \d+\//.test($('gt-stats').textContent), 'a step to be reported');
  [...$('gt-job').querySelectorAll('button')].find((b) => b.textContent === 'Stop').click();
  await until(() => /^(stopped|done|error|failed)/.test(status()), 'the stop to take effect');
  check(status() === 'stopped', `stop: ${status()}`);
  steps.push('train-stop');
}

export async function finetune({ $, check, until, steps, params, st, finished }) {
  // Fine-tune the model Train saved, evaluate the result, and generate from it
  $('tab-finetune').click();
  $('ft-model').value = params.get('gtout');
  $('ft-data').value = params.get('file');
  $('ft-out').value = params.get('ftout');
  $('ft-lengthkind').value = 'steps'; $('ft-length').value = 10; $('ft-batch').value = 4; $('ft-device').value = 'cpu';
  $('ft-start').click();
  await until(() => finished('ft-job'), 'the fine-tune to finish');
  check(st('ft-job') === 'done', `fine-tune: ${st('ft-job')} ${$('ft-job').textContent.slice(-300)}`);
  steps.push('finetune');
}

export async function evaluate({ $, check, until, steps, params, st, finished }) {
  $('tab-evaluate').click();
  $('ev-model').value = params.get('ftout');
  $('ev-data').value = params.get('file');
  $('ev-start').click();
  await until(() => finished('ev-job'), 'the evaluation to finish');
  check(st('ev-job') === 'done', `evaluate: ${st('ev-job')} ${$('ev-job').textContent.slice(-300)}`);
  check(/perplexity [\d.]+ · loss [\d.]+ · over \d+ tokens/.test($('ev-result').textContent), `result: ${$('ev-result').textContent}`);
  steps.push('evaluate');
}

export async function generate({ $, check, until, steps, params, st, finished }) {
  $('tab-ggen').click();
  $('gg-model').value = params.get('ftout');
  $('gg-prompt').value = 'document ';
  $('gg-maxnew').value = 12; $('gg-device').value = 'cpu';
  $('gg-start').click();
  await until(() => finished('gg-job'), 'the generation to finish');
  check(st('gg-job') === 'done', `generate: ${st('gg-job')} ${$('gg-job').textContent.slice(-300)}`);
  check($('gg-output').textContent.includes('document'), `output: ${$('gg-output').textContent}`);
  check(!/^device:/m.test($('gg-output').textContent), 'status lines are kept out of the text box');
  steps.push('generate');
}
