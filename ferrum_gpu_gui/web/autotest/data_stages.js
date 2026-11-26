// Studio autotest stages: Datasets, the file picker, Prepare and Tokenize. Each stage drives the real controls and throws on a failed check.

export async function inspect({ $, check, until, steps, params, same }) {
  // Datasets: inspect a file, then pick one through the browser
  $('ds-path').value = params.get('file');
  $('ds-inspect').click();
  await until(() => $('ds-summary').textContent.startsWith('File'), 'the file summary');
  check(/line/.test($('ds-summary').textContent), $('ds-summary').textContent);
  check($('ds-preview').textContent.length > 20, 'a preview');
  check($('ds-use').disabled === false, 'Use for training is enabled');
  steps.push('inspect');
}

export async function picker({ $, check, until, steps, params, change, same }) {
  $('ds-path').value = '';
  const picker = $('ds-path').closest('.picker');
  picker.querySelector('.browse').click();
  await until(() => picker.querySelector('.picker-panel').querySelector('.listing'), 'the first listing');
  // go to the test folder through the breadcrumb-free route: type the folder, reopen
  $('ds-path').value = `${params.get('dir')}/placeholder.txt`;
  picker.querySelector('.browse').click();   // closes
  picker.querySelector('.browse').click();   // reopens at the typed file's folder
  await until(() => [...picker.querySelectorAll('.listing button')].some((b) => b.textContent === 'corpus.txt'), 'corpus.txt in the listing');
  [...picker.querySelectorAll('.listing button')].find((b) => b.textContent === 'corpus.txt').click();
  check(same($('ds-path').value, `${params.get('dir')}/corpus.txt`), `picked ${$('ds-path').value}`);
  // picking fires `change`, which inspects again; the button is disabled until that finishes
  await until(() => !$('ds-use').disabled, 'the re-inspect after picking');
  $('ds-use').click();
  check($('ds-summary').textContent.includes('chosen as the training data'), 'chosen');
  steps.push('picker');
}

export async function prepare({ $, check, until, steps, params, same }) {
  // Prepare: a bad form is refused in the page, a good one runs the real command
  $('tab-prepare').click();
  // "Use for training" filled the inputs with the chosen dataset (by design); empty them to test the refusal
  check(same($('pr-inputs').value, `${params.get('dir')}/corpus.txt`), `inputs were prefilled with ${$('pr-inputs').value}`);
  $('pr-inputs').value = '';
  $('pr-run').click();
  check(/at least one input/.test($('pr-job').querySelector('[role="status"]').textContent), 'empty form refused');
  $('pr-inputs').value = params.get('inputs');
  $('pr-out').value = params.get('out');
  $('pr-dedup').checked = true;
  $('pr-run').click();
  await until(() => /^(done|error|failed)/.test($('pr-job').querySelector('[role="status"]').textContent), 'prepare to finish');
  check($('pr-job').querySelector('[role="status"]').textContent === 'done', `prepare: ${$('pr-job').textContent}`);
  check($('pr-job').querySelector('pre').textContent.length > 10, 'the report was streamed');
  steps.push('prepare');
}

export async function tokenize({ $, check, until, steps, params }) {
  // Tokenize with the tokenizer a training run saved
  $('tk-data').value = `${params.get('out')}/train.txt`;
  $('tk-tokenizer').value = params.get('tok');
  $('tk-out').value = params.get('tokout');
  $('tk-run').click();
  await until(() => /^(done|error|failed)/.test($('tk-job').querySelector('[role="status"]').textContent), 'tokenize to finish');
  check($('tk-job').querySelector('[role="status"]').textContent === 'done', `tokenize: ${$('tk-job').textContent}`);
  steps.push('tokenize');
}
