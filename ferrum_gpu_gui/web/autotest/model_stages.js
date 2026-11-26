// Studio autotest stages: Models and Export. Each stage drives the real controls and throws on a failed check.

export async function models({ $, check, until, steps, params, same }) {
  // Models: find the models the earlier steps saved, look inside one, filter its tensors, choose it
  $('tab-models').click();
  $('md-dir').value = params.get('dir');
  $('md-scan').click();
  await until(() => [...$('md-list').querySelectorAll('button')].some((b) => b.textContent === 'trained.gguf'), 'the model list');
  check(/models? under/.test($('md-scan-note').textContent), $('md-scan-note').textContent);
  [...$('md-list').querySelectorAll('button')].find((b) => b.textContent === 'trained.gguf').click();
  await until(() => /parameters/.test($('md-summary').textContent), 'the inspection');
  check(/^gpt2/.test($('md-summary').textContent), `summary: ${$('md-summary').textContent}`);
  check(same($('md-file').value, `${params.get('dir')}/trained.gguf`), 'the file box follows the list');
  check($('md-tensors').querySelectorAll('tr').length > 5, 'tensors are listed');
  check($('md-meta').textContent.includes('general.architecture'), 'metadata is listed');
  check(/Q8_0|F32/.test($('md-kinds').textContent), `kinds: ${$('md-kinds').textContent}`);
  const all = $('md-tensors').querySelectorAll('tr').length;
  $('md-filter').value = 'blk.0';
  $('md-filter').dispatchEvent(new Event('input', { bubbles: true }));
  const some = $('md-tensors').querySelectorAll('tr').length;
  check(some > 0 && some < all, `the filter narrowed ${all} to ${some}`);
  $('md-file').value = `${params.get('dir')}/tok.gguf`;
  $('md-inspect').click();
  await until(() => $('md-summary').textContent.length > 0 && $('md-summary').textContent !== 'reading…', 'the tokenizer file');
  $('md-file').value = `${params.get('dir')}/corpus.txt`;
  $('md-inspect').click();
  await until(() => $('md-summary').className === 'bad', 'a refusal of a text file');
  check(/magic/.test($('md-summary').textContent), `refusal: ${$('md-summary').textContent}`);
  steps.push('models');
}

export async function exportTools({ $, check, until, steps, params, st, finished }) {
  // Export: the installed tools are listed; another precision is written; an Ollama Modelfile is
  // written without Ollama; a bad LM Studio name is refused in the page
  $('tab-export').click();
  await until(() => /ollama:/.test($('ex-tools').textContent) && /lms:/.test($('ex-tools').textContent), 'the tool list');
  $('ex-model').value = params.get('ftout');
  $('exg-out').value = params.get('exgout'); $('exg-quant').value = 'int4';
  $('exg-run').click();
  await until(() => finished('exg-job'), 'the GGUF export');
  check(st('exg-job') === 'done', `export-gguf: ${st('exg-job')} ${$('exg-job').textContent.slice(-300)}`);
  $('exo-modelfile').value = params.get('exomf');
  $('exo-run').click();
  await until(() => finished('exo-job'), 'the Modelfile export');
  check(st('exo-job') === 'done', `export-ollama: ${st('exo-job')} ${$('exo-job').textContent.slice(-300)}`);
  $('exl-repo').value = 'not a valid name';
  $('exl-run').click();
  check(/publisher\/name/.test(st('exl-job')), `a bad LM Studio name is refused: ${st('exl-job')}`);
  steps.push('export');
}
