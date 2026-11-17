// Scripted run of the Studio tabs for the headless-browser test: drives the real controls (tabs,
// mode, tool-tips, Datasets, the file picker, Prepare, Tokenize) and returns what happened.
// Parameters come from the query string: file, dir, inputs, out, tok, tokout.

export async function run(params) {
  const $ = (id) => document.getElementById(id);
  const steps = [];
  const fail = (m) => { throw new Error(m); };
  const check = (cond, m) => { if (!cond) fail(m); };
  const until = async (fn, what, ms = 90000) => {
    const t0 = performance.now();
    while (!fn()) {
      if (performance.now() - t0 > ms) fail(`timed out waiting for ${what}`);
      await new Promise((r) => setTimeout(r, 40));
    }
  };
  const key = (el, k) => el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true }));
  const change = (el, v) => { el.value = v; el.dispatchEvent(new Event('change', { bubbles: true })); };
  const selected = () => document.querySelector('[role="tab"][aria-selected="true"]').id;
  const tip = () => $('ferrum-tooltip');
  try {
    // tabs: the new ones exist, and the arrow keys move between them
    const names = [...document.querySelectorAll('[role="tab"]')].map((t) => t.textContent.trim());
    check(names.includes('Datasets') && names.includes('Prepare'), `tabs: ${names}`);
    $('tab-datasets').click();
    $('tab-datasets').focus();
    key($('tab-datasets'), 'ArrowRight');
    check(selected() === 'tab-prepare', `ArrowRight selected ${selected()}`);
    key($('tab-prepare'), 'Home');
    check(selected() === 'tab-datasets', `Home selected ${selected()}`);
    steps.push('tabs');

    // beginner mode hides the advanced tabs and switches the tool-tip text
    change($('ui-mode'), 'beginner');
    check($('tab-prepare').hidden === true, 'Prepare is hidden for beginners');
    check(selected() === 'tab-datasets', 'still on a visible tab');
    $('ds-path').focus();
    await until(() => !tip().hidden, 'the tool-tip');
    check(tip().textContent.includes('stored on this computer'), `beginner tip: ${tip().textContent}`);
    check($('ds-path').getAttribute('aria-describedby') === 'help-ds-path', 'aria-describedby');
    check(document.getElementById('help-ds-path').textContent === tip().textContent, 'the hidden copy matches');
    key(document, 'Escape');
    check(tip().hidden, 'Escape hides the tip');
    $('ds-path').blur();
    change($('ui-mode'), 'expert');
    check($('tab-prepare').hidden === false, 'Prepare is back for experts');
    $('ds-path').focus();
    await until(() => !tip().hidden, 'the expert tool-tip');
    check(tip().textContent.includes('machine running this program'), `expert tip: ${tip().textContent}`);
    $('ds-path').blur();
    steps.push('mode+tips');

    // Datasets: inspect a file, then pick one through the browser
    $('ds-path').value = params.get('file');
    $('ds-inspect').click();
    await until(() => $('ds-summary').textContent.startsWith('File'), 'the file summary');
    check(/line/.test($('ds-summary').textContent), $('ds-summary').textContent);
    check($('ds-preview').textContent.length > 20, 'a preview');
    check($('ds-use').disabled === false, 'Use for training is enabled');
    steps.push('inspect');

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
    check($('ds-path').value === `${params.get('dir')}/corpus.txt`, `picked ${$('ds-path').value}`);
    // picking fires `change`, which inspects again; the button is disabled until that finishes
    await until(() => !$('ds-use').disabled, 'the re-inspect after picking');
    $('ds-use').click();
    check($('ds-summary').textContent.includes('chosen as the training data'), 'chosen');
    steps.push('picker');

    // Prepare: a bad form is refused in the page, a good one runs the real command
    $('tab-prepare').click();
    // "Use for training" filled the inputs with the chosen dataset (by design); empty them to test the refusal
    check($('pr-inputs').value === `${params.get('dir')}/corpus.txt`, `inputs were prefilled with ${$('pr-inputs').value}`);
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

    // Tokenize with the tokenizer a training run saved
    $('tk-data').value = `${params.get('out')}/train.txt`;
    $('tk-tokenizer').value = params.get('tok');
    $('tk-out').value = params.get('tokout');
    $('tk-run').click();
    await until(() => /^(done|error|failed)/.test($('tk-job').querySelector('[role="status"]').textContent), 'tokenize to finish');
    check($('tk-job').querySelector('[role="status"]').textContent === 'done', `tokenize: ${$('tk-job').textContent}`);
    steps.push('tokenize');
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

    // Fine-tune the model Train saved, evaluate the result, and generate from it
    const st = (id) => $(id).querySelector('[role="status"]').textContent;
    const finished = (id) => /^(done|error|failed|stopped)/.test(st(id));
    $('tab-finetune').click();
    $('ft-model').value = params.get('gtout');
    $('ft-data').value = params.get('file');
    $('ft-out').value = params.get('ftout');
    $('ft-lengthkind').value = 'steps'; $('ft-length').value = 10; $('ft-batch').value = 4; $('ft-device').value = 'cpu';
    $('ft-start').click();
    await until(() => finished('ft-job'), 'the fine-tune to finish');
    check(st('ft-job') === 'done', `fine-tune: ${st('ft-job')} ${$('ft-job').textContent.slice(-300)}`);
    steps.push('finetune');

    $('tab-evaluate').click();
    $('ev-model').value = params.get('ftout');
    $('ev-data').value = params.get('file');
    $('ev-start').click();
    await until(() => finished('ev-job'), 'the evaluation to finish');
    check(st('ev-job') === 'done', `evaluate: ${st('ev-job')} ${$('ev-job').textContent.slice(-300)}`);
    check(/perplexity [\d.]+ · loss [\d.]+ · over \d+ tokens/.test($('ev-result').textContent), `result: ${$('ev-result').textContent}`);
    steps.push('evaluate');

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

    // Models: find the models the earlier steps saved, look inside one, filter its tensors, choose it
    $('tab-models').click();
    $('md-dir').value = params.get('dir');
    $('md-scan').click();
    await until(() => [...$('md-list').querySelectorAll('button')].some((b) => b.textContent === 'trained.gguf'), 'the model list');
    check(/models? under/.test($('md-scan-note').textContent), $('md-scan-note').textContent);
    [...$('md-list').querySelectorAll('button')].find((b) => b.textContent === 'trained.gguf').click();
    await until(() => /parameters/.test($('md-summary').textContent), 'the inspection');
    check(/^gpt2/.test($('md-summary').textContent), `summary: ${$('md-summary').textContent}`);
    check($('md-file').value === `${params.get('dir')}/trained.gguf`, 'the file box follows the list');
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
    return { kind: 'studio', pass: true, steps };
  } catch (e) {
    return { kind: 'studio', pass: false, steps, error: String(e.message || e) };
  }
}
