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
    return { kind: 'studio', pass: true, steps };
  } catch (e) {
    return { kind: 'studio', pass: false, steps, error: String(e.message || e) };
  }
}
