// Studio autotest stages: tabs, mode and tool-tips. Each stage drives the real controls and throws on a failed check.

export async function tabs({ $, check, steps, key, selected }) {
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
}

export async function modeAndTips({ $, check, until, steps, key, change, selected, tip }) {
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
}
