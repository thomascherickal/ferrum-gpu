// Stop strings for the Generate tab: parsing the box, and cutting text for the in-browser runner
// (the server cuts its own output). Pure functions, tested under Node (js_tests/stop.test.mjs).

// \n, \t and \\ are unescaped (as ferrum-cpu's GUI does); any other backslash is kept as typed.
export const unescapeStop = (s) =>
  s.replace(/\\(n|t|\\)/g, (_, c) => (c === 'n' ? '\n' : c === 't' ? '\t' : '\\'));

// One stop string per line; empty lines are ignored.
export const stopStrings = (box) =>
  box.split('\n').filter((l) => l !== '').map(unescapeStop);

// Cut `text` before the earliest stop string (empty strings never match).
export const cutAtStop = (text, stops) => {
  const at = stops.filter((s) => s).map((s) => text.indexOf(s)).filter((i) => i >= 0);
  return at.length ? text.slice(0, Math.min(...at)) : text;
};
