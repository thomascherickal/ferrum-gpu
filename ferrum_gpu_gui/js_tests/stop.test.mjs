// Node tests of the Generate tab's stop-string helpers (no browser, no server).
// usage: node stop.test.mjs
import assert from 'node:assert/strict';
import { unescapeStop, stopStrings, cutAtStop } from '../web/stop.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };

// unescaping: \n \t \\ only
eq(unescapeStop('a\\nb'), 'a\nb');
eq(unescapeStop('a\\tb'), 'a\tb');
eq(unescapeStop('a\\\\nb'), 'a\\nb');        // an escaped backslash, then a literal n
eq(unescapeStop('a\\xb'), 'a\\xb');          // unknown escapes are left alone
eq(unescapeStop('plain'), 'plain');

// the box: one per line, blank lines dropped, escapes applied
eq(stopStrings(''), []);
eq(stopStrings('.\n\nEND\\n\n'), ['.', 'END\n']);
eq(stopStrings('only'), ['only']);

// cutting
eq(cutAtStop('abc. def', ['def', '.']), 'abc');
eq(cutAtStop('abc', []), 'abc');
eq(cutAtStop('abc', ['zz']), 'abc');
eq(cutAtStop('abc', ['']), 'abc');           // an empty stop never matches
eq(cutAtStop('abc', ['a']), '');
eq(cutAtStop('ab\ncd', ['\n']), 'ab');

console.log(`OK ${checks} checks`);
