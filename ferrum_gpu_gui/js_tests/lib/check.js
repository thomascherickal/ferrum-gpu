// The tiny assertion kit every Node test file shares: `eq` (deep equal), `ok`, and `summary`, which
// prints the `OK <n> checks` line the Rust runner looks for.
import assert from 'node:assert/strict';

let checks = 0;
export const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
export const ok = (v, m) => { assert.ok(v, m); checks++; };
/** Count a check made some other way (for example an `assert.rejects`). */
export const counted = () => { checks++; };
export const summary = () => console.log(`OK ${checks} checks`);
