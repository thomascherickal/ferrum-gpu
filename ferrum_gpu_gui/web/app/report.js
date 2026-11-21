// What every scripted run shares: send a result to the server, wait for a condition.
export async function reportSelfTest(result) {
  await fetch('/api/selftest/result', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(result) });
}

export const until = async (fn, ms = 120000) => {
  const t0 = performance.now();
  while (!fn()) { if (performance.now() - t0 > ms) throw new Error('timed out'); await new Promise((r) => setTimeout(r, 50)); }
};
