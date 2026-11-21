// A fake fetch: `routes` maps "METHOD path-with-query" to a handler returning [status, json].
export function fakeFetch(routes) {
  const calls = [];
  const f = async (path, opts = {}) => {
    const key = `${opts.method || 'GET'} ${path}`;
    calls.push({ key, body: opts.body });
    const h = routes[key];
    if (!h) return { ok: false, status: 404, headers: { get: () => 'application/json' }, json: async () => ({ error: `no route ${key}` }) };
    const [status, body] = typeof h === 'function' ? h(opts) : h;
    return { ok: status < 400, status, headers: { get: () => 'application/json' }, json: async () => body };
  };
  f.calls = calls;
  return f;
}
