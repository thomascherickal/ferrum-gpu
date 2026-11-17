// Shared page helpers: HTML escaping and JSON calls to the local server. `fetch` is a parameter so
// the logic can be tested under node without a server.

export const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** GET or send `opts`; JSON bodies are parsed, anything else is returned as an ArrayBuffer. */
export async function api(path, opts = {}, f = globalThis.fetch) {
  const r = await f(path, opts);
  const type = (r.headers && r.headers.get('content-type')) || '';
  const data = type.includes('json') ? await r.json() : await r.arrayBuffer();
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}

export const post = (path, body, f) =>
  api(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }, f);

/** `path` with `params` as a query string; undefined and null values are left out. */
export function withQuery(path, params = {}) {
  const q = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');
  return q ? `${path}?${q}` : path;
}

/** Bytes as `1.5 KiB`, `2.0 MiB`...; whole bytes below 1 KiB. */
export function humanBytes(n) {
  if (!Number.isFinite(n) || n < 0) return '–';
  if (n < 1024) return `${n} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(1)} ${units[i]}`;
}

/** Seconds as `42s`, `3m 05s` or `1h 02m`. */
export function humanTime(s) {
  if (!Number.isFinite(s) || s < 0) return '–';
  const t = Math.round(s);
  if (t < 60) return `${t}s`;
  if (t < 3600) return `${Math.floor(t / 60)}m ${String(t % 60).padStart(2, '0')}s`;
  return `${Math.floor(t / 3600)}h ${String(Math.floor((t % 3600) / 60)).padStart(2, '0')}m`;
}
