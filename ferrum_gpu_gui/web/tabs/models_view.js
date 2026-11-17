// How the Models tab shows a GGUF inspection. Pure text helpers, tested under node.
import { humanBytes } from '../shell/api.js';

/** `4 × 3`: the dimensions of a tensor. */
export const dimsText = (dims) => dims.join(' × ');

/** A parameter count as `76`, `1.5K`, `163M`, `1.1B`. */
export function paramsText(n) {
  if (!Number.isFinite(n) || n < 0) return '–';
  if (n < 1000) return String(n);
  const units = [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [v, u] of units) if (n >= v) return `${(n / v).toFixed(n / v >= 100 ? 0 : 1)}${u}`;
  return String(n);
}

/** `2026-11-17 18:04` (UTC) from seconds since the epoch, or a dash. */
export function timeText(sec) {
  if (!Number.isFinite(sec)) return '–';
  return new Date(sec * 1000).toISOString().slice(0, 16).replace('T', ' ');
}

/** One line about a model: architecture, name, size, parameters and tensor count. */
export function summaryLine(info) {
  const parts = [
    info.architecture ?? 'unknown architecture',
    info.name ?? null,
    `${paramsText(info.parameters)} parameters`,
    `${humanBytes(info.file_size)} file`,
    `${info.n_tensors} tensor${info.n_tensors === 1 ? '' : 's'}`,
    `${info.n_kv} metadata entr${info.n_kv === 1 ? 'y' : 'ies'}`,
  ];
  return parts.filter((p) => p !== null).join(' · ');
}

/** The mix of tensor types as a sentence: `Q8_0 59% · F32 41%` (by bytes), or by count when no sizes are known. */
export function kindsText(kinds) {
  const total = kinds.reduce((s, k) => s + k.bytes, 0);
  if (!kinds.length) return 'no tensors';
  if (total === 0) return kinds.map((k) => `${k.kind} ×${k.tensors}`).join(' · ');
  return kinds.map((k) => `${k.kind} ${Math.round((k.bytes / total) * 100)}%`).join(' · ');
}

/** Does a tensor row match the filter text (a case-insensitive piece of its name or type)? */
export function rowMatches(row, q) {
  const s = String(q ?? '').trim().toLowerCase();
  return s === '' || row.name.toLowerCase().includes(s) || row.kind.toLowerCase().includes(s);
}
