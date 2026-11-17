// Path helpers for the file picker. Pure string functions, so they are tested under node. Paths
// from the server use `/` (Unix) or `\` (Windows); these keep whichever the path already uses.

export function separator(path) {
  return path.includes('\\') && !path.includes('/') ? '\\' : '/';
}

/** `dir` and `name` joined with the separator `dir` uses, without doubling it. */
export function join(dir, name) {
  const sep = separator(dir);
  return dir.endsWith(sep) ? dir + name : dir + sep + name;
}

/** The `[label, path]` pairs of a path's breadcrumb bar, root first. */
export function crumbs(path) {
  const sep = separator(path);
  const parts = path.split(sep).filter(Boolean);
  const out = [];
  const lead = path.startsWith(sep) ? sep : '';
  let acc = lead;
  if (lead) out.push([sep, sep]);
  for (const p of parts) {
    acc = acc === '' ? p : join(acc, p);
    out.push([p, acc]);
  }
  return out;
}

/** The file name at the end of a path. */
export function baseName(path) {
  const sep = separator(path);
  const parts = path.split(sep).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : path;
}

/** True if `name` ends with one of `exts` (case-insensitive, with the dot), or no filter is given. */
export function matchesExt(name, exts) {
  if (!exts || !exts.length) return true;
  const lower = name.toLowerCase();
  return exts.some((e) => lower.endsWith(e.toLowerCase()));
}
