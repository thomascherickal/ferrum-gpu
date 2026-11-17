// A tiny shared store so one tab can hand a value to another (the dataset chosen in Datasets is
// the training data in Train). Subscribers are called with the new value whenever a key changes.

export function createStore(initial = {}) {
  const values = { ...initial };
  const subs = new Map();
  return {
    get: (k) => values[k],
    set(k, v) {
      if (values[k] === v) return;
      values[k] = v;
      for (const fn of subs.get(k) ?? []) fn(v);
    },
    /** Call `fn(value)` on every change of `k`; returns the function that unsubscribes. */
    on(k, fn) {
      if (!subs.has(k)) subs.set(k, new Set());
      subs.get(k).add(fn);
      return () => subs.get(k).delete(fn);
    },
  };
}

export const store = createStore();
