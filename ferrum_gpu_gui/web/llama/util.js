// Small helpers the Llama browser runtime shares.

/** Throw an Error with message `m`. */
export const fail = (m) => { throw new Error(m); };

/** Transpose a row-major `rows` x `cols` matrix. */
export function transpose(a, rows, cols) {
  const out = new Float32Array(a.length);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = a[r * cols + c];
  return out;
}
