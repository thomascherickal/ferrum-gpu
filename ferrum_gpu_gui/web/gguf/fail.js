// The one way the GGUF reader refuses a file.
export function fail(msg) { throw new Error(`GGUF: ${msg}`); }
