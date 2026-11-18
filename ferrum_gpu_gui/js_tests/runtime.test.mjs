// Node tests of the Export and Serve tabs' argument builders and their refusals.
// usage: node runtime.test.mjs
import assert from 'node:assert/strict';
import { exportGgufArgs, exportOllamaArgs, exportLmstudioArgs, serveArgs, allowedTool, isLoopback, toolText } from '../web/tabs/runtime_args.js';

let checks = 0;
const eq = (a, b) => { assert.deepEqual(a, b); checks++; };
const ok = (v, m) => { assert.ok(v, m); checks++; };
const err = (r, needle) => { ok(r.error && r.error.includes(needle), `${JSON.stringify(r)} should say ${needle}`); eq(r.args, undefined); };

// ---- export-gguf
eq(exportGgufArgs({ model: 'a.gguf', out: 'b.gguf' }), { args: ['--model', 'a.gguf', '--out', 'b.gguf'] });
eq(exportGgufArgs({ model: ' a.gguf ', out: ' b.gguf ', quant: 'int4' }).args, ['--model', 'a.gguf', '--out', 'b.gguf', '--quant', 'int4']);
for (const q of ['f32', 'int8', 'int4', 'f16', 'fp4', 'q2']) ok(exportGgufArgs({ model: 'a', out: 'b', quant: q }).args);
err(exportGgufArgs({ out: 'b' }), 'model to export');
err(exportGgufArgs({ model: 'a' }), 'new file');
err(exportGgufArgs({ model: 'a', out: 'a' }), 'different file');
err(exportGgufArgs({ model: 'a', out: 'b', quant: 'q8_0' }), 'int8, int4, f32, f16, fp4, q2');

// ---- export-ollama
eq(exportOllamaArgs({ model: 'a.gguf', name: 'my-model' }), { args: ['--model', 'a.gguf', '--name', 'my-model'] });
eq(exportOllamaArgs({ model: 'a', name: 'me/my-model:latest' }).args.slice(2), ['--name', 'me/my-model:latest']);
eq(exportOllamaArgs({ model: 'a', modelfile: 'out/Modelfile' }).args, ['--model', 'a', '--modelfile-only', 'out/Modelfile']);
eq(exportOllamaArgs({ model: 'a', modelfile: 'M', name: 'ignored' }).args, ['--model', 'a', '--modelfile-only', 'M']);   // writing only the Modelfile needs no name
eq(exportOllamaArgs({ model: 'a', name: 'n', bin: '/usr/local/bin/ollama' }).args.slice(-2), ['--bin', '/usr/local/bin/ollama']);
err(exportOllamaArgs({ name: 'n' }), 'model to export');
err(exportOllamaArgs({ model: 'a' }), 'Name the model');
err(exportOllamaArgs({ model: 'a', name: 'bad name' }), 'letters, digits');
err(exportOllamaArgs({ model: 'a', name: '-lead' }), 'letters, digits');
err(exportOllamaArgs({ model: 'a', name: 'n', bin: '/bin/sh' }), 'ollama program');

// ---- export-lmstudio
eq(exportLmstudioArgs({ model: 'a.gguf' }), { args: ['--model', 'a.gguf'] });
eq(exportLmstudioArgs({ model: 'a', repo: 'me/tiny-gpt2' }).args, ['--model', 'a', '--repo', 'me/tiny-gpt2']);
eq(exportLmstudioArgs({ model: 'a', bin: 'C:\\Users\\x\\.lmstudio\\bin\\lms.exe' }).args.slice(-2), ['--bin', 'C:\\Users\\x\\.lmstudio\\bin\\lms.exe']);
err(exportLmstudioArgs({}), 'model to export');
for (const bad of ['noslash', 'a/b/c', '/b', 'a/', 'a b/c', '../x']) err(exportLmstudioArgs({ model: 'a', repo: bad }), 'publisher/name');
err(exportLmstudioArgs({ model: 'a', bin: 'ollama' }), 'lms program');

// ---- each form accepts only its own tool (the server accepts either name, nothing else)
for (const p of ['ollama', '/usr/bin/ollama', 'ollama.exe']) { ok(allowedTool(p, 'ollama'), p); ok(!allowedTool(p, 'lms'), p); }
for (const p of ['lms', 'C:\\x\\lms.exe']) { ok(allowedTool(p, 'lms'), p); ok(!allowedTool(p, 'ollama'), p); }
for (const p of ['sh', '/bin/sh', '/tmp/ollama2', '', 'ollama/../sh']) { ok(!allowedTool(p, 'ollama'), p); ok(!allowedTool(p, 'lms'), p); }

// ---- serve
eq(serveArgs({ model: 'a.gguf' }), { args: ['--model', 'a.gguf'] });
eq(serveArgs({ model: 'a', port: '8787', token: 's3cret', device: 'cpu', threads: '2' }).args,
  ['--model', 'a', '--port', '8787', '--token', 's3cret', '--device', 'cpu', '--threads', '2']);
eq(serveArgs({ model: 'a', port: '0' }).args, ['--model', 'a', '--port', '0']);
eq(serveArgs({ model: 'a', device: 'auto' }).args, ['--model', 'a']);
eq(serveArgs({ model: 'a', adapter: 'MX330', gpuMem: '1500', device: 'gpu' }).args.slice(2), ['--device', 'gpu', '--gpu-adapter', 'MX330', '--gpu-mem-mb', '1500']);
eq(serveArgs({ model: 'a', host: 'localhost' }).args, ['--model', 'a', '--host', 'localhost']);
eq(serveArgs({ model: 'a', host: '0.0.0.0', token: 't' }).args, ['--model', 'a', '--host', '0.0.0.0', '--token', 't']);
err(serveArgs({}), 'model to serve');
err(serveArgs({ model: 'a', port: '65536' }), 'port');
err(serveArgs({ model: 'a', port: '-1' }), 'port');
err(serveArgs({ model: 'a', port: 'http' }), 'port');
err(serveArgs({ model: 'a', host: '0.0.0.0' }), 'needs a token');             // the same rule the server enforces
err(serveArgs({ model: 'a', host: '192.168.1.5', token: '  ' }), 'needs a token');
err(serveArgs({ model: 'a', host: 'bad host!' }), 'address');
err(serveArgs({ model: 'a', token: 'has space' }), 'spaces');
err(serveArgs({ model: 'a', device: 'tpu' }), 'Device');
err(serveArgs({ model: 'a', gpuMem: '0' }), 'memory budget');
err(serveArgs({ model: 'a', threads: '0' }), 'Threads');
for (const h of ['localhost', '127.0.0.1', '127.1.2.3', '::1', '[::1]']) ok(isLoopback(h), h);
for (const h of ['0.0.0.0', '192.168.0.1', 'example.com', '']) ok(!isLoopback(h), h);

eq(toolText({ name: 'ollama', path: '/x/ollama', note: null }), 'ollama: found at /x/ollama');
eq(toolText({ name: 'lms', path: null, note: 'cannot find lms' }), 'lms: not found (cannot find lms)');
eq(toolText({ name: 'lms', path: null }), 'lms: not found (unknown reason)');

console.log(`OK ${checks} checks`);
