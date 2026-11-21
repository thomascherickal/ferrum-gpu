// Compile one native WGSL kernel (served by the Rust crate, behind the shared `gid` prelude).

/** The compute pipeline for kernel `path` (`group/name`); `prelude` is the shared WGSL source. */
export async function compileKernel(device, path, prelude) {
  const r = await fetch(`/shaders/${path}.wgsl`);
  if (!r.ok) throw new Error(`kernel ${path} is not served (${r.status})`);
  const module = device.createShaderModule({ code: `${prelude}\n${await r.text()}` });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`kernel ${path}: ${errors.map((m) => `${m.lineNum}: ${m.message}`).join('; ')}`);
  return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
}

export const fetchPrelude = async () => (await fetch('/shaders/prelude.wgsl')).text();
