// Copy a GPU buffer back to the host.

/** `n` f32s from the start of `buf`; throws if the device was lost meanwhile. */
export async function readF32(gpu, buf, n) {
  const { device } = gpu;
  const rb = device.createBuffer({ size: Math.max(n * 4, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0, n * 4));
  rb.unmap(); rb.destroy();
  if (gpu.lost) throw new Error(`GPU device lost: ${gpu.lost}`);
  return out;
}
