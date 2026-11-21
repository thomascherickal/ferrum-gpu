use super::*;

impl Gpu {
    /// True while a batch is recording.
    pub fn batching(&self) -> bool {
        self.batch
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Copy `n` f32 from `src[src_off..]` to `dst[dst_off..]` on the device (no host round trip).
    pub fn copy_f32(
        &self,
        src: &GpuBuf,
        src_off: usize,
        dst: &GpuBuf,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        if src_off + n > src.len() || dst_off + n > dst.len() {
            return Err(GpuError(format!(
                "copy of {n} elements at {src_off}->{dst_off} exceeds the buffers ({} and {})",
                src.len(),
                dst.len()
            )));
        }
        if n == 0 {
            return Ok(());
        }
        self.record(|enc| {
            enc.copy_buffer_to_buffer(
                &src.buf,
                (src_off * 4) as u64,
                &dst.buf,
                (dst_off * 4) as u64,
                (n * 4) as u64,
            );
        })
    }

    /// Check a plan against the adapter's binding limit and `budget`, then (only if it fits)
    /// allocate one zeroed buffer per row, in order. Rows are rounded up to whole f32s.
    pub fn alloc_plan(
        &self,
        plan: &BufferPlan,
        budget: Option<Budget>,
    ) -> Result<Vec<GpuBuf>, GpuError> {
        plan.check(self.info().max_binding, budget)
            .map_err(|e| GpuError(e.to_string()))?;
        plan.rows()
            .iter()
            .map(|(_, bytes)| self.zeros(bytes.div_ceil(4) as usize))
            .collect()
    }
}
