use super::*;

/// Turn a wgpu error into our error type.
pub(crate) fn map_error(e: wgpu::Error) -> GpuError {
    match e {
        wgpu::Error::OutOfMemory { .. } => GpuError("GPU out of memory".into()),
        wgpu::Error::Validation { description, .. } => {
            GpuError(format!("GPU validation error: {description}"))
        }
        wgpu::Error::Internal { description, .. } => {
            GpuError(format!("GPU internal error: {description}"))
        }
    }
}

/// Workgroup counts `(x, y)` covering `work = (columns, rows)` with `tile = (columns, rows)`
/// per workgroup, or an error naming the per-dimension limit.
pub fn groups_for(
    work: (usize, usize),
    tile: (u32, u32),
    max: u32,
) -> Result<(u32, u32), GpuError> {
    let x = work.0.div_ceil(tile.0.max(1) as usize);
    let y = work.1.div_ceil(tile.1.max(1) as usize);
    if x > max as usize || y > max as usize {
        return Err(GpuError(format!(
            "dispatch of {}x{} needs {x}x{y} workgroups, adapter allows {max} per dimension",
            work.0, work.1
        )));
    }
    Ok((x as u32, y as u32))
}

/// Workgroups along x for 1-D dispatches. Fixed at the WebGPU baseline limit (not the adapter's
/// larger one) so [`gid`](PRELUDE) can use a constant stride.
pub const GROUPS_X: usize = 65535;
/// Invocations per 1-D workgroup.
pub const WG_1D: usize = 64;

/// WGSL prepended to every shader: the global index of a 1-D kernel dispatched with
/// [`groups_1d`] (rows of `GROUPS_X * WG_1D` invocations stacked along y).
pub const PRELUDE: &str = "fn gid(g: vec3<u32>) -> u32 { return g.y * 4194240u + g.x; }";

/// Workgroup counts `(x, y)` for a 1-D kernel over `n` elements with 64-wide workgroups,
/// spilling into y beyond 65,535 groups. Errors only past `65535 * 65535 * 64` elements.
pub fn groups_1d(n: usize) -> Result<(u32, u32), GpuError> {
    let g = n.div_ceil(WG_1D).max(1);
    let x = g.min(GROUPS_X);
    let y = g.div_ceil(x);
    if y > GROUPS_X {
        return Err(GpuError(format!(
            "{n} elements exceed the 1-D dispatch limit"
        )));
    }
    Ok((x as u32, y as u32))
}
