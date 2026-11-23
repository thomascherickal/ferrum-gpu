use super::rig::Rig;
use ferrum_wgpu::matmul::{Matmul, Variant};
use ferrum_wgpu::ops_bwd as bwd;

/// Forward NN, and the two backward flavours (dx = dy w^T, dw = x^T dy), per matmul shape.
pub(crate) fn rows(rig: &mut Rig) {
    let (r, c, h, v, layers) = (rig.d.rows(), rig.d.c, rig.d.h, rig.d.v, rig.d.layers);
    for (name, m, k, n, per) in [
        ("matmul NN  qkvo   [1024x512x512]", r, c, c, 4 * layers),
        ("matmul NN  ffn up [1024x512x2048]", r, c, h, layers),
        ("matmul NN  ffn dn [1024x2048x512]", r, h, c, layers),
        ("matmul NN  head   [1024x512x4096]", r, c, v, 1),
    ] {
        let (a, w, o) = (rig.buf(m * k), rig.buf(k * n), rig.zeros(m * n));
        let mm = Matmul::new(&rig.g, Variant::Blocked4).unwrap();
        rig.record(name, per, |g| mm.submit(g, &a, &w, &o, (m, k, n)).unwrap());
        // dx: dy[m,n] * w^T -> [m,k]; dw: x^T[k,m] * dy[m,n] -> [k,n]
        let (dy, dx, dw) = (rig.buf(m * n), rig.zeros(m * k), rig.zeros(k * n));
        let named = |from: &str| {
            name.replace("NN ", from).replace(
                &format!("matmul {from}"),
                &format!("matmul {}", from.trim_end()),
            )
        };
        rig.record(&named("dX "), per, |g| {
            bwd::matmul_ex(g, (&dy, &w, &dx), (m, n, k), (false, true)).unwrap()
        });
        rig.record(&named("dW "), per, |g| {
            bwd::matmul_ex(g, (&a, &dy, &dw), (k, m, n), (true, false)).unwrap()
        });
    }
}
