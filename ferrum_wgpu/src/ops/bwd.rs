//! Backward kernels of the GPT-2 stack. Same conventions as [`crate::ops`]: resident buffers,
//! recorded without waiting, every reduction in a fixed ascending order (no atomics).

use crate::device::{Gpu, GpuError};
use crate::runtime::groups_for;
use crate::tensor::GpuBuf;

fn err(m: String) -> GpuError {
    GpuError(m)
}

fn need(name: &str, b: &GpuBuf, want: usize) -> Result<(), GpuError> {
    if b.len() == want {
        Ok(())
    } else {
        Err(err(format!(
            "{name}: buffer holds {} elements, expected {want}",
            b.len()
        )))
    }
}

/// `C[m,n] = op(A) * op(B)` where `a_t` means A is stored `[k, m]` and `b_t` means B is stored `[n, k]`.
pub fn matmul_ex(
    gpu: &Gpu,
    (a, b, c): (&GpuBuf, &GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
    (a_t, b_t): (bool, bool),
) -> Result<(), GpuError> {
    need("matmul_ex a", a, m * k)?;
    need("matmul_ex b", b, k * n)?;
    need("matmul_ex c", c, m * n)?;
    let p = pipe!(gpu, "matmul/matmul_blocked4t")?;
    let groups = groups_for((n, m), (64, 64), gpu.max_workgroups())?;
    let flags = u32::from(a_t) | (u32::from(b_t) << 1);
    gpu.dispatch(
        &p,
        &[a, b, c],
        &[m as u32, k as u32, n as u32, flags],
        groups,
    )
}

/// `out[c] = sum_r x[r, c]` for `x` of `rows x cols`.
pub fn colsum(
    gpu: &Gpu,
    x: &GpuBuf,
    out: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    need("colsum x", x, rows * cols)?;
    need("colsum out", out, cols)?;
    let p = pipe!(gpu, "bwd/colsum")?;
    gpu.dispatch_1d(&p, &[x, out], &[rows as u32, cols as u32], cols)
}

/// Gradient of `y = x w + b` (`x` `[m,k]`, `w` `[k,n]`): fills `dx [m,k]`, `dw [k,n]`, `db [n]`.
pub fn linear_bwd(
    gpu: &Gpu,
    (x, w, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dw, db): (&GpuBuf, &GpuBuf, &GpuBuf),
    (m, k, n): (usize, usize, usize),
) -> Result<(), GpuError> {
    matmul_ex(gpu, (dy, w, dx), (m, n, k), (false, true))?; // dx = dy * w^T
    matmul_ex(gpu, (x, dy, dw), (k, m, n), (true, false))?; // dw = x^T * dy
    colsum(gpu, dy, db, (m, n))
}

/// `df = dact * gelu'(f1)`.
pub fn gelu_bwd(gpu: &Gpu, dact: &GpuBuf, f1: &GpuBuf, df: &GpuBuf) -> Result<(), GpuError> {
    need("gelu_bwd f1", f1, dact.len())?;
    need("gelu_bwd df", df, dact.len())?;
    let p = pipe!(gpu, "bwd/gelu_bwd")?;
    gpu.dispatch_1d(&p, &[dact, f1, df], &[dact.len() as u32], dact.len())
}

/// LayerNorm backward. `stats` is scratch of `2 * rows` (per-row mean and inverse std).
pub fn layernorm_bwd(
    gpu: &Gpu,
    (x, gain, dy): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dx, dgain, dshift): (&GpuBuf, &GpuBuf, &GpuBuf),
    stats: &GpuBuf,
    (rows, cols): (usize, usize),
) -> Result<(), GpuError> {
    for (n, b) in [("x", x), ("dy", dy), ("dx", dx)] {
        need(&format!("layernorm_bwd {n}"), b, rows * cols)?;
    }
    for (n, b) in [("gain", gain), ("dgain", dgain), ("dshift", dshift)] {
        need(&format!("layernorm_bwd {n}"), b, cols)?;
    }
    need("layernorm_bwd stats", stats, 2 * rows)?;
    let u = [rows as u32, cols as u32];
    let p1 = pipe!(gpu, "bwd/ln_bwd_dx")?;
    gpu.dispatch_1d(&p1, &[x, gain, dy, dx, stats], &u, rows)?;
    let p2 = pipe!(gpu, "bwd/ln_bwd_params")?;
    gpu.dispatch_1d(&p2, &[x, dy, stats, dgain, dshift], &u, cols)
}

/// Causal attention backward from the stored probabilities. `dot` (`batch*heads*t`) and `ds`
/// (`batch*heads*t*t`) are scratch.
pub fn attention_bwd(
    gpu: &Gpu,
    (q, k, v, probs, dout): (&GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf, &GpuBuf),
    (dq, dk, dv): (&GpuBuf, &GpuBuf, &GpuBuf),
    (dot, ds): (&GpuBuf, &GpuBuf),
    (batch, t, heads, hd): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let c = heads * hd;
    for (n, b) in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("dout", dout),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    ] {
        need(&format!("attention_bwd {n}"), b, batch * t * c)?;
    }
    need("attention_bwd probs", probs, batch * heads * t * t)?;
    need("attention_bwd dot", dot, batch * heads * t)?;
    need("attention_bwd ds", ds, batch * heads * t * t)?;
    let u = [batch as u32, t as u32, heads as u32, hd as u32];
    let (rows, cells, outs) = (batch * heads * t, batch * heads * t * t, batch * t * c);
    // dP = dO V^T once per cell, then dS = P (dP - rowdot) / sqrt(hd) in place; dQ, dK and dV
    // are plain gathers over dS / P. (Recomputing dP per output element was 64x redundant.)
    let p = pipe!(gpu, "bwd/attn_bwd_dp")?;
    gpu.dispatch_1d(&p, &[dout, v, ds], &u, cells)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dot")?;
    gpu.dispatch_1d(&p, &[probs, ds, dot], &u, rows)?;
    let p = pipe!(gpu, "bwd/attn_bwd_ds")?;
    gpu.dispatch_1d(&p, &[probs, dot, ds], &u, cells)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dv")?;
    gpu.dispatch_1d(&p, &[probs, dout, dv], &u, outs)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dq")?;
    gpu.dispatch_1d(&p, &[ds, k, dq], &u, outs)?;
    let p = pipe!(gpu, "bwd/attn_bwd_dk")?;
    gpu.dispatch_1d(&p, &[ds, q, dk], &u, outs)
}

/// The CSR index the token-embedding gradient needs: `start[v]..start[v+1]` indexes `order`,
/// which lists the rows holding token `v` in ascending row order (a stable counting sort).
pub fn token_csr(ids: &[usize], vocab: usize) -> Result<(Vec<u32>, Vec<u32>), GpuError> {
    if let Some(bad) = ids.iter().find(|&&i| i >= vocab) {
        return Err(err(format!(
            "token {bad} is out of range for vocab {vocab}"
        )));
    }
    let mut start = vec![0u32; vocab + 1];
    for &i in ids {
        start[i + 1] += 1;
    }
    for v in 0..vocab {
        start[v + 1] += start[v];
    }
    let mut next = start.clone();
    let mut order = vec![0u32; ids.len()];
    for (r, &i) in ids.iter().enumerate() {
        order[next[i] as usize] = r as u32;
        next[i] += 1;
    }
    Ok((start, order))
}

/// Embedding gradients from `dx` (`rows x c`): `dtok [vocab x c]` and `dpos [ctx x c]`.
/// `ids` are the host-side token ids (the CSR index is built from them).
pub fn embed_bwd(
    gpu: &Gpu,
    dx: &GpuBuf,
    ids: &[usize],
    (dtok, dpos): (&GpuBuf, &GpuBuf),
    (vocab, c, t, ctx): (usize, usize, usize, usize),
) -> Result<(), GpuError> {
    let rows = ids.len();
    need("embed_bwd dx", dx, rows * c)?;
    need("embed_bwd dtok", dtok, vocab * c)?;
    need("embed_bwd dpos", dpos, ctx * c)?;
    if t == 0 || t > ctx {
        return Err(err(format!(
            "embed_bwd: sequence length {t} must be in 1..={ctx}"
        )));
    }
    let (start, order) = token_csr(ids, vocab)?;
    let (gs, go) = (
        gpu.upload_raw(bytemuck::cast_slice(&start), start.len())?,
        gpu.upload_raw(bytemuck::cast_slice(&order), order.len())?,
    );
    let p = pipe!(gpu, "bwd/embed_bwd_tok")?;
    gpu.dispatch_1d(
        &p,
        &[dx, &gs, &go, dtok],
        &[vocab as u32, c as u32],
        vocab * c,
    )?;
    let p = pipe!(gpu, "bwd/embed_bwd_pos")?;
    gpu.dispatch_1d(
        &p,
        &[dx, dpos],
        &[rows as u32, c as u32, t as u32, ctx as u32],
        ctx * c,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use crate::tolerance::{max_rel_err, REDUCING};
    use ferrum_ref::grad;
    use ferrum_ref::matmul::{matmul, rand_vec};
    use ferrum_ref::nn;

    fn show(name: &str, shape: &str, e: f32, tol: f32) {
        eprintln!("{name} {shape}: max rel err {e:.2e} (tol {tol:.0e})");
        assert!(e < tol, "{name} {shape}: max rel err {e} >= {tol}");
    }

    fn spread(n: usize, seed: u64, by: f32) -> Vec<f32> {
        rand_vec(n, seed).into_iter().map(|v| v * by).collect()
    }

    #[test]
    fn matmul_ex_matches_cpu_for_all_four_transpose_combinations() {
        let Some(g) = open_for_test() else { return };
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (5, 7, 3),
            (17, 33, 15),
            (64, 64, 64),
            (65, 17, 130),
        ] {
            for (at, bt) in [(false, false), (true, false), (false, true), (true, true)] {
                let a = rand_vec(m * k, 1);
                let b = rand_vec(k * n, 2);
                let a_eff = if at {
                    grad::transpose(&a, k, m)
                } else {
                    a.clone()
                };
                let b_eff = if bt {
                    grad::transpose(&b, n, k)
                } else {
                    b.clone()
                };
                let want = matmul(&a_eff, &b_eff, m, k, n).unwrap();
                let (ga, gb, gc) = (
                    g.upload(&a).unwrap(),
                    g.upload(&b).unwrap(),
                    g.zeros(m * n).unwrap(),
                );
                matmul_ex(&g, (&ga, &gb, &gc), (m, k, n), (at, bt)).unwrap();
                show(
                    "matmul_ex",
                    &format!("{m}x{k}x{n} at={at} bt={bt}"),
                    max_rel_err(&g.read(&gc).unwrap(), &want),
                    1e-4,
                );
            }
        }
    }

    #[test]
    fn colsum_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for (rows, cols) in [(1usize, 1usize), (7, 3), (128, 33), (2048, 96)] {
            let x = rand_vec(rows * cols, 3);
            let mut want = vec![0.0f32; cols];
            for r in 0..rows {
                for c in 0..cols {
                    want[c] += x[r * cols + c];
                }
            }
            let (gx, go) = (g.upload(&x).unwrap(), g.zeros(cols).unwrap());
            colsum(&g, &gx, &go, (rows, cols)).unwrap();
            show(
                "colsum",
                &format!("{rows}x{cols}"),
                max_rel_err(&g.read(&go).unwrap(), &want),
                REDUCING,
            );
        }
    }

    #[test]
    fn linear_bwd_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for (m, k, n) in [
            (1usize, 1usize, 1usize),
            (6, 5, 7),
            (33, 20, 17),
            (128, 96, 64),
        ] {
            let (x, w, dy) = (rand_vec(m * k, 4), rand_vec(k * n, 5), rand_vec(m * n, 6));
            let (wdx, wdw, wdb) = grad::linear_bwd(&x, &w, &dy, (m, k, n), 1);
            let (gx, gw, gdy) = (
                g.upload(&x).unwrap(),
                g.upload(&w).unwrap(),
                g.upload(&dy).unwrap(),
            );
            let (gdx, gdw, gdb) = (
                g.zeros(m * k).unwrap(),
                g.zeros(k * n).unwrap(),
                g.zeros(n).unwrap(),
            );
            linear_bwd(&g, (&gx, &gw, &gdy), (&gdx, &gdw, &gdb), (m, k, n)).unwrap();
            let s = format!("{m}x{k}x{n}");
            show(
                "linear_bwd dx",
                &s,
                max_rel_err(&g.read(&gdx).unwrap(), &wdx),
                1e-4,
            );
            show(
                "linear_bwd dw",
                &s,
                max_rel_err(&g.read(&gdw).unwrap(), &wdw),
                1e-4,
            );
            show(
                "linear_bwd db",
                &s,
                max_rel_err(&g.read(&gdb).unwrap(), &wdb),
                REDUCING,
            );
        }
    }

    #[test]
    fn gelu_bwd_matches_cpu_including_saturation() {
        let Some(g) = open_for_test() else { return };
        let mut f1 = spread(1000, 7, 8.0);
        f1.extend([0.0, 9.999, -9.999, 10.0, -10.0, 100.0, -100.0, 1e6, -1e6]);
        let dact = spread(f1.len(), 8, 1.0);
        let want: Vec<f32> = f1
            .iter()
            .zip(&dact)
            .map(|(&f, &d)| d * grad::gelu_grad(f))
            .collect();
        let (gd, gf, go) = (
            g.upload(&dact).unwrap(),
            g.upload(&f1).unwrap(),
            g.zeros(f1.len()).unwrap(),
        );
        gelu_bwd(&g, &gd, &gf, &go).unwrap();
        let got = g.read(&go).unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
        show("gelu_bwd", "mixed", max_rel_err(&got, &want), REDUCING);
    }

    #[test]
    fn layernorm_bwd_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for (rows, cols) in [(1usize, 2usize), (5, 3), (64, 96), (33, 33), (3, 500)] {
            let x = spread(rows * cols, 9, 3.0);
            let (gain, dy) = (rand_vec(cols, 10), rand_vec(rows * cols, 11));
            let (wdx, wdg, wdb) = grad::layernorm_bwd(&x, &gain, &dy, rows, cols);
            let (gx, gg, gdy) = (
                g.upload(&x).unwrap(),
                g.upload(&gain).unwrap(),
                g.upload(&dy).unwrap(),
            );
            let (gdx, gdg, gdb, st) = (
                g.zeros(rows * cols).unwrap(),
                g.zeros(cols).unwrap(),
                g.zeros(cols).unwrap(),
                g.zeros(2 * rows).unwrap(),
            );
            layernorm_bwd(&g, (&gx, &gg, &gdy), (&gdx, &gdg, &gdb), &st, (rows, cols)).unwrap();
            let s = format!("{rows}x{cols}");
            show(
                "layernorm_bwd dx",
                &s,
                max_rel_err(&g.read(&gdx).unwrap(), &wdx),
                REDUCING,
            );
            show(
                "layernorm_bwd dgain",
                &s,
                max_rel_err(&g.read(&gdg).unwrap(), &wdg),
                REDUCING,
            );
            show(
                "layernorm_bwd dshift",
                &s,
                max_rel_err(&g.read(&gdb).unwrap(), &wdb),
                REDUCING,
            );
        }
    }

    #[test]
    fn attention_bwd_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        for shape in [
            (1usize, 1usize, 1usize, 1usize),
            (1, 4, 2, 3),
            (2, 5, 2, 3),
            (1, 17, 4, 8),
            (2, 33, 3, 5),
        ] {
            let (batch, t, heads, hd) = shape;
            let n = batch * t * heads * hd;
            let (q, k, v, dout) = (
                spread(n, 12, 2.0),
                spread(n, 13, 2.0),
                rand_vec(n, 14),
                rand_vec(n, 15),
            );
            let (_, probs) = nn::attention(&q, &k, &v, shape);
            let (wdq, wdk, wdv) = grad::attention_bwd(&q, &k, &v, &probs, &dout, shape);
            let up = |d: &[f32]| g.upload(d).unwrap();
            let (gq, gk, gv, gp, go) = (up(&q), up(&k), up(&v), up(&probs), up(&dout));
            let (gdq, gdk, gdv, dot, ds) = (
                g.zeros(n).unwrap(),
                g.zeros(n).unwrap(),
                g.zeros(n).unwrap(),
                g.zeros(batch * heads * t).unwrap(),
                g.zeros(batch * heads * t * t).unwrap(),
            );
            attention_bwd(
                &g,
                (&gq, &gk, &gv, &gp, &go),
                (&gdq, &gdk, &gdv),
                (&dot, &ds),
                shape,
            )
            .unwrap();
            let s = format!("{shape:?}");
            show(
                "attention_bwd dq",
                &s,
                max_rel_err(&g.read(&gdq).unwrap(), &wdq),
                REDUCING,
            );
            show(
                "attention_bwd dk",
                &s,
                max_rel_err(&g.read(&gdk).unwrap(), &wdk),
                REDUCING,
            );
            show(
                "attention_bwd dv",
                &s,
                max_rel_err(&g.read(&gdv).unwrap(), &wdv),
                REDUCING,
            );
        }
    }

    #[test]
    fn multi_command_backward_ops_report_a_fault_at_each_of_their_commands() {
        let Some(g) = open_for_test() else { return };
        let (m, k, n) = (3usize, 4usize, 5usize);
        let (x, w, dy) = (
            g.zeros(m * k).unwrap(),
            g.zeros(k * n).unwrap(),
            g.zeros(m * n).unwrap(),
        );
        let (dx, dw, db) = (
            g.zeros(m * k).unwrap(),
            g.zeros(k * n).unwrap(),
            g.zeros(n).unwrap(),
        );
        // dx matmul, dw matmul, bias column sum
        assert_eq!(
            g.sweep_faults(|| linear_bwd(&g, (&x, &w, &dy), (&dx, &dw, &db), (m, k, n))),
            3
        );
        // layernorm: input gradient, parameter gradients
        let (rows, cols) = (3usize, 4usize);
        let (lx, lg, ldy) = (
            g.zeros(rows * cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(rows * cols).unwrap(),
        );
        let (ldx, dg, ds, st) = (
            g.zeros(rows * cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(cols).unwrap(),
            g.zeros(2 * rows).unwrap(),
        );
        assert_eq!(
            g.sweep_faults(|| layernorm_bwd(
                &g,
                (&lx, &lg, &ldy),
                (&ldx, &dg, &ds),
                &st,
                (rows, cols)
            )),
            2
        );
        // attention backward: dP, dot, dS, dV, dQ, dK
        let shape = (1usize, 3usize, 2usize, 2usize);
        let nn_ = shape.0 * shape.1 * shape.2 * shape.3;
        let z = |len: usize| g.zeros(len).unwrap();
        let (q, kk, v, dout, dq, dk, dv) = (z(nn_), z(nn_), z(nn_), z(nn_), z(nn_), z(nn_), z(nn_));
        let (probs, dot, dsb) = (
            z(shape.0 * shape.2 * shape.1 * shape.1),
            z(shape.0 * shape.2 * shape.1),
            z(shape.0 * shape.2 * shape.1 * shape.1),
        );
        assert_eq!(
            g.sweep_faults(|| attention_bwd(
                &g,
                (&q, &kk, &v, &probs, &dout),
                (&dq, &dk, &dv),
                (&dot, &dsb),
                shape
            )),
            6
        );
        // embedding gradient: token rows, position rows
        let (edx, dtok, dpos) = (z(2 * 3), z(4 * 3), z(2 * 3));
        assert_eq!(
            g.sweep_faults(|| embed_bwd(&g, &edx, &[1, 2], (&dtok, &dpos), (4, 3, 2, 2))),
            2
        );
        // gelu backward and colsum are single commands
        let (a, b, c) = (z(4), z(4), z(4));
        assert_eq!(g.sweep_faults(|| gelu_bwd(&g, &a, &b, &c)), 1);
    }

    #[test]
    fn token_csr_is_a_stable_counting_sort() {
        let (start, order) = token_csr(&[2, 0, 2, 2, 1, 0], 4).unwrap();
        assert_eq!(start, vec![0, 2, 3, 6, 6]);
        assert_eq!(
            order,
            vec![1, 5, 4, 0, 2, 3],
            "rows of each token stay in ascending order"
        );
        assert_eq!(token_csr(&[], 3).unwrap(), (vec![0, 0, 0, 0], vec![]));
        assert!(token_csr(&[3], 3)
            .unwrap_err()
            .to_string()
            .contains("out of range"));
    }

    #[test]
    fn embedding_grad_with_duplicate_tokens_matches_cpu() {
        let Some(g) = open_for_test() else { return };
        let (vocab, c, t, ctx) = (9usize, 6usize, 4usize, 6usize);
        // Two sequences with many repeated tokens, and token 8 never used.
        let ids = vec![2usize, 2, 5, 2, 0, 5, 5, 2];
        let rows = ids.len();
        let dx = rand_vec(rows * c, 16);
        let mut want_tok = vec![0.0f32; vocab * c];
        let mut want_pos = vec![0.0f32; ctx * c];
        for (r, &id) in ids.iter().enumerate() {
            for i in 0..c {
                want_tok[id * c + i] += dx[r * c + i];
                want_pos[(r % t) * c + i] += dx[r * c + i];
            }
        }
        let (gdx, gt, gp) = (
            g.upload(&dx).unwrap(),
            g.zeros(vocab * c).unwrap(),
            g.zeros(ctx * c).unwrap(),
        );
        embed_bwd(&g, &gdx, &ids, (&gt, &gp), (vocab, c, t, ctx)).unwrap();
        let (got_tok, got_pos) = (g.read(&gt).unwrap(), g.read(&gp).unwrap());
        // Same ascending row order as the CPU loop, so the sums are bit-identical.
        assert_eq!(got_tok, want_tok);
        assert_eq!(got_pos, want_pos);
        assert!(got_tok[8 * c..].iter().all(|&v| v == 0.0));
        assert!(
            got_pos[t * c..].iter().all(|&v| v == 0.0),
            "unused positions get no gradient"
        );
    }

    #[test]
    fn shape_and_range_errors() {
        let Some(g) = open_for_test() else { return };
        let (a, b, c) = (
            g.zeros(4).unwrap(),
            g.zeros(4).unwrap(),
            g.zeros(3).unwrap(),
        );
        assert!(matmul_ex(&g, (&a, &b, &c), (2, 2, 2), (false, false)).is_err());
        assert!(colsum(&g, &a, &c, (2, 3)).is_err());
        assert!(gelu_bwd(&g, &a, &c, &b).is_err());
        let dx = g.zeros(4).unwrap();
        assert!(embed_bwd(&g, &dx, &[1, 9], (&a, &b), (2, 2, 2, 2)).is_err());
        assert!(
            embed_bwd(&g, &dx, &[0, 1], (&a, &b), (2, 2, 3, 2)).is_err(),
            "t above ctx"
        );
    }

    #[test]
    fn backward_kernels_are_bit_identical_across_runs() {
        let Some(g) = open_for_test() else { return };
        let (rows, cols) = (40usize, 24usize);
        let (x, gain, dy) = (
            spread(rows * cols, 17, 2.0),
            rand_vec(cols, 18),
            rand_vec(rows * cols, 19),
        );
        let run = || {
            let (gx, gg, gdy) = (
                g.upload(&x).unwrap(),
                g.upload(&gain).unwrap(),
                g.upload(&dy).unwrap(),
            );
            let (gdx, gdg, gdb, st) = (
                g.zeros(rows * cols).unwrap(),
                g.zeros(cols).unwrap(),
                g.zeros(cols).unwrap(),
                g.zeros(2 * rows).unwrap(),
            );
            layernorm_bwd(&g, (&gx, &gg, &gdy), (&gdx, &gdg, &gdb), &st, (rows, cols)).unwrap();
            (
                g.read(&gdx).unwrap(),
                g.read(&gdg).unwrap(),
                g.read(&gdb).unwrap(),
            )
        };
        assert_eq!(run(), run());
    }
}
