//! Device activations and scratch for one `(batch, t)` shape of the Llama trainer.

use crate::device::{Gpu, GpuError};
use crate::tensor::GpuBuf;
use ferrum_ref::llama::{rope_tables, Config, RopeScaling};

/// Activations kept for the backward pass (per layer) plus shared scratch.
pub(super) struct Acts {
    pub batch: usize,
    pub t: usize,
    pub x: Vec<GpuBuf>,
    pub h1: Vec<GpuBuf>,
    pub q: Vec<GpuBuf>,
    pub k: Vec<GpuBuf>,
    pub v: Vec<GpuBuf>,
    pub probs: Vec<GpuBuf>,
    pub a: Vec<GpuBuf>,
    pub x2: Vec<GpuBuf>,
    pub h2: Vec<GpuBuf>,
    pub gate: Vec<GpuBuf>,
    pub up: Vec<GpuBuf>,
    pub act: Vec<GpuBuf>,
    pub cos: GpuBuf,
    pub sin: GpuBuf,
    pub neg_sin: GpuBuf,
    pub o: GpuBuf,
    pub f2: GpuBuf,
    pub tmp_q: GpuBuf,
    pub tmp_kv: GpuBuf,
    pub hf: GpuBuf,
    pub logits: GpuBuf,
    pub dlogits: GpuBuf,
    pub row_loss: GpuBuf,
    pub loss: GpuBuf,
    pub dhf: GpuBuf,
    pub dx_a: GpuBuf,
    pub dx_b: GpuBuf,
    pub dact: GpuBuf,
    pub dgate: GpuBuf,
    pub dup: GpuBuf,
    pub dh2a: GpuBuf,
    pub dh2b: GpuBuf,
    pub dh2: GpuBuf,
    pub dx2n: GpuBuf,
    pub dx2: GpuBuf,
    pub da: GpuBuf,
    pub dq: GpuBuf,
    pub dk: GpuBuf,
    pub dv: GpuBuf,
    pub dh1a: GpuBuf,
    pub dh1b: GpuBuf,
    pub dh1c: GpuBuf,
    pub t1: GpuBuf,
    pub dh1: GpuBuf,
    pub dxn: GpuBuf,
    pub stats: GpuBuf,
    pub dot: GpuBuf,
    pub ds: GpuBuf,
    pub junk_pos: GpuBuf,
}

impl Acts {
    pub(super) fn new(
        gpu: &Gpu,
        c: &Config,
        rope: &RopeScaling,
        batch: usize,
        t: usize,
    ) -> Result<Self, GpuError> {
        let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
        let (qd, kvd) = (c.q_dim(), c.kv_dim());
        let z = |n: usize| gpu.zeros(n);
        let per = |n: usize, count: usize| (0..count).map(|_| z(n)).collect::<Result<Vec<_>, _>>();
        // The tables of one sequence repeat for every sequence of the batch (row r uses
        // position r % t), and the backward pass rotates by the opposite angle.
        let (c1, s1) = rope_tables(0, t, c.rope_dim, c.rope_base, rope);
        let neg: Vec<f32> = s1.iter().map(|s| -s).collect();
        let (cos, sin, nsin) = (c1.repeat(batch), s1.repeat(batch), neg.repeat(batch));
        Ok(Acts {
            batch,
            t,
            x: per(r * d, c.layers + 1)?,
            h1: per(r * d, c.layers)?,
            q: per(r * qd, c.layers)?,
            k: per(r * kvd, c.layers)?,
            v: per(r * kvd, c.layers)?,
            probs: per(batch * c.heads * t * t, c.layers)?,
            a: per(r * qd, c.layers)?,
            x2: per(r * d, c.layers)?,
            h2: per(r * d, c.layers)?,
            gate: per(r * h, c.layers)?,
            up: per(r * h, c.layers)?,
            act: per(r * h, c.layers)?,
            cos: gpu.upload(&cos)?,
            sin: gpu.upload(&sin)?,
            neg_sin: gpu.upload(&nsin)?,
            o: z(r * d)?,
            f2: z(r * d)?,
            tmp_q: z(r * qd)?,
            tmp_kv: z(r * kvd)?,
            hf: z(r * d)?,
            logits: z(r * v)?,
            dlogits: z(r * v)?,
            row_loss: z(r)?,
            loss: z(1)?,
            dhf: z(r * d)?,
            dx_a: z(r * d)?,
            dx_b: z(r * d)?,
            dact: z(r * h)?,
            dgate: z(r * h)?,
            dup: z(r * h)?,
            dh2a: z(r * d)?,
            dh2b: z(r * d)?,
            dh2: z(r * d)?,
            dx2n: z(r * d)?,
            dx2: z(r * d)?,
            da: z(r * qd)?,
            dq: z(r * qd)?,
            dk: z(r * kvd)?,
            dv: z(r * kvd)?,
            dh1a: z(r * d)?,
            dh1b: z(r * d)?,
            dh1c: z(r * d)?,
            t1: z(r * d)?,
            dh1: z(r * d)?,
            dxn: z(r * d)?,
            stats: z(r)?,
            dot: z(batch * c.heads * t)?,
            ds: z(batch * c.heads * t * t)?,
            junk_pos: z(c.ctx * d)?,
        })
    }
}
