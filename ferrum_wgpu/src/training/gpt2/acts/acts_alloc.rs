use super::*;

impl Acts {
    pub(crate) fn new(gpu: &Gpu, c: &Config, batch: usize, t: usize) -> Result<Acts, GpuError> {
        let (r, d, h, v) = (batch * t, c.dim, c.hidden, c.vocab);
        let z = |n: usize| gpu.zeros(n);
        let per = |n: usize| (0..c.layers).map(|_| z(n)).collect::<Result<Vec<_>, _>>();
        Ok(Acts {
            batch,
            t,
            x: (0..=c.layers).map(|_| z(r * d)).collect::<Result<_, _>>()?,
            h1: per(r * d)?,
            q: per(r * d)?,
            k: per(r * d)?,
            v: per(r * d)?,
            probs: per(batch * c.heads * t * t)?,
            a: per(r * d)?,
            x2: per(r * d)?,
            h2: per(r * d)?,
            f1: per(r * h)?,
            act: per(r * h)?,
            o: z(r * d)?,
            f2: z(r * d)?,
            tmp_c: z(r * d)?,
            tmp_h: z(r * h)?,
            head: crate::training::HeadActs::new(gpu, r, d, v)?,
            dx_a: z(r * d)?,
            dx_b: z(r * d)?,
            dact: z(r * h)?,
            df1: z(r * h)?,
            dh2: z(r * d)?,
            dx2ln: z(r * d)?,
            dx2: z(r * d)?,
            da: z(r * d)?,
            dq: z(r * d)?,
            dk: z(r * d)?,
            dv: z(r * d)?,
            dh1a: z(r * d)?,
            dh1b: z(r * d)?,
            dh1c: z(r * d)?,
            t1: z(r * d)?,
            dh1: z(r * d)?,
            dxln: z(r * d)?,
            stats: z(2 * r)?,
            dot: z(batch * c.heads * t)?,
            ds: z(batch * c.heads * t * t)?,
            junk_v: z(v)?,
        })
    }
}
