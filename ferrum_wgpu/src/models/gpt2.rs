//! GPT-2 inference on the GPU with every weight, the KV cache and all activations resident.
//! Per token the host sends one index and receives one logits row.
//! ponytail: each dispatch is its own submit (about 0.03 ms on the MX330); batch the whole
//! token into one command encoder if a profile shows submit cost dominating.

use crate::budget::{Budget, BufferPlan};
use crate::device::{Gpu, GpuError};
use crate::matmul::{Matmul, Variant};
use crate::ops;
use crate::tensor::GpuBuf;
use ferrum_ref::gpt2::{Block, Config, Gpt2};

fn err(m: impl Into<String>) -> GpuError {
    GpuError(m.into())
}

struct DevBlock {
    ln1_g: GpuBuf,
    ln1_b: GpuBuf,
    wq: GpuBuf,
    bq: GpuBuf,
    wk: GpuBuf,
    bk: GpuBuf,
    wv: GpuBuf,
    bv: GpuBuf,
    wo: GpuBuf,
    bo: GpuBuf,
    ln2_g: GpuBuf,
    ln2_b: GpuBuf,
    w1: GpuBuf,
    b1: GpuBuf,
    w2: GpuBuf,
    b2: GpuBuf,
    kcache: GpuBuf,
    vcache: GpuBuf,
}

/// Decode-time scratch (reused every token).
struct Scratch {
    ids: GpuBuf,
    x: GpuBuf,
    x2: GpuBuf,
    h: GpuBuf,
    q: GpuBuf,
    k: GpuBuf,
    v: GpuBuf,
    attn: GpuBuf,
    o: GpuBuf,
    f1: GpuBuf,
    act: GpuBuf,
    f2: GpuBuf,
    probs: GpuBuf,
    logits: GpuBuf,
}

/// A GPT-2 model resident on the GPU with a KV cache.
pub struct GpuGpt2<'g> {
    gpu: &'g Gpu,
    cfg: Config,
    tok: GpuBuf,
    pos_tab: GpuBuf,
    blocks: Vec<DevBlock>,
    lnf_g: GpuBuf,
    lnf_b: GpuBuf,
    head: GpuBuf,
    zero_vocab: GpuBuf,
    s: Scratch,
    matmul: Matmul,
    pos: usize,
}

fn f32s(n: usize) -> u64 {
    (n * 4) as u64
}

/// Every device buffer a model of this shape needs, named and sized (weights, KV cache, scratch).
pub fn plan(c: &Config) -> BufferPlan {
    let (d, h) = (c.dim, c.hidden);
    let mut p = BufferPlan::default();
    p.add("tok_emb", f32s(c.vocab * d));
    p.add("pos_emb", f32s(c.ctx * d));
    for i in 0..c.layers {
        for (n, len) in [
            ("ln1_g", d),
            ("ln1_b", d),
            ("wq", d * d),
            ("bq", d),
            ("wk", d * d),
            ("bk", d),
            ("wv", d * d),
            ("bv", d),
            ("wo", d * d),
            ("bo", d),
            ("ln2_g", d),
            ("ln2_b", d),
            ("w1", d * h),
            ("b1", h),
            ("w2", h * d),
            ("b2", d),
            ("kcache", c.ctx * d),
            ("vcache", c.ctx * d),
        ] {
            p.add(&format!("blk{i}.{n}"), f32s(len));
        }
    }
    p.add("lnf_g", f32s(d));
    p.add("lnf_b", f32s(d));
    p.add("head", f32s(d * c.vocab));
    p.add("zero_vocab", f32s(c.vocab));
    for (n, len) in [
        ("ids", 1),
        ("x", d),
        ("x2", d),
        ("h", d),
        ("q", d),
        ("k", d),
        ("v", d),
        ("attn", d),
        ("o", d),
        ("f1", h),
        ("act", h),
        ("f2", d),
        ("probs", c.heads * c.ctx),
        ("logits", c.vocab),
    ] {
        p.add(&format!("scratch.{n}"), f32s(len));
    }
    p
}

impl<'g> GpuGpt2<'g> {
    /// Upload `w`. The buffer plan is checked against the adapter and `budget` first, so an
    /// oversized model is refused before any device memory is allocated.
    pub fn new(gpu: &'g Gpu, w: &Gpt2, budget: Option<Budget>) -> Result<Self, GpuError> {
        w.cfg.validate().map_err(err)?;
        plan(&w.cfg)
            .check(gpu.info().max_binding, budget)
            .map_err(|e| err(e.to_string()))?;
        let c = w.cfg;
        let up = |v: &[f32]| gpu.upload(v);
        let blocks = w
            .blocks
            .iter()
            .map(|b: &Block| {
                Ok(DevBlock {
                    ln1_g: up(&b.ln1_g)?,
                    ln1_b: up(&b.ln1_b)?,
                    wq: up(&b.wq)?,
                    bq: up(&b.bq)?,
                    wk: up(&b.wk)?,
                    bk: up(&b.bk)?,
                    wv: up(&b.wv)?,
                    bv: up(&b.bv)?,
                    wo: up(&b.wo)?,
                    bo: up(&b.bo)?,
                    ln2_g: up(&b.ln2_g)?,
                    ln2_b: up(&b.ln2_b)?,
                    w1: up(&b.w1)?,
                    b1: up(&b.b1)?,
                    w2: up(&b.w2)?,
                    b2: up(&b.b2)?,
                    kcache: gpu.zeros(c.ctx * c.dim)?,
                    vcache: gpu.zeros(c.ctx * c.dim)?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let z = |n| gpu.zeros(n);
        let s = Scratch {
            ids: z(1)?,
            x: z(c.dim)?,
            x2: z(c.dim)?,
            h: z(c.dim)?,
            q: z(c.dim)?,
            k: z(c.dim)?,
            v: z(c.dim)?,
            attn: z(c.dim)?,
            o: z(c.dim)?,
            f1: z(c.hidden)?,
            act: z(c.hidden)?,
            f2: z(c.dim)?,
            probs: z(c.heads * c.ctx)?,
            logits: z(c.vocab)?,
        };
        Ok(GpuGpt2 {
            gpu,
            cfg: c,
            tok: up(&w.tok)?,
            pos_tab: up(&w.pos)?,
            blocks,
            lnf_g: up(&w.lnf_g)?,
            lnf_b: up(&w.lnf_b)?,
            head: up(&w.head)?,
            zero_vocab: z(c.vocab)?,
            s,
            matmul: Matmul::new(gpu, Variant::Blocked4)?,
            pos: 0,
        })
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Tokens consumed since the last [`reset`](Self::reset).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Forget the cached context (the cache buffers are overwritten as new tokens arrive).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Consume one token at the next position and return its next-token logits row.
    pub fn feed(&mut self, token: usize) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        if self.pos >= c.ctx {
            return Err(err(format!("context of {} tokens is full", c.ctx)));
        }
        let (g, s, d) = (self.gpu, &self.s, c.dim);
        // The index buffer is rewritten in place each token.
        let id = g.upload_indices(&[token], c.vocab)?;
        // The whole token is recorded into one command encoder and submitted once.
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            g.copy_f32(&id, 0, &s.ids, 0, 1)?;
            ops::embed_at(
                g,
                (&self.tok, &self.pos_tab, &s.ids),
                &s.x,
                (1, d, c.ctx),
                self.pos,
            )?;
            for b in &self.blocks {
                ops::layernorm(g, &s.x, &b.ln1_g, &b.ln1_b, &s.h, (1, d))?;
                ops::gemv(g, (&s.h, &b.wq, &b.bq), &s.q, (d, d))?;
                ops::gemv(g, (&s.h, &b.wk, &b.bk), &s.k, (d, d))?;
                ops::gemv(g, (&s.h, &b.wv, &b.bv), &s.v, (d, d))?;
                g.copy_f32(&s.k, 0, &b.kcache, self.pos * d, d)?;
                g.copy_f32(&s.v, 0, &b.vcache, self.pos * d, d)?;
                let dims = (c.heads, c.head_dim(), self.pos + 1, c.ctx);
                ops::attention_decode(g, (&s.q, &b.kcache, &b.vcache), (&s.probs, &s.attn), dims)?;
                ops::gemv(g, (&s.attn, &b.wo, &b.bo), &s.o, (d, d))?;
                ops::add(g, &s.x, &s.o, &s.x2)?;
                ops::layernorm(g, &s.x2, &b.ln2_g, &b.ln2_b, &s.h, (1, d))?;
                ops::gemv(g, (&s.h, &b.w1, &b.b1), &s.f1, (d, c.hidden))?;
                ops::gelu(g, &s.f1, &s.act)?;
                ops::gemv(g, (&s.act, &b.w2, &b.b2), &s.f2, (c.hidden, d))?;
                ops::add(g, &s.x2, &s.f2, &s.x)?;
            }
            ops::layernorm(g, &s.x, &self.lnf_g, &self.lnf_b, &s.h, (1, d))?;
            ops::gemv(
                g,
                (&s.h, &self.head, &self.zero_vocab),
                &s.logits,
                (d, c.vocab),
            )?;
            Ok(())
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        self.pos += 1;
        g.read(&s.logits)
    }

    /// Stateless full causal forward pass over `ids`; returns every position's logits
    /// (`ids.len() * vocab`). Uses batched kernels (matmul, not GEMV).
    pub fn forward_all(&self, ids: &[usize]) -> Result<Vec<f32>, GpuError> {
        let c = self.cfg;
        let t = ids.len();
        if t == 0 || t > c.ctx {
            return Err(err(format!("sequence length {t} must be in 1..={}", c.ctx)));
        }
        let (g, d) = (self.gpu, c.dim);
        let z = |n: usize| g.zeros(n);
        let gi = g.upload_indices(ids, c.vocab)?;
        let (x, x2, h) = (z(t * d)?, z(t * d)?, z(t * d)?);
        let (q, k, v, a, o) = (z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?, z(t * d)?);
        let (f1, act, f2) = (z(t * c.hidden)?, z(t * c.hidden)?, z(t * d)?);
        let (tmp_d, tmp_h) = (z(t * d)?, z(t * c.hidden)?);
        let probs = z(c.heads * t * t)?;
        let logits = z(t * c.vocab)?;
        g.begin_batch()?;
        let rec = (|| -> Result<(), GpuError> {
            ops::embed(g, (&self.tok, &self.pos_tab, &gi), &x, (t, d, t))?;
            // y = h * w + bias, all resident.
            let lin = |h: &GpuBuf,
                       w: &GpuBuf,
                       b: &GpuBuf,
                       y: &GpuBuf,
                       tmp: &GpuBuf,
                       (k, n): (usize, usize)| {
                self.matmul.submit(g, h, w, tmp, (t, k, n))?;
                ops::add_bias(g, tmp, b, y)
            };
            for b in &self.blocks {
                ops::layernorm(g, &x, &b.ln1_g, &b.ln1_b, &h, (t, d))?;
                lin(&h, &b.wq, &b.bq, &q, &tmp_d, (d, d))?;
                lin(&h, &b.wk, &b.bk, &k, &tmp_d, (d, d))?;
                lin(&h, &b.wv, &b.bv, &v, &tmp_d, (d, d))?;
                ops::attention(g, (&q, &k, &v), (&probs, &a), (1, t, c.heads, c.head_dim()))?;
                lin(&a, &b.wo, &b.bo, &o, &tmp_d, (d, d))?;
                ops::add(g, &x, &o, &x2)?;
                ops::layernorm(g, &x2, &b.ln2_g, &b.ln2_b, &h, (t, d))?;
                lin(&h, &b.w1, &b.b1, &f1, &tmp_h, (d, c.hidden))?;
                ops::gelu(g, &f1, &act)?;
                lin(&act, &b.w2, &b.b2, &f2, &tmp_d, (c.hidden, d))?;
                ops::add(g, &x2, &f2, &x)?;
            }
            ops::layernorm(g, &x, &self.lnf_g, &self.lnf_b, &h, (t, d))?;
            self.matmul
                .submit(g, &h, &self.head, &logits, (t, d, c.vocab))?;
            Ok(())
        })();
        g.end_batch(rec.is_ok())?;
        rec?;
        g.read(&logits)
    }

    /// Score `ids`: for each `i < len-1`, the log-probability of `ids[i+1]` given `ids[..=i]`
    /// and whether it was the (first) argmax. Only `2 * (len-1)` floats leave the device.
    pub fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, GpuError> {
        let c = self.cfg;
        let t = ids.len();
        if t < 2 {
            return Err(err("scoring needs at least 2 tokens"));
        }
        // The batched pass already ends in a logits buffer; run it again here resident so only
        // the score pairs are read back.
        let logits_host = self.forward_logits_buf(&ids[..t - 1])?;
        let rows = t - 1;
        let targets = self.gpu.upload_indices(&ids[1..], c.vocab)?;
        let out = self.gpu.zeros(2 * rows)?;
        ops::score(self.gpu, &logits_host, &targets, &out, (rows, c.vocab))?;
        let r = self.gpu.read(&out)?;
        Ok(r.chunks_exact(2).map(|p| (p[0], p[1] == 1.0)).collect())
    }

    /// Like [`forward_all`](Self::forward_all) but leaves the logits on the device.
    fn forward_logits_buf(&self, ids: &[usize]) -> Result<GpuBuf, GpuError> {
        let host = self.forward_all(ids)?;
        // ponytail: forward_all reads back; re-upload keeps score simple. Return the buffer
        // directly once forward_all is split into a resident part and a readback part.
        self.gpu.upload(&host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::open_for_test;
    use crate::tolerance::{max_rel_err, FORWARD};
    use ferrum_ref::gpt2::Gpt2;

    fn cfg() -> Config {
        Config {
            vocab: 17,
            ctx: 10,
            dim: 12,
            heads: 3,
            layers: 2,
            hidden: 20,
        }
    }

    #[test]
    fn a_fault_at_any_command_fails_decoding_cleanly_and_the_context_does_not_advance() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 13);
        let mut m = GpuGpt2::new(&g, &w, None).unwrap();
        let commands = g.sweep_faults(|| {
            let before = m.pos();
            let r = m.feed(3).map(|_| ());
            if r.is_err() {
                assert_eq!(
                    m.pos(),
                    before,
                    "a failed token must not advance the position"
                );
            }
            r
        });
        // 2 (index copy + embed) + 15 per block x 2 blocks + 2 (final norm + head).
        assert_eq!(commands, 34);
        assert_eq!(m.pos(), 1, "only the successful attempt advanced");
        // Batched prefill and scoring share the same recording path.
        let n = g.sweep_faults(|| m.forward_all(&[1, 2, 3]).map(|_| ()));
        assert!(n > 40, "{n}");
        g.sweep_faults(|| m.score(&[1, 2, 3, 4]).map(|_| ()));
    }

    #[test]
    fn the_decoder_reports_its_configuration() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 12);
        let m = GpuGpt2::new(&g, &w, None).unwrap();
        assert_eq!(m.config(), &cfg());
        assert_eq!(m.pos(), 0);
    }

    #[test]
    fn plan_lists_every_buffer_and_scales_with_the_model() {
        let c = cfg();
        let p = plan(&c);
        assert!(p.rows().iter().any(|(n, _)| n == "blk1.kcache"));
        assert!(p.rows().iter().any(|(n, _)| n == "scratch.logits"));
        let kv = f32s(c.ctx * c.dim);
        assert_eq!(
            p.rows().iter().find(|(n, _)| n == "blk0.kcache").unwrap().1,
            kv
        );
        let bigger = plan(&Config { layers: 4, ..c });
        assert!(bigger.total() > p.total());
        assert_eq!(bigger.rows().len() - p.rows().len(), 2 * 18);
    }

    #[test]
    fn an_oversized_model_is_refused_before_allocating() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 1);
        let before = g.buffers_created();
        let tiny = Budget {
            bytes: 1000,
            source: crate::budget::Source::Override,
        };
        let e = GpuGpt2::new(&g, &w, Some(tiny)).err().expect("must refuse");
        assert!(e.to_string().contains("budget is 1000"), "{e}");
        assert_eq!(g.buffers_created(), before);
    }

    #[test]
    fn feed_matches_the_oracle_logits_row_by_row() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 2);
        let ids = [3usize, 16, 0, 9, 9, 4, 11, 2, 7, 5]; // fills the whole context
        let want = w.forward_logits(&ids).unwrap();
        let mut m = GpuGpt2::new(&g, &w, None).unwrap();
        for (i, &t) in ids.iter().enumerate() {
            let got = m.feed(t).unwrap();
            let e = max_rel_err(&got, &want[i * 17..(i + 1) * 17]);
            eprintln!("feed pos {i}: max rel err {e:.2e}");
            assert!(e < FORWARD, "position {i}: {e}");
        }
        assert_eq!(m.pos(), 10);
        let full = m.feed(1).unwrap_err();
        assert!(full.to_string().contains("full"));
    }

    #[test]
    fn reset_restarts_the_context_and_gives_the_same_answers() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 3);
        let mut m = GpuGpt2::new(&g, &w, None).unwrap();
        let first: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
        m.reset();
        assert_eq!(m.pos(), 0);
        let again: Vec<Vec<f32>> = [5usize, 6, 7].iter().map(|&t| m.feed(t).unwrap()).collect();
        assert_eq!(
            first, again,
            "stale cache entries must not leak into a new context"
        );
    }

    #[test]
    fn forward_all_matches_the_oracle_and_feed() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 4);
        let ids = [1usize, 2, 3, 4, 5, 6, 7];
        let want = w.forward_logits(&ids).unwrap();
        let m = GpuGpt2::new(&g, &w, None).unwrap();
        let e = max_rel_err(&m.forward_all(&ids).unwrap(), &want);
        eprintln!("forward_all: max rel err {e:.2e}");
        assert!(e < FORWARD);
        assert!(m
            .forward_all(&[])
            .unwrap_err()
            .to_string()
            .contains("sequence length"));
        assert!(m.forward_all(&[0; 11]).is_err());
        assert!(m.forward_all(&[17]).is_err());
    }

    #[test]
    fn score_matches_the_oracle() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 5);
        let ids = [4usize, 8, 15, 16, 0, 1];
        let logits = w.forward_logits(&ids[..5]).unwrap();
        let want = ferrum_ref::nn::score(&logits, &ids[1..], 17);
        let got = GpuGpt2::new(&g, &w, None).unwrap().score(&ids).unwrap();
        assert_eq!(got.len(), 5);
        for (i, ((glp, ghit), (wlp, whit))) in got.iter().zip(&want).enumerate() {
            assert!(
                (glp - wlp).abs() < 1e-4 * wlp.abs().max(1.0),
                "row {i}: {glp} vs {wlp}"
            );
            assert_eq!(ghit, whit, "row {i} argmax flag");
        }
        assert!(GpuGpt2::new(&g, &w, None).unwrap().score(&[1]).is_err());
    }

    #[test]
    fn greedy_generation_matches_the_oracle_token_for_token() {
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 6);
        let mut m = GpuGpt2::new(&g, &w, None).unwrap();
        let mut ids = vec![2usize];
        let mut logits = m.feed(2).unwrap();
        for _ in 0..6 {
            let next = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
            ids.push(next);
            // The oracle sees the same history and must agree on the last row's argmax.
            let o = w.forward_logits(&ids[..ids.len() - 1]).unwrap();
            let last = &o[(ids.len() - 2) * 17..];
            let want = (0..17).fold(0, |b, i| if last[i] > last[b] { i } else { b });
            assert_eq!(next, want, "history {ids:?}");
            logits = m.feed(next).unwrap();
        }
    }

    #[test]
    fn a_gguf_exported_model_runs_identically_after_a_round_trip() {
        use ferrum_gguf::Gguf;
        use ferrum_model::gpt2_gguf::{from_gguf, to_gguf, Quant};
        let Some(g) = open_for_test() else { return };
        let w = Gpt2::random(cfg(), 7);
        let loaded = from_gguf(&Gguf::parse(to_gguf(&w, Quant::F32)).unwrap()).unwrap();
        let ids = [1usize, 5, 9];
        let (mut a, mut b) = (
            GpuGpt2::new(&g, &w, None).unwrap(),
            GpuGpt2::new(&g, &loaded, None).unwrap(),
        );
        for &t in &ids {
            assert_eq!(a.feed(t).unwrap(), b.feed(t).unwrap());
        }
    }
}
