use super::*;
use crate::serving::decoder::{GenReq, Info};

/// Bytes as tokens, context `ctx`, and a score that depends only on the target token, so the
/// expected sums are easy to write down: log-prob `-(1 + t % 3)`, argmax when `t` is even.
struct Table {
    ctx: usize,
    calls: std::sync::Mutex<Vec<usize>>,
}

impl Table {
    fn new(ctx: usize) -> Self {
        Table {
            ctx,
            calls: Default::default(),
        }
    }
}

impl Decoder for Table {
    fn info(&self) -> Info {
        Info {
            device: "t".into(),
            kind: "cpu".into(),
            arch: "t".into(),
            params: 0,
            vocab: 256,
            ctx: self.ctx,
        }
    }
    fn encode(&self, text: &str) -> Result<Vec<usize>, String> {
        Ok(text.bytes().map(usize::from).collect())
    }
    fn bos(&self) -> usize {
        0
    }
    fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
        Err("unused".into())
    }
    fn score(&self, ids: &[usize]) -> Result<Vec<(f32, bool)>, String> {
        assert!(
            ids.len() <= self.ctx + 1,
            "window of {} for context {}",
            ids.len(),
            self.ctx
        );
        self.calls.lock().unwrap().push(ids.len());
        Ok(ids[1..]
            .iter()
            .map(|&t| (-(1.0 + (t % 3) as f32), t % 2 == 0))
            .collect())
    }
}

fn lp(t: usize) -> f64 {
    -(1.0 + (t % 3) as f64)
}

mod a_decoder_error_is_passed_up;
mod pair_scores_only_the_continuation;
