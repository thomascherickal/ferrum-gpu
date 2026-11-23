use super::*;

pub(crate) struct Out {
    pub(crate) b: GgufBuilder,
    pub(crate) quant: Quant,
    pub(crate) skip: Vec<String>,
}

impl Out {
    pub(crate) fn vec(&mut self, name: &str, data: &[f32]) {
        if !self.skip.iter().any(|s| s == name) {
            self.b.tensor_f32(name, &[data.len() as u64], data);
        }
    }

    /// `rows` GGML rows of `ne0` values, quantised per the precision when `ne0 % 32 == 0`.
    pub(crate) fn rows(&mut self, name: &str, data: &[f32], rows: usize, ne0: usize) {
        if self.skip.iter().any(|s| s == name) {
            return;
        }
        let ty = self.quant.tensor_type(ne0);
        let bytes = encode(ty, data).expect("block-aligned by the check above");
        self.b
            .tensor_raw(name, &[ne0 as u64, rows as u64], ty, bytes);
    }

    /// Our `[in, out]` matrix, stored transposed.
    pub(crate) fn matrix(&mut self, name: &str, w: &[f32], inp: usize, out: usize) {
        self.rows(name, &transpose(w, inp, out), out, inp);
    }

    pub(crate) fn bias(&mut self, name: &str, data: &[f32]) {
        if !data.is_empty() {
            self.vec(name, data);
        }
    }
}

/// Metadata entries in file order.
pub(super) type Meta = Vec<(String, Value)>;
