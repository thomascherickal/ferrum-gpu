//! GGUF v3 builder (used for fixtures, exports and checkpoints).

use crate::quant::encode;
use crate::reader::Value;
use crate::{DEFAULT_ALIGNMENT, GGML_F32, MAGIC};

/// Accumulates metadata and tensors, then serialises one GGUF v3 file.
#[derive(Default)]
pub struct GgufBuilder {
    kv: Vec<(String, Value)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn tag(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::Str(_) => 8,
        Value::Arr(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::Str(s) => put_str(out, s),
        Value::Arr(a) => {
            out.extend_from_slice(&a.first().map_or(4, tag).to_le_bytes());
            out.extend_from_slice(&(a.len() as u64).to_le_bytes());
            a.iter().for_each(|e| put_value(out, e));
        }
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_bits().to_le_bytes()),
    }
}

impl GgufBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or append) a metadata entry.
    pub fn meta(&mut self, key: &str, v: Value) {
        self.kv.push((key.to_string(), v));
    }

    /// Add an F32 tensor with GGML dims (innermost first).
    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) {
        let bytes = encode(GGML_F32, data).expect("F32 encodes any length");
        self.tensor_raw(name, dims, GGML_F32, bytes);
    }

    /// Add a tensor already encoded as `ggml_type`.
    pub fn tensor_raw(&mut self, name: &str, dims: &[u64], ggml_type: u32, bytes: Vec<u8>) {
        self.tensors
            .push((name.to_string(), dims.to_vec(), ggml_type, bytes));
    }

    /// Serialise the file.
    pub fn into_bytes(self) -> Vec<u8> {
        let mut head = Vec::new();
        head.extend_from_slice(&MAGIC.to_le_bytes());
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        let mut alignment = DEFAULT_ALIGNMENT;
        for (k, v) in &self.kv {
            put_str(&mut head, k);
            head.extend_from_slice(&tag(v).to_le_bytes());
            put_value(&mut head, v);
            if let ("general.alignment", Value::U32(a)) = (k.as_str(), v) {
                alignment = u64::from(*a).max(1);
            }
        }
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (name, dims, ty, bytes) in &self.tensors {
            put_str(&mut head, name);
            head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            dims.iter()
                .for_each(|d| head.extend_from_slice(&d.to_le_bytes()));
            head.extend_from_slice(&ty.to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(bytes);
            let padded = (bytes.len() as u64).div_ceil(alignment) * alignment;
            data.resize(data.len() + (padded - bytes.len() as u64) as usize, 0);
            offset += padded;
        }
        let pad = (head.len() as u64).div_ceil(alignment) * alignment - head.len() as u64;
        head.resize(head.len() + pad as usize, 0);
        head.extend_from_slice(&data);
        head
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::Gguf;
    use crate::{GGML_Q4_0, GGML_Q8_0};

    #[test]
    fn empty_file_parses() {
        let g = Gguf::parse(GgufBuilder::new().into_bytes()).unwrap();
        assert!(g.kv.is_empty() && g.tensors.is_empty());
    }

    #[test]
    fn every_value_type_round_trips() {
        let vals = vec![
            Value::U8(200),
            Value::I8(-5),
            Value::U16(60000),
            Value::I16(-30000),
            Value::U32(4_000_000_000),
            Value::I32(-7),
            Value::F32(1.5),
            Value::Bool(true),
            Value::Str("héllo".into()),
            Value::Arr(vec![Value::I32(1), Value::I32(2)]),
            Value::U64(u64::MAX),
            Value::I64(i64::MIN),
            Value::F64(2.25),
        ];
        let mut b = GgufBuilder::new();
        for (i, v) in vals.iter().enumerate() {
            b.meta(&format!("k{i}"), v.clone());
        }
        let g = Gguf::parse(b.into_bytes()).unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(g.get(&format!("k{i}")), Some(v), "k{i}");
        }
    }

    #[test]
    fn tensors_are_aligned_and_independent() {
        let mut b = GgufBuilder::new();
        b.tensor_f32("a", &[3], &[1.0, 2.0, 3.0]); // 12 bytes, padded to 32
        b.tensor_f32("b", &[1], &[9.0]);
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(g.tensor("a").unwrap().offset, 0);
        assert_eq!(g.tensor("b").unwrap().offset, 32);
        assert_eq!(g.tensor_f32("b").unwrap(), vec![9.0]);
    }

    #[test]
    fn quantised_tensors_load_back_close() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 31.5) / 9.0).collect();
        let mut b = GgufBuilder::new();
        b.tensor_raw("q8", &[32, 2], GGML_Q8_0, encode(GGML_Q8_0, &data).unwrap());
        b.tensor_raw("q4", &[32, 2], GGML_Q4_0, encode(GGML_Q4_0, &data).unwrap());
        let g = Gguf::parse(b.into_bytes()).unwrap();
        let (q8, q4) = (g.tensor_f32("q8").unwrap(), g.tensor_f32("q4").unwrap());
        let err = |q: &[f32]| {
            q.iter()
                .zip(&data)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
        };
        assert!(err(&q8) < 0.03, "q8 err {}", err(&q8));
        assert!(err(&q4) < 0.5, "q4 err {}", err(&q4));
        assert!(err(&q8) < err(&q4));
    }
}
