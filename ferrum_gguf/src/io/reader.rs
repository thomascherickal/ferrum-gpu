//! GGUF v3 parser. Hostile-input safe: every read is bounds-checked, every size is checked
//! arithmetic, and nothing is allocated from a header number before it is proven to fit in
//! the file.

use crate::quant::{decode, nbytes};
use crate::{DEFAULT_ALIGNMENT, MAGIC};
use std::collections::HashMap;

/// A parse or decode failure.
#[derive(Debug, PartialEq, Eq)]
pub struct GgufError(pub String);

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf: {}", self.0)
    }
}

impl std::error::Error for GgufError {}

fn err(m: impl Into<String>) -> GgufError {
    GgufError(m.into())
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// u8
    U8(u8),
    /// i8
    I8(i8),
    /// u16
    U16(u16),
    /// i16
    I16(i16),
    /// u32
    U32(u32),
    /// i32
    I32(i32),
    /// f32
    F32(f32),
    /// bool
    Bool(bool),
    /// UTF-8 string
    Str(String),
    /// homogeneous array
    Arr(Vec<Value>),
    /// u64
    U64(u64),
    /// i64
    I64(i64),
    /// f64
    F64(f64),
}

/// One tensor's directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name.
    pub name: String,
    /// GGML dims, innermost (contiguous) first.
    pub dims: Vec<u64>,
    /// ggml element type id.
    pub ggml_type: u32,
    /// Byte offset from the start of the data section.
    pub offset: u64,
}

impl TensorInfo {
    /// Element count (checked).
    pub fn elements(&self) -> Result<usize, GgufError> {
        self.dims.iter().try_fold(1usize, |a, &d| {
            usize::try_from(d)
                .ok()
                .and_then(|d| a.checked_mul(d))
                .ok_or_else(|| err("tensor element count overflows"))
        })
    }
}

/// A parsed GGUF file (keeps the bytes; tensors decode on demand).
pub struct Gguf {
    /// Metadata in file order.
    pub kv: Vec<(String, Value)>,
    /// Tensor directory in file order.
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    data_start: usize,
    bytes: Vec<u8>,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gguf {{ {} keys, {} tensors, {} bytes }}",
            self.kv.len(),
            self.tensors.len(),
            self.bytes.len()
        )
    }
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| err("unexpected end of file"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = usize::try_from(self.u64()?).map_err(|_| err("string length overflows"))?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| err("string is not valid UTF-8"))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<Value, GgufError> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 0 {
                    return Err(err("nested arrays are not supported"));
                }
                let et = self.u32()?;
                let n = usize::try_from(self.u64()?).map_err(|_| err("array length overflows"))?;
                // Every element needs at least one byte, so a count beyond the remaining bytes is hostile.
                if n > self.b.len() - self.p {
                    return Err(err("array length exceeds the file"));
                }
                Value::Arr(
                    (0..n)
                        .map(|_| self.value(et, depth + 1))
                        .collect::<Result<_, _>>()?,
                )
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(err(format!("unknown metadata value type {other}"))),
        })
    }
}

impl Gguf {
    /// Parse a GGUF v3 file.
    pub fn parse(bytes: Vec<u8>) -> Result<Gguf, GgufError> {
        let mut c = Cur { b: &bytes, p: 0 };
        if c.u32()? != MAGIC {
            return Err(err("bad magic: not a GGUF file"));
        }
        let version = c.u32()?;
        if version != 3 {
            return Err(err(format!("unsupported GGUF version {version} (need 3)")));
        }
        let n_tensors = usize::try_from(c.u64()?).map_err(|_| err("tensor count overflows"))?;
        let n_kv = usize::try_from(c.u64()?).map_err(|_| err("metadata count overflows"))?;
        // Each entry takes at least 8 bytes, which bounds what the counts may claim.
        if n_kv > bytes.len() / 8 || n_tensors > bytes.len() / 8 {
            return Err(err("header counts exceed the file size"));
        }
        let mut kv = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            kv.push((key, c.value(ty, 0)?));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            if nd == 0 || nd > 4 {
                return Err(err(format!(
                    "tensor '{name}' has {nd} dims (1 to 4 supported)"
                )));
            }
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }
        let alignment = match kv.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, Value::U32(a))) if *a > 0 => u64::from(*a),
            Some(_) => return Err(err("general.alignment must be a positive u32")),
            None => DEFAULT_ALIGNMENT,
        };
        let data_start = (c.p as u64).div_ceil(alignment) * alignment;
        let data_start = usize::try_from(data_start).map_err(|_| err("data offset overflows"))?;
        if data_start > bytes.len() {
            return Err(err("file ends before the tensor data"));
        }
        let mut index = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            let size = nbytes(t.ggml_type, t.elements()?)?;
            let start = usize::try_from(t.offset)
                .ok()
                .and_then(|o| o.checked_add(data_start));
            if start
                .and_then(|s| s.checked_add(size))
                .is_none_or(|e| e > bytes.len())
            {
                return Err(err(format!(
                    "tensor '{}' extends past the end of the file",
                    t.name
                )));
            }
            if index.insert(t.name.clone(), i).is_some() {
                return Err(err(format!("duplicate tensor name '{}'", t.name)));
            }
        }
        Ok(Gguf {
            kv,
            tensors,
            index,
            data_start,
            bytes,
        })
    }

    /// A metadata value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// An unsigned integer metadata value (any integer width, non-negative).
    pub fn uint(&self, key: &str) -> Result<usize, GgufError> {
        let v = match self.get(key) {
            Some(Value::U8(v)) => u64::from(*v),
            Some(Value::U16(v)) => u64::from(*v),
            Some(Value::U32(v)) => u64::from(*v),
            Some(Value::U64(v)) => *v,
            Some(Value::I32(v)) if *v >= 0 => *v as u64,
            Some(Value::I64(v)) if *v >= 0 => *v as u64,
            Some(_) => return Err(err(format!("'{key}' is not a non-negative integer"))),
            None => return Err(err(format!("missing key '{key}'"))),
        };
        usize::try_from(v).map_err(|_| err(format!("'{key}' is too large")))
    }

    /// An f32 metadata value.
    pub fn float(&self, key: &str) -> Result<f32, GgufError> {
        match self.get(key) {
            Some(Value::F32(v)) => Ok(*v),
            Some(_) => Err(err(format!("'{key}' is not an f32"))),
            None => Err(err(format!("missing key '{key}'"))),
        }
    }

    /// A string metadata value.
    pub fn string(&self, key: &str) -> Result<&str, GgufError> {
        match self.get(key) {
            Some(Value::Str(s)) => Ok(s),
            Some(_) => Err(err(format!("'{key}' is not a string"))),
            None => Err(err(format!("missing key '{key}'"))),
        }
    }

    /// A tensor's directory entry.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    /// The stored bytes of a tensor, undecoded (blocks of Q8_0 / Q4_0 stay packed).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
        let start = self.data_start + t.offset as usize; // proven in range by parse
        Ok(&self.bytes[start..start + nbytes(t.ggml_type, t.elements()?)?])
    }

    /// Decode a tensor to f32 (row-major, GGML dim 0 contiguous).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| err(format!("missing tensor '{name}'")))?;
        decode(t.ggml_type, self.tensor_bytes(name)?, t.elements()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::GgufBuilder;
    use crate::GGML_F32;

    fn sample() -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.meta("general.architecture", Value::Str("gpt2".into()));
        b.meta("gpt2.block_count", Value::U32(2));
        b.meta("x.eps", Value::F32(1e-5));
        b.meta(
            "x.list",
            Value::Arr(vec![Value::Str("a".into()), Value::Str("bc".into())]),
        );
        b.tensor_f32("w", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        b.tensor_f32("b", &[2], &[0.5, -0.5]);
        b.into_bytes()
    }

    #[test]
    fn raw_bytes_are_the_stored_encoding() {
        let g = Gguf::parse(sample()).unwrap();
        let raw = g.tensor_bytes("b").unwrap();
        assert_eq!(
            raw,
            [0.5f32.to_le_bytes(), (-0.5f32).to_le_bytes()].concat()
        );
        assert_eq!(g.tensor_bytes("w").unwrap().len(), 24);
        assert!(g
            .tensor_bytes("nope")
            .unwrap_err()
            .to_string()
            .contains("missing tensor"));
    }

    #[test]
    fn round_trips_metadata_and_tensors() {
        let g = Gguf::parse(sample()).unwrap();
        assert_eq!(g.string("general.architecture").unwrap(), "gpt2");
        assert_eq!(g.uint("gpt2.block_count").unwrap(), 2);
        assert_eq!(g.float("x.eps").unwrap(), 1e-5);
        assert_eq!(
            g.get("x.list"),
            Some(&Value::Arr(vec![
                Value::Str("a".into()),
                Value::Str("bc".into())
            ]))
        );
        assert_eq!(g.tensor("w").unwrap().dims, vec![3, 2]);
        assert_eq!(g.tensor("w").unwrap().ggml_type, GGML_F32);
        assert_eq!(
            g.tensor_f32("w").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(g.tensor_f32("b").unwrap(), vec![0.5, -0.5]);
    }

    #[test]
    fn typed_getters_report_what_is_wrong() {
        let g = Gguf::parse(sample()).unwrap();
        assert!(g
            .uint("missing")
            .unwrap_err()
            .to_string()
            .contains("missing key"));
        assert!(g
            .uint("general.architecture")
            .unwrap_err()
            .to_string()
            .contains("non-negative integer"));
        assert!(g.float("gpt2.block_count").is_err());
        assert!(g.string("gpt2.block_count").is_err());
        assert!(g
            .tensor_f32("nope")
            .unwrap_err()
            .to_string()
            .contains("missing tensor"));
        assert!(g.tensor("nope").is_none());
    }

    #[test]
    fn rejects_bad_headers() {
        let good = sample();
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(Gguf::parse(bad_magic)
            .unwrap_err()
            .to_string()
            .contains("magic"));
        let mut v2 = good.clone();
        v2[4] = 2;
        assert!(Gguf::parse(v2).unwrap_err().to_string().contains("version"));
        assert!(Gguf::parse(vec![]).is_err());
        assert!(Gguf::parse(good[..10].to_vec()).is_err());
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let good = sample();
        // The last tensor ("b") holds 8 real bytes followed by 24 bytes of alignment padding;
        // cutting only padding leaves a valid file, so stop before it.
        for cut in 0..good.len() - 24 {
            assert!(Gguf::parse(good[..cut].to_vec()).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn hostile_counts_do_not_allocate_or_panic() {
        let mut huge_kv = sample();
        huge_kv[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(huge_kv)
            .unwrap_err()
            .to_string()
            .contains("exceed"));
        let mut huge_tensors = sample();
        huge_tensors[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(huge_tensors).is_err());
    }

    #[test]
    fn hostile_array_and_tensor_sizes_are_rejected() {
        let mut b = GgufBuilder::new();
        b.meta("a", Value::Arr(vec![Value::U8(1)]));
        let mut bytes = b.into_bytes();
        // The array length field sits after: header(24) + key(8+1) + type(4) + elem type(4).
        let at = 24 + 9 + 4 + 4;
        bytes[at..at + 8].copy_from_slice(&(1u64 << 60).to_le_bytes());
        assert!(Gguf::parse(bytes)
            .unwrap_err()
            .to_string()
            .contains("array length"));

        let mut t = GgufBuilder::new();
        t.tensor_raw("w", &[1 << 40, 1 << 40], GGML_F32, vec![0; 4]);
        assert!(Gguf::parse(t.into_bytes()).is_err());
        let mut t = GgufBuilder::new();
        t.tensor_raw("w", &[1000], GGML_F32, vec![0; 4]);
        assert!(Gguf::parse(t.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("past the end"));
    }

    #[test]
    fn duplicate_names_and_bad_dims_are_rejected() {
        let mut b = GgufBuilder::new();
        b.tensor_f32("w", &[1], &[1.0]);
        b.tensor_f32("w", &[1], &[2.0]);
        assert!(Gguf::parse(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        let mut z = GgufBuilder::new();
        z.tensor_raw("w", &[], GGML_F32, vec![]);
        assert!(Gguf::parse(z.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("dims"));
    }

    #[test]
    fn custom_alignment_is_honoured() {
        let mut b = GgufBuilder::new();
        b.meta("general.alignment", Value::U32(64));
        b.tensor_f32("w", &[2], &[7.0, 8.0]);
        let g = Gguf::parse(b.into_bytes()).unwrap();
        assert_eq!(g.tensor_f32("w").unwrap(), vec![7.0, 8.0]);
        let mut bad = GgufBuilder::new();
        bad.meta("general.alignment", Value::U32(0));
        assert!(Gguf::parse(bad.into_bytes()).is_err());
    }

    #[test]
    fn debug_output_summarises_without_dumping_bytes() {
        let g = Gguf::parse(sample()).unwrap();
        let s = format!("{g:?}");
        assert!(s.contains("4 keys") && s.contains("2 tensors"), "{s}");
        assert!(s.len() < 100);
    }

    #[test]
    fn nested_arrays_and_unknown_value_types_are_rejected() {
        let mut b = GgufBuilder::new();
        b.meta("a", Value::Arr(vec![Value::Arr(vec![Value::U8(1)])]));
        assert!(Gguf::parse(b.into_bytes())
            .unwrap_err()
            .to_string()
            .contains("nested arrays"));
        let mut b = GgufBuilder::new();
        b.meta("a", Value::U8(1));
        let mut bytes = b.into_bytes();
        // The value type tag follows the header (24 bytes) and the key (8-byte length + "a").
        bytes[33..37].copy_from_slice(&99u32.to_le_bytes());
        assert!(Gguf::parse(bytes)
            .unwrap_err()
            .to_string()
            .contains("unknown metadata value type 99"));
    }

    #[test]
    fn uint_accepts_every_integer_width_and_rejects_negatives() {
        let mut b = GgufBuilder::new();
        for (k, v) in [
            ("u8", Value::U8(7)),
            ("u16", Value::U16(300)),
            ("u32", Value::U32(70_000)),
            ("u64", Value::U64(5_000_000_000)),
            ("i32", Value::I32(12)),
            ("i64", Value::I64(34)),
            ("neg32", Value::I32(-1)),
            ("neg64", Value::I64(-1)),
        ] {
            b.meta(k, v);
        }
        let g = Gguf::parse(b.into_bytes()).unwrap();
        for (k, want) in [
            ("u8", 7usize),
            ("u16", 300),
            ("u32", 70_000),
            ("u64", 5_000_000_000),
            ("i32", 12),
            ("i64", 34),
        ] {
            assert_eq!(g.uint(k).unwrap(), want, "{k}");
        }
        assert!(g.uint("neg32").is_err() && g.uint("neg64").is_err());
    }

    #[test]
    fn element_count_overflow_is_an_error() {
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![u64::MAX, 2],
            ggml_type: GGML_F32,
            offset: 0,
        };
        assert!(t.elements().is_err());
    }
}
