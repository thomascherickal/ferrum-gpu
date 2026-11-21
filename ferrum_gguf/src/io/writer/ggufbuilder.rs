use super::*;

/// Accumulates metadata and tensors, then serialises one GGUF v3 file.
#[derive(Default)]
pub struct GgufBuilder {
    pub(super) kv: Vec<(String, Value)>,
    pub(super) tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

pub(super) fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub(super) fn tag(v: &Value) -> u32 {
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

pub(super) fn put_value(out: &mut Vec<u8>, v: &Value) {
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
