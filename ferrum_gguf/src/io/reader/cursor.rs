use super::*;

impl<'a> Cur<'a> {
    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| err("unexpected end of file"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    pub(super) fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub(super) fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub(super) fn string(&mut self) -> Result<String, GgufError> {
        let n = narrow(self.u64()?, "string length overflows")?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| err("string is not valid UTF-8"))
    }
    pub(super) fn value(&mut self, ty: u32, depth: u32) -> Result<Value, GgufError> {
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
                let n = narrow(self.u64()?, "array length overflows")?;
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
