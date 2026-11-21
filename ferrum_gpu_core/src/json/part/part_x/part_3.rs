use super::*;

impl P<'_> {
    pub(crate) fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(
                self.b[self.i],
                b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
            )
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).expect("ascii");
        let n: f64 = s.parse().map_err(|_| format!("bad number '{s}'"))?;
        if !n.is_finite() {
            return Err(format!("number '{s}' is not finite"));
        }
        Ok(Json::Num(n))
    }

    pub(crate) fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| format!("bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(h)
    }

    pub(crate) fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) {
                                if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err("unpaired surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("unpaired surrogate".into());
                                }
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                            char::from_u32(cp).ok_or("invalid code point")?
                        }
                        _ => return Err(format!("bad escape '\\{}'", e as char)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0..=0x1f => return Err("control character in string".into()),
                _ => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| "invalid UTF-8 in string".to_string())
    }
}
