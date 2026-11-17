//! A small JSON reader and writer (std only). The reader is strict about syntax, bounds nesting
//! depth and refuses trailing data, because it parses request bodies from a browser.

use std::fmt::Write;

/// A JSON value. Objects keep insertion order.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// Any number (JSON has one number type).
    Num(f64),
    /// A string.
    Str(String),
    /// An array.
    Arr(Vec<Json>),
    /// An object.
    Obj(Vec<(String, Json)>),
}

/// Deepest nesting the reader accepts.
pub const MAX_DEPTH: usize = 32;

impl Json {
    /// Build an object from `(key, value)` pairs.
    pub fn obj<const N: usize>(pairs: [(&str, Json); N]) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// Build an object from a runtime list of `(key, value)` pairs.
    pub fn obj_vec(pairs: Vec<(&str, Json)>) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// The value of `key` if this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(o) => o.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The number, if this is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// A non-negative whole number that fits `usize`, if this is one.
    pub fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        (n >= 0.0 && n.fract() == 0.0 && n <= 9.0e15).then_some(n as usize)
    }

    /// Parse a complete document.
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = P {
            b: text.as_bytes(),
            i: 0,
        };
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing data at byte {}", p.i));
        }
        Ok(v)
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("invalid literal at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err("nesting is too deep".into());
        }
        self.ws();
        match self.b.get(self.i) {
            None => Err("unexpected end of input".into()),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut o: Vec<(String, Json)> = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(o));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    self.eat(b':')?;
                    o.push((k, self.value(depth + 1)?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(o));
                        }
                        _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
                    }
                }
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(format!("unexpected character at byte {}", self.i)),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
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

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self
            .b
            .get(self.i..self.i + 4)
            .and_then(|h| std::str::from_utf8(h).ok())
            .and_then(|h| u32::from_str_radix(h, 16).ok())
            .ok_or_else(|| format!("bad \\u escape at byte {}", self.i))?;
        self.i += 4;
        Ok(h)
    }

    fn string(&mut self) -> Result<String, String> {
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

impl std::fmt::Display for Json {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Json::Null => f.write_str("null"),
            Json::Bool(b) => write!(f, "{b}"),
            Json::Num(n) if n.is_finite() => write!(f, "{n}"),
            Json::Num(_) => f.write_str("null"), // JSON has no NaN or infinity
            Json::Str(s) => {
                f.write_char('"')?;
                for c in s.chars() {
                    match c {
                        '"' => f.write_str("\\\"")?,
                        '\\' => f.write_str("\\\\")?,
                        '\n' => f.write_str("\\n")?,
                        '\r' => f.write_str("\\r")?,
                        '\t' => f.write_str("\\t")?,
                        c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?,
                        c => f.write_char(c)?,
                    }
                }
                f.write_char('"')
            }
            Json::Arr(a) => {
                f.write_char('[')?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{v}")?;
                }
                f.write_char(']')
            }
            Json::Obj(o) => {
                f.write_char('{')?;
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        f.write_char(',')?;
                    }
                    write!(f, "{}:{v}", Json::Str(k.clone()))?;
                }
                f.write_char('}')
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Json, String> {
        Json::parse(s)
    }

    #[test]
    fn parses_every_value_kind() {
        assert_eq!(p("null"), Ok(Json::Null));
        assert_eq!(p(" true "), Ok(Json::Bool(true)));
        assert_eq!(p("false"), Ok(Json::Bool(false)));
        assert_eq!(p("-12.5e1"), Ok(Json::Num(-125.0)));
        assert_eq!(
            p("\"a\\n\\u00e9\\ud83d\\ude00\""),
            Ok(Json::Str("a\né😀".into()))
        );
        assert_eq!(p("[]"), Ok(Json::Arr(vec![])));
        assert_eq!(p("{}"), Ok(Json::Obj(vec![])));
        let v = p("{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}").unwrap();
        assert_eq!(v.get("c").and_then(Json::as_str), Some("x"));
        assert_eq!(
            v.get("a"),
            Some(&Json::Arr(vec![
                Json::Num(1.0),
                Json::Num(2.0),
                Json::obj([("b", Json::Null)])
            ]))
        );
        assert_eq!(v.get("missing"), None);
        assert_eq!(Json::Null.get("a"), None);
    }

    #[test]
    fn display_round_trips_and_escapes() {
        let v = Json::obj([
            ("s", Json::Str("q\"\\\n\t\r\u{1}é".into())),
            (
                "a",
                Json::Arr(vec![Json::Num(1.5), Json::Bool(false), Json::Null]),
            ),
            ("n", Json::Num(-3.0)),
        ]);
        assert_eq!(p(&v.to_string()), Ok(v.clone()));
        assert!(v.to_string().contains("\\u0001"));
        assert_eq!(Json::Num(f64::NAN).to_string(), "null");
        assert_eq!(Json::Num(f64::INFINITY).to_string(), "null");
        assert_eq!(Json::Num(7.0).to_string(), "7");
    }

    #[test]
    fn malformed_documents_are_errors_not_panics() {
        for bad in [
            "",
            "   ",
            "nul",
            "tru",
            "{",
            "[",
            "[1,",
            "[1 2]",
            "{\"a\"}",
            "{\"a\":}",
            "{1:2}",
            "{\"a\":1,}",
            "\"abc",
            "\"\\x\"",
            "\"\\u12\"",
            "\"\\ud800\"",
            "\"\\ud800\\u0041\"",
            "\"\u{1}\"",
            "1 2",
            "[1]]",
            "-",
            "1e999",
            "--1",
            "@",
            "{\"a\":1 \"b\":2}",
            "\"\\",
            "\"\\ud800\\",
            "[\"a\" \"b\"]",
        ] {
            assert!(p(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn nesting_is_bounded() {
        let ok = format!("{}1{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(p(&ok).is_ok());
        let deep = format!(
            "{}1{}",
            "[".repeat(MAX_DEPTH + 2),
            "]".repeat(MAX_DEPTH + 2)
        );
        assert!(p(&deep).unwrap_err().contains("deep"));
        let huge = "[".repeat(100_000);
        assert!(p(&huge).is_err(), "no stack overflow on hostile depth");
    }

    #[test]
    fn typed_accessors() {
        assert_eq!(Json::Num(3.0).as_usize(), Some(3));
        assert_eq!(Json::Num(3.5).as_usize(), None);
        assert_eq!(Json::Num(-1.0).as_usize(), None);
        assert_eq!(Json::Num(1e300).as_usize(), None);
        assert_eq!(Json::Str("x".into()).as_usize(), None);
        assert_eq!(Json::Str("x".into()).as_f64(), None);
        assert_eq!(Json::Null.as_str(), None);
        assert_eq!(Json::Num(2.5).as_f64(), Some(2.5));
    }
}
