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

pub(super) struct P<'a> {
    pub(super) b: &'a [u8],
    pub(super) i: usize,
}
