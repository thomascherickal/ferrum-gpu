//! A tiny JSON value with sorted keys, printed like `serde_json::to_string_pretty`, so manifests
//! are byte-compatible with ferrum-cpu's without a JSON dependency.

use std::collections::BTreeMap;

pub enum Json {
    Num(u64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

pub fn obj<const N: usize>(items: [(&str, Json); N]) -> Json {
    Json::Obj(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl Json {
    pub fn pretty(&self) -> String {
        let mut out = String::new();
        self.write(0, &mut out);
        out
    }
    fn write(&self, depth: usize, out: &mut String) {
        let pad = |n: usize, out: &mut String| out.push_str(&"  ".repeat(n));
        match self {
            Json::Num(n) => out.push_str(&n.to_string()),
            Json::Str(s) => quote(s, out),
            Json::Arr(v) if v.is_empty() => out.push_str("[]"),
            Json::Obj(m) if m.is_empty() => out.push_str("{}"),
            Json::Arr(v) => {
                out.push_str("[\n");
                for (i, x) in v.iter().enumerate() {
                    pad(depth + 1, out);
                    x.write(depth + 1, out);
                    out.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                pad(depth, out);
                out.push(']');
            }
            Json::Obj(m) => {
                out.push_str("{\n");
                for (i, (k, x)) in m.iter().enumerate() {
                    pad(depth + 1, out);
                    quote(k, out);
                    out.push_str(": ");
                    x.write(depth + 1, out);
                    out.push_str(if i + 1 < m.len() { ",\n" } else { "\n" });
                }
                pad(depth, out);
                out.push('}');
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_sorted_keys_two_space_indent_and_escapes() {
        let v = obj([
            (
                "b",
                Json::Arr(vec![Json::Num(1), Json::Str("a\"\\\n\u{1}é".into())]),
            ),
            ("a", obj([])),
            ("c", Json::Arr(vec![])),
        ]);
        assert_eq!(
            v.pretty(),
            "{\n  \"a\": {},\n  \"b\": [\n    1,\n    \"a\\\"\\\\\\n\\u0001é\"\n  ],\n  \"c\": []\n}"
        );
    }
}
