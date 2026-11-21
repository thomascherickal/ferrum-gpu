use super::*;

/// The JSON the page reads.
pub fn to_json(i: &Info) -> Json {
    let num = |n: u64| Json::Num(n as f64);
    let opt = |s: &Option<String>| s.clone().map_or(Json::Null, Json::Str);
    Json::obj([
        ("file_size", num(i.file_size)),
        ("version", num(u64::from(i.version))),
        ("architecture", opt(&i.architecture)),
        ("name", opt(&i.name)),
        ("n_tensors", num(i.n_tensors as u64)),
        ("n_kv", num(i.n_kv as u64)),
        ("parameters", num(i.parameters)),
        ("tensor_bytes", num(i.tensor_bytes)),
        (
            "tokenizer",
            i.tokenizer.as_ref().map_or(Json::Null, |t| {
                Json::obj([
                    ("model", Json::Str(t.model.clone())),
                    ("tokens", num(t.tokens as u64)),
                    (
                        "chat_format",
                        t.chat_format.map_or(Json::Null, |f| Json::Str(f.into())),
                    ),
                ])
            }),
        ),
        (
            "kinds",
            Json::Arr(
                i.kinds
                    .iter()
                    .map(|k| {
                        Json::obj([
                            ("kind", Json::Str(k.kind.clone())),
                            ("tensors", num(k.tensors as u64)),
                            ("bytes", num(k.bytes)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "metadata",
            Json::Arr(
                i.metadata
                    .iter()
                    .map(|(k, v)| {
                        Json::obj([
                            ("key", Json::Str(k.clone())),
                            ("value", Json::Str(v.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "tensors",
            Json::Arr(
                i.tensors
                    .iter()
                    .map(|t| {
                        Json::obj([
                            ("name", Json::Str(t.name.clone())),
                            ("dims", Json::Arr(t.dims.iter().map(|d| num(*d)).collect())),
                            ("kind", Json::Str(t.kind.clone())),
                            ("bytes", t.bytes.map_or(Json::Null, num)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("truncated", Json::Bool(i.truncated)),
    ])
}
