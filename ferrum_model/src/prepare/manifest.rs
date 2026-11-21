use super::*;

/// The `manifest.json` body: the arguments, every input and output pinned by sha256, the reject
/// counts and the train statistics.
pub(super) fn manifest(
    o: &PrepareOptions,
    inputs: Vec<Json>,
    outputs: Json,
    rejected: &BTreeMap<&str, u64>,
    train_stats: Json,
) -> BTreeMap<String, Json> {
    BTreeMap::from([
        (
            "args".to_string(),
            Json::Arr(o.args.iter().cloned().map(Json::Str).collect()),
        ),
        (
            "ferrum_version".into(),
            Json::Str(env!("CARGO_PKG_VERSION").into()),
        ),
        ("inputs".into(), Json::Arr(inputs)),
        ("outputs".into(), outputs),
        (
            "rejected".into(),
            Json::Obj(
                rejected
                    .iter()
                    .map(|(k, n)| (k.to_string(), Json::Num(*n)))
                    .collect(),
            ),
        ),
        ("train_stats".into(), train_stats),
    ])
}
