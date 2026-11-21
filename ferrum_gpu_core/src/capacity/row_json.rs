use super::*;

/// The JSON the page reads for a row.
pub fn row_json(r: &Row) -> Json {
    let infer = Precision::ALL
        .iter()
        .zip(r.infer)
        .map(|(p, v)| (p.name(), v))
        .collect::<Vec<_>>();
    Json::obj([
        ("memory_gb", Json::Num(r.memory_gb)),
        ("usable_gb", Json::Num(r.usable / 1e9)),
        (
            "infer",
            Json::Arr(
                infer
                    .iter()
                    .map(|(n, v)| {
                        Json::obj([
                            ("precision", Json::Str((*n).into())),
                            ("params", Json::Num(*v)),
                            ("text", Json::Str(format_params(*v))),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "train",
            Json::obj([
                ("params", Json::Num(r.train)),
                ("text", Json::Str(format_params(r.train))),
            ]),
        ),
    ])
}
