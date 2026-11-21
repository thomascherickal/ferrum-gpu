use super::*;

#[test]
fn the_json_has_what_the_page_reads() {
    let j = row_json(&row(8.0, false));
    assert_eq!(j.get("memory_gb").and_then(Json::as_f64), Some(8.0));
    let Some(Json::Arr(infer)) = j.get("infer") else {
        panic!("{j}")
    };
    let names: Vec<_> = infer
        .iter()
        .map(|v| v.get("precision").and_then(Json::as_str).unwrap())
        .collect();
    assert_eq!(names, ["Q4_0", "Q8_0", "f32"]);
    assert_eq!(infer[0].get("text").and_then(Json::as_str), Some("11.2B"));
    assert_eq!(
        j.get("train")
            .and_then(|t| t.get("text"))
            .and_then(Json::as_str),
        Some("399M")
    );
}

#[test]
fn the_rust_table_equals_the_one_the_python_script_prints_for_every_memory_size() {
    let Some(rows) = script_rows() else {
        eprintln!("SKIP: python3 or scripts/capability-estimates.py is not available");
        return;
    };
    assert!(rows.len() >= 10, "the script table has {} rows", rows.len());
    for (gb, cells) in rows {
        let (d, u) = (row(gb, false), row(gb, true));
        let mine = [
            (format_params(d.infer[0]), format_params(u.infer[0])),
            (format_params(d.infer[1]), format_params(u.infer[1])),
            (format_params(d.infer[2]), format_params(u.infer[2])),
            (format_params(d.train), format_params(u.train)),
        ];
        assert_eq!(
            cells, mine,
            "{gb} GB: the script and the Rust formulas disagree"
        );
    }
}
