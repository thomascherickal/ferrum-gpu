/// One table row: the speed columns are `-` unless the row was measured (`ms` per token).
pub fn print(name: &str, params: &str, label: &str, total: &str, ms: Option<f64>, note: &str) {
    let speed = ms.map_or("- | -".to_string(), |ms| {
        format!("{ms:.1} | {:.1}", 1e3 / ms)
    });
    println!("| {name} | {params} | {label} | {total} | {speed} | {note} |");
}
