use super::*;
use std::path::Path;
use std::process::Command;

/// The script's "Largest model by memory" table: `(memory GB, [discrete/unified text] x 4)`.
/// One row of the script's table: the memory and the four `discrete / unified` cell pairs.
type ScriptRow = (f64, Vec<(String, String)>);

fn script_rows() -> Option<Vec<ScriptRow>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    let out = Command::new("python3")
        .args(["-I", "scripts/capability-estimates.py"])
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    let start = text.find("### Largest model by memory")?;
    let end = text[start..]
        .find("Each cell is")
        .map_or(text.len(), |e| start + e);
    let mut rows = vec![];
    for line in text[start..end]
        .lines()
        .filter(|l| l.starts_with("| ") && l.contains(" GB |"))
    {
        let cells: Vec<&str> = line.trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() < 6 {
            continue;
        }
        let gb: f64 = cells[0].trim_end_matches(" GB").parse().ok()?;
        let pair = |c: &str| {
            let (d, u) = c.split_once(" / ").unwrap();
            (d.to_string(), u.to_string())
        };
        rows.push((gb, cells[2..6].iter().map(|c| pair(c)).collect()));
    }
    Some(rows)
}

mod bytes_per_weight_follow_the_block_layouts;
mod the_json_has_what_the_page_reads;
