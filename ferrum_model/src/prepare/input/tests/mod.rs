use super::*;

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrum_prep_in_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn split(path: &Path, sep: &str) -> (Vec<Vec<u8>>, Vec<u64>, String, u64) {
    let (mut docs, mut offs) = (Vec::new(), Vec::new());
    let (mut h, mut total) = (Sha256::new(), 0);
    for_each_raw_doc(path, sep, &mut h, &mut total, |d, o| {
        docs.push(d.to_vec());
        offs.push(o);
        Ok(())
    })
    .unwrap();
    (docs, offs, h.hex(), total)
}

mod a_separator_on_the_chunk_boundary_is_found_once;
mod expand_inputs_walks_skips_hidden_and_own_outputs;
