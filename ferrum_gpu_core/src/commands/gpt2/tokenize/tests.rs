use super::*;

#[test]
fn a_folder_that_cannot_be_listed_is_named_and_one_without_files_is_refused() {
    let d = std::env::temp_dir().join(format!("ferrum_tok_inputs_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let path = d.to_string_lossy().into_owned();
    let refuse = |_: &Path| Err(std::io::Error::other("denied"));
    assert_eq!(inputs_via(&path, &refuse), Err(format!("{path}: denied")));
    let empty = inputs_via(&path, &dir_paths).unwrap_err();
    assert_eq!(empty, format!("{path}: the directory holds no files"));
    let file = d.join("a.txt");
    std::fs::write(&file, b"a").unwrap();
    assert_eq!(inputs_via(&path, &dir_paths), Ok(vec![file]));
}
