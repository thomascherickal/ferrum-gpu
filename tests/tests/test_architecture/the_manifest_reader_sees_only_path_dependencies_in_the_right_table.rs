use super::*;

#[test]
fn the_manifest_reader_sees_only_path_dependencies_in_the_right_table() {
    let m = "[package]\nname = \"x\"\n[dependencies]\na = { path = \"../a\" }\nserde = \"1\"\nb = { path = \"../b\" }\n[dev-dependencies]\nc = { path = \"../c\" }\n";
    assert_eq!(names(m), ["a", "b"]);
    assert!(names("[package]\nname = \"x\"\n").is_empty());
}
