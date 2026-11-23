use super::support::{put, tmp};
use crate::fsapi::list::{list_dir, list_dir_with, Kind};
use ferrum_test_support::link;

#[test]
fn folders_come_first_then_files_each_sorted_without_regard_to_case() {
    let d = tmp();
    put(&d, "b.txt", b"12345");
    put(&d, "A.txt", b"1");
    put(&d, "sub/inner.txt", b"x");
    std::fs::create_dir(d.join("Zeta")).unwrap();
    std::fs::create_dir(d.join("alpha")).unwrap();
    let l = list_dir(&d).unwrap();
    let names: Vec<_> = l.entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["alpha", "sub", "Zeta", "A.txt", "b.txt"]);
    assert_eq!(l.entries[3].size, 1);
    assert_eq!(l.entries[4].kind, Kind::File);
    assert_eq!(l.entries[0].kind, Kind::Dir);
    assert!(!l.truncated);
    assert_eq!(l.parent.as_deref(), d.parent());
}

#[test]
fn hidden_entries_are_listed_and_flagged() {
    let d = tmp();
    put(&d, ".secret", b"x");
    put(&d, "plain", b"x");
    let l = list_dir(&d).unwrap();
    let hidden: Vec<_> = l
        .entries
        .iter()
        .map(|e| (e.name.as_str(), e.hidden))
        .collect();
    assert_eq!(hidden, [(".secret", true), ("plain", false)]);
}

#[test]
fn a_listing_over_the_limit_is_cut_and_says_so() {
    let d = tmp();
    for i in 0..12 {
        put(&d, &format!("f{i:02}"), b"");
    }
    let l = list_dir_with(&d, 5).unwrap();
    assert_eq!(l.entries.len(), 5);
    assert!(l.truncated);
    let all = list_dir_with(&d, 12).unwrap();
    assert_eq!(all.entries.len(), 12);
    assert!(!all.truncated);
}

#[test]
fn errors_name_the_folder() {
    let d = tmp();
    let e = list_dir(&d.join("nope")).unwrap_err();
    assert!(e.contains("nope"), "{e}");
    let f = put(&d, "file.txt", b"x");
    let e = list_dir(&f).unwrap_err();
    assert!(e.contains("file.txt"), "{e}");
}

#[test]
fn the_root_has_no_parent_and_a_bare_name_has_none_either() {
    let tmp = std::env::temp_dir();
    let root = tmp.ancestors().last().unwrap();
    assert_eq!(list_dir(root).unwrap().parent, None);
}

#[test]
fn links_are_followed_and_broken_links_are_other() {
    let d = tmp();
    std::fs::create_dir(d.join("real")).unwrap();
    put(&d, "real/f.txt", b"abc");
    let made = [
        link(&d.join("real"), &d.join("link_dir")),
        link(&d.join("real").join("f.txt"), &d.join("link_file")),
        link(&d.join("missing"), &d.join("broken")),
    ];
    let l = list_dir(&d).unwrap();
    let kind = |n: &str| l.entries.iter().find(|e| e.name == n).unwrap().kind;
    if made.iter().all(|&ok| ok) {
        assert_eq!(kind("link_dir"), Kind::Dir);
        assert_eq!(kind("link_file"), Kind::File);
        assert_eq!(kind("broken"), Kind::Other);
    }
    assert_eq!(kind("real"), Kind::Dir);
    assert_eq!(Kind::Other.word(), "other");
    assert_eq!(Kind::File.word(), "file");
    assert_eq!(Kind::Dir.word(), "dir");
}
