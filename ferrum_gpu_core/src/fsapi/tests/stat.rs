use super::support::{put, tmp};
use crate::fsapi::stat::{stat, stat_with, walk};
use ferrum_test_support::link;

#[test]
fn a_file_a_folder_and_a_missing_path_are_told_apart() {
    let d = tmp();
    let f = put(&d, "a/b.txt", b"hello");
    let s = stat(&f);
    assert_eq!(
        (s.kind, s.bytes, s.files, s.truncated),
        ("file", 5, 1, false)
    );
    put(&d, "a/c.txt", b"12");
    put(&d, "a/deep/d.txt", b"123");
    let s = stat(&d);
    assert_eq!(
        (s.kind, s.bytes, s.files, s.truncated),
        ("dir", 10, 3, false)
    );
    let s = stat(&d.join("nope"));
    assert_eq!((s.kind, s.bytes, s.files), ("missing", 0, 0));
}

#[test]
fn an_empty_folder_has_no_files() {
    let d = tmp();
    let s = stat(&d);
    assert_eq!((s.kind, s.files, s.bytes), ("dir", 0, 0));
}

#[test]
fn the_file_limit_makes_the_totals_lower_bounds_and_says_so() {
    let d = tmp();
    for i in 0..10 {
        put(&d, &format!("f{i}"), b"xx");
    }
    let s = stat_with(&d, (4, 6));
    assert!(s.truncated);
    assert_eq!(s.files, 4);
    assert_eq!(s.bytes, 8);
    assert!(!stat_with(&d, (10, 6)).truncated || stat_with(&d, (10, 6)).files == 10);
}

#[test]
fn the_depth_limit_stops_the_descent_and_says_so() {
    let d = tmp();
    put(&d, "top.txt", b"1");
    put(&d, "l1/l2/l3/deep.txt", b"1");
    let shallow = stat_with(&d, (100, 1));
    assert!(shallow.truncated, "deep.txt is below the limit");
    assert_eq!(shallow.files, 1);
    let deep = stat_with(&d, (100, 6));
    assert!(!deep.truncated);
    assert_eq!(deep.files, 2);
}

#[test]
fn a_folder_that_cannot_be_read_and_a_dangling_link_do_not_stop_the_count() {
    let d = tmp();
    let f = put(&d, "a.txt", b"x");
    // a file where a folder is expected cannot be listed, on any platform
    let mut s = stat_with(&d, (10, 10));
    walk(&f, 0, (10, 10), &mut s);
    assert_eq!((s.files, s.truncated), (1, false));
    if link(&d.join("missing"), &d.join("dangling")) {
        let s = stat(&d);
        assert_eq!((s.files, s.truncated), (1, false));
    }
}
