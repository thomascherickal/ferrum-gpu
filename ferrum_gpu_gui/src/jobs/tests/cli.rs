use crate::jobs::cli::{find, NAME};
use std::path::{Path, PathBuf};

#[test]
fn an_explicit_path_wins_and_must_exist() {
    let r = find(
        Some(Path::new("/x/cli")),
        Some(Path::new("/e")),
        Some("/p"),
        &|p| p == Path::new("/x/cli"),
    );
    assert_eq!(r.unwrap(), PathBuf::from("/x/cli"));
    let e = find(Some(Path::new("/x/cli")), None, None, &|_| false).unwrap_err();
    assert!(e.contains("--cli /x/cli"), "{e}");
}

#[test]
fn the_executable_directory_comes_before_path() {
    let any = |p: &Path| p.file_name().is_some_and(|f| f == NAME);
    let r = find(None, Some(Path::new("/exe")), Some("/a:/b"), &any).unwrap();
    assert_eq!(r, Path::new("/exe").join(NAME));
}

#[test]
fn path_entries_are_searched_in_order() {
    let sep = if cfg!(windows) { ';' } else { ':' };
    let want = Path::new("/b").join(NAME);
    let r = find(
        None,
        Some(Path::new("/exe")),
        Some(&format!("/a{sep}/b")),
        &|p| p == want,
    );
    assert_eq!(r.unwrap(), want);
}

#[test]
fn nothing_found_says_what_to_do() {
    let e = find(None, Some(Path::new("/e")), Some("/p"), &|_| false).unwrap_err();
    assert!(e.contains(NAME) && e.contains("--cli"), "{e}");
    assert!(find(None, None, None, &|_| false).is_err());
}

#[test]
fn find_here_reports_a_missing_explicit_path() {
    let e = crate::jobs::cli::find_here(Some(Path::new("/definitely/not/here/ferrum-gpu")))
        .unwrap_err();
    assert!(e.contains("no such file"), "{e}");
}
