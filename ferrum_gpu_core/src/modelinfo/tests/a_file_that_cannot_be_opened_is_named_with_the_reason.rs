use super::*;

// /proc/1/environ is a regular file that only root can open (see the fsapi preview test).
#[test]
fn a_file_that_cannot_be_opened_is_named_with_the_reason() {
    let f = std::path::Path::new("/proc/1/environ");
    if std::fs::File::open(f).is_ok() {
        eprintln!("SKIP: running as a user who can read /proc/1/environ");
        return;
    }
    let e = inspect_with(f, 100).unwrap_err();
    assert!(
        e.starts_with("/proc/1/environ: ") && e.contains("denied"),
        "{e}"
    );
}
