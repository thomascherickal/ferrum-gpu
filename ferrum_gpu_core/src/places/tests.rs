use super::*;

#[test]
fn home_is_home_then_userprofile_and_a_missing_cwd_is_dot() {
    let env = |pairs: &'static [(&str, &str)]| {
        move |k: &str| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| OsString::from(v))
        }
    };
    let both = env(&[("HOME", "/h"), ("USERPROFILE", "C:/u")]);
    assert_eq!(home_from(both), Some(PathBuf::from("/h")));
    let windows = env(&[("USERPROFILE", "C:/u")]);
    assert_eq!(home_from(windows), Some(PathBuf::from("C:/u")));
    assert_eq!(home_from(env(&[])), None);
    assert_eq!(
        or_dot(Err(std::io::Error::other("gone"))),
        PathBuf::from(".")
    );
    assert!(cwd_or_dot().is_absolute() && home_dir().is_some());
}
