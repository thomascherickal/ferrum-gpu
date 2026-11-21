use super::*;

const T: &[Flag] = &[val("--steps"), val("--out"), switch("--quiet")];
fn a(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

#[test]
fn parses_values_switches_and_equals_form() {
    let p = Args::parse(T, &a("--steps 5 --out=x.gguf --quiet")).unwrap();
    assert_eq!(p.num("--steps", 1usize).unwrap(), 5);
    assert_eq!(p.get("--out"), Some("x.gguf"));
    assert!(p.has("--quiet") && !p.has("--nope"));
    assert_eq!(p.num("--absent", 7u32).unwrap(), 7);
}

#[test]
fn rejects_every_malformed_form() {
    for (bad, why) in [
        ("--nope", "unknown flag"),
        ("--steps", "needs a value"),
        ("--steps 1 --steps 2", "twice"),
        ("--quiet=1", "takes no value"),
        ("stray", "unknown flag"),
    ] {
        let e = Args::parse(T, &a(bad)).unwrap_err();
        assert!(e.contains(why), "{bad}: {e}");
    }
}

#[test]
fn repeatable_flags_collect_every_value() {
    let t = [many("--stop"), val("--a"), val("--b")];
    let p = Args::parse(&t, &a("--stop x --stop=y")).unwrap();
    assert_eq!(p.all("--stop"), ["x", "y"]);
    assert!(p.all("--a").is_empty());
    assert_eq!(p.first(&["--a", "--stop"]), Some("--stop"));
    assert_eq!(p.first(&["--a", "--b"]), None);
}

#[test]
fn typed_getters_report_the_flag() {
    let p = Args::parse(T, &a("--steps abc")).unwrap();
    assert!(p.num("--steps", 1usize).unwrap_err().contains("--steps"));
    assert!(p.need("--out").unwrap_err().contains("--out is required"));
    assert_eq!(p.need("--steps").unwrap(), "abc");
}
