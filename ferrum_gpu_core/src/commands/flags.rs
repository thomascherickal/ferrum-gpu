//! A flag-table parser: each command lists the flags it accepts; anything else is an error.

use std::collections::HashMap;
use std::str::FromStr;

/// One accepted flag.
#[derive(Clone, Copy, Debug)]
pub struct Flag {
    /// Spelling, with the leading `--`.
    pub name: &'static str,
    /// `true` if a value follows (`--steps 10` or `--steps=10`); `false` for a switch.
    pub value: bool,
    /// `true` if the flag may be given more than once (read with [`Args::all`]).
    pub many: bool,
}

/// A flag that takes a value.
pub const fn val(name: &'static str) -> Flag {
    Flag {
        name,
        value: true,
        many: false,
    }
}

/// A flag that takes a value and may be repeated.
pub const fn many(name: &'static str) -> Flag {
    Flag {
        name,
        value: true,
        many: true,
    }
}

/// A switch.
pub const fn switch(name: &'static str) -> Flag {
    Flag {
        name,
        value: false,
        many: false,
    }
}

/// Parsed flags.
#[derive(Debug, Default)]
pub struct Args {
    map: HashMap<&'static str, String>,
    rep: HashMap<&'static str, Vec<String>>,
}

impl Args {
    /// Parse `args` against `table`. Unknown flags, repeated flags, missing values, values given
    /// to switches and stray positional words are errors.
    pub fn parse(table: &[Flag], args: &[String]) -> Result<Args, String> {
        let mut out = Args::default();
        let mut it = args.iter();
        while let Some(raw) = it.next() {
            let (name, inline) = match raw.split_once('=') {
                Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
                _ => (raw.as_str(), None),
            };
            let flag = table
                .iter()
                .find(|f| f.name == name)
                .ok_or_else(|| format!("unknown flag '{raw}'"))?;
            let value = match (flag.value, inline) {
                (true, Some(v)) => v,
                (true, None) => it
                    .next()
                    .cloned()
                    .ok_or_else(|| format!("{name} needs a value"))?,
                (false, None) => String::new(),
                (false, Some(_)) => return Err(format!("{name} takes no value")),
            };
            if flag.many {
                out.rep.entry(flag.name).or_default().push(value.clone());
                out.map.insert(flag.name, value);
            } else if out.map.insert(flag.name, value).is_some() {
                return Err(format!("{name} given twice"));
            }
        }
        Ok(out)
    }

    /// Was the flag given?
    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    /// The raw value, if given.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    /// Every value of a repeatable flag, in order (empty if absent).
    pub fn all(&self, name: &str) -> &[String] {
        self.rep.get(name).map_or(&[], Vec::as_slice)
    }

    /// The first of `names` that was given (for flags with several spellings).
    pub fn first(&self, names: &[&'static str]) -> Option<&'static str> {
        names.iter().copied().find(|n| self.has(n))
    }

    /// The raw value, or an error naming the flag.
    pub fn need(&self, name: &str) -> Result<&str, String> {
        self.get(name).ok_or_else(|| format!("{name} is required"))
    }

    /// The value parsed as `T`, or `default` when absent.
    pub fn num<T: FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("{name}: '{v}' is not a valid value")),
        }
    }
}

#[cfg(test)]
mod tests {
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
}
