use super::*;

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
