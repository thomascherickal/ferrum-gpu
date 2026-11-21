use super::*;

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
    pub(super) map: HashMap<&'static str, String>,
    pub(super) rep: HashMap<&'static str, Vec<String>>,
}
