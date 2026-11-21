use super::*;

pub(crate) fn find_tool_in(
    flag: Option<&str>,
    env_val: Option<&str>,
    env: &str,
    path_var: Option<&std::ffi::OsStr>,
    name: &str,
    fallback: &str,
) -> Result<PathBuf, String> {
    if let Some(f) = flag {
        let p = PathBuf::from(f);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("--bin {f}: no such file"))
        };
    }
    if let Some(v) = env_val.filter(|v| !v.is_empty()) {
        let p = PathBuf::from(v);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("{env}={v}: no such file"))
        };
    }
    let names = [name.to_string(), format!("{name}.exe")];
    let on_path = path_var.and_then(|pv| {
        std::env::split_paths(pv)
            .flat_map(|d| names.iter().map(move |n| d.join(n)))
            .find(|c| c.is_file())
    });
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| Path::new(&h).join(fallback));
    on_path.or(home.filter(|h| h.is_file())).ok_or_else(|| {
        format!("cannot find `{name}`: pass --bin PATH, set {env}, or put it on PATH")
    })
}

pub(crate) fn run_tool(tool: &Path, args: &[&str]) -> Result<(bool, String), String> {
    let o = Command::new(tool)
        .args(args)
        .output()
        .map_err(|e| format!("{}: {e}", tool.display()))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    Ok((o.status.success(), text))
}
