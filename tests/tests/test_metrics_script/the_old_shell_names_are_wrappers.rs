use super::*;

#[test]
fn the_old_shell_names_are_one_line_wrappers_around_the_python_scripts() {
    let names = [
        ("gate", "gate"),
        ("coverage", "coverage"),
        ("test-time-limit", "time_limit"),
        ("test-all", "all_tests"),
    ];
    for (wrapper, script) in names {
        let text = read(&format!("scripts/{wrapper}.sh"));
        let call = format!("exec python3 \"$(dirname \"$0\")/{script}.py\" \"$@\"");
        assert!(text.contains(&call), "{wrapper}.sh must call {script}.py");
        assert!(text.lines().count() <= 3, "{wrapper}.sh is only a wrapper");
        assert!(repo().join(format!("scripts/{script}.py")).is_file());
    }
}
