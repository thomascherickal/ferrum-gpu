use super::support::{call, field, trained};
use ferrum_gpu_gui::json::Json;
use std::sync::Arc;

use ferrum_gpu_gui::app::App;

fn generate(app: &Arc<App>, stop: Option<&str>) -> (u16, Json) {
    let stop = stop.map_or(String::new(), |s| format!(r#","stop":{s}"#));
    call(
        app,
        "POST",
        "/api/generate",
        &format!(r#"{{"prompt":"the quick brown ","max_new":12{stop}}}"#),
    )
}

fn text(r: &(u16, Json)) -> String {
    assert_eq!(r.0, 200, "{}", r.1);
    field(&r.1, "text").as_str().unwrap().to_string()
}

#[test]
fn without_stops_the_whole_continuation_comes_back() {
    let a = trained();
    let t = text(&generate(&a, None));
    assert!(t.starts_with("fox") && t.len() > 3, "{t:?}");
}

#[test]
fn text_is_cut_before_the_first_stop() {
    let a = trained();
    assert_eq!(text(&generate(&a, Some(r#"[" "]"#))), "fox");
    assert_eq!(text(&generate(&a, Some(r#"["x"]"#))), "fo");
}

#[test]
fn the_earliest_of_several_stops_wins() {
    let a = trained();
    assert_eq!(text(&generate(&a, Some(r#"["jum"," ","x"]"#))), "fo");
}

#[test]
fn a_stop_that_never_appears_changes_nothing() {
    let a = trained();
    let plain = text(&generate(&a, None));
    assert_eq!(text(&generate(&a, Some(r#"["QQQ"]"#))), plain);
    assert_eq!(
        text(&generate(&a, Some(r#"[""]"#))),
        plain,
        "an empty stop never matches"
    );
}

#[test]
fn stop_strings_do_not_change_the_token_list() {
    // the cut is on the text; the token ids returned are the model's full output
    let a = trained();
    let plain = generate(&a, None);
    let cut = generate(&a, Some(r#"[" "]"#));
    assert_eq!(field(&plain.1, "tokens"), field(&cut.1, "tokens"));
}

#[test]
fn bad_stop_shapes_are_400s_that_say_why() {
    let a = trained();
    for (bad, why) in [
        (r#""x""#, "array of strings"),
        (r#"[1]"#, "array of strings"),
        (r#"["a","b","c","d","e","f","g","h","i"]"#, "at most"),
    ] {
        let r = generate(&a, Some(bad));
        assert_eq!(r.0, 400, "{bad}");
        assert!(
            field(&r.1, "error").as_str().unwrap().contains(why),
            "{bad}: {}",
            r.1
        );
    }
    let long = format!(r#"["{}"]"#, "x".repeat(65));
    assert_eq!(generate(&a, Some(&long)).0, 400);
}
