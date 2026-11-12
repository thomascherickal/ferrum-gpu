use super::support::{app_with_model, call, field, train};
use ferrum_gpu_gui::json::Json;

fn num(j: &Json, k: &str) -> Option<f64> {
    field(j, k).as_f64()
}

#[test]
fn no_share_means_no_validation_numbers() {
    let a = app_with_model();
    let (_, done) = train(&a, None);
    assert_eq!(field(&done, "state").as_str(), Some("done"));
    assert_eq!(field(&done, "val_before"), Json::Null);
    assert_eq!(field(&done, "val_after"), Json::Null);
}

#[test]
fn a_share_reports_the_held_out_loss_before_and_after_and_it_falls() {
    let a = app_with_model();
    let (first, done) = train(&a, Some(0.2));
    assert_eq!(first.0, 200, "{}", first.1);
    assert_eq!(field(&done, "state").as_str(), Some("done"), "{done}");
    let (before, after) = (
        num(&done, "val_before").unwrap(),
        num(&done, "val_after").unwrap(),
    );
    assert!(before.is_finite() && after.is_finite());
    assert!(
        before > 3.0,
        "an untrained model is far from the text: {before}"
    );
    assert!(after < 0.6 * before, "held-out loss {before} -> {after}");
}

#[test]
fn the_run_still_takes_every_step_on_the_remaining_text() {
    let a = app_with_model();
    let (_, done) = train(&a, Some(0.3));
    assert_eq!(num(&done, "step"), Some(60.0));
    match done.get("losses") {
        Some(Json::Arr(l)) => assert_eq!(l.len(), 60),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_validation_text_is_never_trained_on() {
    // train on text whose tail is a different alphabet; if the tail were trained on, its loss
    // would fall as far as the head's does
    let a = app_with_model();
    let text = format!("{}{}", "ab".repeat(300), "xyzw".repeat(60));
    let body = Json::obj([
        ("text", Json::Str(text)),
        ("steps", Json::Num(80.0)),
        ("lr", Json::Num(0.02)),
        ("batch", Json::Num(4.0)),
        ("context", Json::Num(16.0)),
        ("val", Json::Num(0.3)),
    ]);
    let (s, _) = call(&a, "POST", "/api/train", &body.to_string());
    assert_eq!(s, 200);
    let done = super::support::wait(&a);
    let (before, after) = (
        num(&done, "val_before").unwrap(),
        num(&done, "val_after").unwrap(),
    );
    assert!(
        after > 0.5 * before,
        "unseen text must not be learned: {before} -> {after}"
    );
}

#[test]
fn shares_outside_the_range_are_400s() {
    let a = app_with_model();
    for bad in [0.6, 1.0, -0.1] {
        let (first, _) = train(&a, Some(bad));
        assert_eq!(first.0, 400, "val {bad}: {}", first.1);
    }
}

#[test]
fn a_text_too_short_to_split_and_still_train_is_a_400() {
    let a = app_with_model();
    let body = r#"{"text":"short text here","steps":5,"context":24,"val":0.5}"#;
    let (s, j) = call(&a, "POST", "/api/train", body);
    assert_eq!(s, 400, "{j}");
}
