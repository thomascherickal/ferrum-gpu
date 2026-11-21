use super::*;

#[test]
fn model_errors_are_422() {
    struct Broken;
    impl Decoder for Broken {
        fn info(&self) -> crate::serving::decoder::Info {
            StubDecoder.info()
        }
        fn encode(&self, t: &str) -> Result<Vec<usize>, String> {
            StubDecoder.encode(t)
        }
        fn bos(&self) -> usize {
            0
        }
        fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
            Err("prompt too long".into())
        }
        fn score(&self, _: &[usize]) -> Result<Vec<(f32, bool)>, String> {
            Err("device lost".into())
        }
    }
    let l = Limits::default();
    for (p, b) in [
        ("/v1/generate", r#"{"requests":[{"prompt":"x"}]}"#),
        (
            "/v1/loglikelihood",
            r#"{"requests":[{"context":"a","continuation":"b"}]}"#,
        ),
        ("/v1/loglikelihood_rolling", r#"{"texts":["ab"]}"#),
    ] {
        let r = route(&Broken, &l, &req("POST", p, b));
        assert_eq!(r.status, 422, "{p}");
    }
}

#[test]
fn loglikelihood_over_the_batch_or_text_limit_is_refused_before_any_scoring() {
    let body =
        r#"{"requests":[{"context":"ab","continuation":"c"},{"context":"a","continuation":"b"}]}"#;
    let few = Limits {
        max_batch: 1,
        ..Limits::default()
    };
    assert_eq!(call_with(&few, "POST", "/v1/loglikelihood", body).0, 400);
    let short = Limits {
        max_chars: 2,
        ..Limits::default()
    };
    assert_eq!(call_with(&short, "POST", "/v1/loglikelihood", body).0, 400);
}
