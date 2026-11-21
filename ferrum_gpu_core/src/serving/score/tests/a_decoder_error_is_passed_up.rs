use super::*;

#[test]
fn a_decoder_error_is_passed_up() {
    struct Failing;
    impl Decoder for Failing {
        fn info(&self) -> Info {
            Info {
                device: "f".into(),
                kind: "cpu".into(),
                arch: "f".into(),
                params: 0,
                vocab: 4,
                ctx: 4,
            }
        }
        fn encode(&self, _: &str) -> Result<Vec<usize>, String> {
            Err("no tokenizer".into())
        }
        fn bos(&self) -> usize {
            0
        }
        fn complete(&self, _: &str, _: &GenReq) -> Result<(String, usize), String> {
            Err("x".into())
        }
        fn score(&self, _: &[usize]) -> Result<Vec<(f32, bool)>, String> {
            Err("device lost".into())
        }
    }
    assert_eq!(pair(&Failing, "a", "b").unwrap_err(), "no tokenizer");
    assert_eq!(rolling(&Failing, "a").unwrap_err(), "no tokenizer");
    assert_eq!(
        token_logprobs(&Failing, &[1, 2, 3], 1).unwrap_err(),
        "device lost"
    );
}
