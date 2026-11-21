use super::*;

#[test]
fn degenerate_inputs_do_not_panic() {
    let mut r = Rng::new(5);
    let hot = Params {
        temperature: 1.0,
        ..Params::default()
    };
    assert_eq!(sample(&[f32::NAN, f32::NAN], &hot, &[], &mut r), 0);
    assert_eq!(sample(&[0.0], &hot, &[], &mut r), 0);
    let big = Params {
        temperature: 1.0,
        top_k: 100,
        top_p: 0.0001,
        ..Params::default()
    };
    assert_eq!(sample(&L, &big, &[], &mut r), 1);
}
