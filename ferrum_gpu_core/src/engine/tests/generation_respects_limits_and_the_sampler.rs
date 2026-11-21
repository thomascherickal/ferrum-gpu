use super::*;

#[test]
fn generation_respects_limits_and_the_sampler() {
    let opts = GenOpts {
        max_new: 5,
        temperature: 0.0,
        top_k: 0,
        seed: 1,
    };
    let mut fed = Vec::new();
    // logits that always prefer (last token + 1) % 8
    let mut feed = |t: usize| -> Result<Vec<f32>, String> {
        fed.push(t);
        let mut l = vec![0.0; 8];
        l[(t + 1) % 8] = 5.0;
        Ok(l)
    };
    let out = generate_with(&mut feed, 100, &[2, 3], &opts).unwrap();
    assert_eq!(out, vec![4, 5, 6, 7, 0], "greedy follows the pattern");
    assert_eq!(
        fed,
        vec![2, 3, 4, 5, 6, 7],
        "the last token is not fed again"
    );
    // the context window ends generation early
    let mut feed2 = |_: usize| -> Result<Vec<f32>, String> { Ok(vec![1.0, 0.0]) };
    let out = generate_with(
        &mut feed2,
        5,
        &[1, 1],
        &GenOpts {
            max_new: 50,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(out.len(), 3, "5 - 2 prompt tokens");
    assert!(generate_with(&mut feed2, 5, &[], &opts)
        .unwrap_err()
        .contains("empty"));
    assert!(generate_with(&mut feed2, 2, &[1, 1], &opts)
        .unwrap_err()
        .contains("context"));
    let mut failing = |_: usize| -> Result<Vec<f32>, String> { Err("boom".into()) };
    assert_eq!(
        generate_with(&mut failing, 9, &[1], &opts).unwrap_err(),
        "boom"
    );
    // temperature sampling is seeded: same seed, same tokens
    let hot = GenOpts {
        temperature: 1.5,
        top_k: 3,
        max_new: 20,
        ..opts
    };
    let mut flat =
        |_: usize| -> Result<Vec<f32>, String> { Ok((0..8).map(|i| i as f32 * 0.1).collect()) };
    let (a, b) = (
        generate_with(&mut flat, 99, &[1], &hot).unwrap(),
        generate_with(&mut flat, 99, &[1], &hot).unwrap(),
    );
    assert_eq!(a, b);
    assert!(a.iter().all(|&t| t < 8));
}
