use super::*;

#[test]
fn stub_is_deterministic() {
    let d = StubDecoder;
    assert_eq!(d.complete("x", &req(3)).unwrap(), ("!!!".to_string(), 3));
    assert_eq!(
        d.complete(
            "x",
            &GenReq {
                stops: vec!["!".into()],
                ..req(3)
            }
        )
        .unwrap()
        .0,
        ""
    );
    assert_eq!(
        d.score(&[1, 2, 3]).unwrap(),
        vec![(-1.0, true), (-1.0, false)]
    );
    assert_eq!(
        (d.info().ctx, d.bos(), d.encode("a").unwrap()),
        (16, 0, vec![97])
    );
}
