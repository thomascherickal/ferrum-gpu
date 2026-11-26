use super::*;

#[test]
fn a_streamed_run_without_a_listener_still_returns_its_text() {
    let mut feed = |_: usize| -> Result<Vec<f32>, String> {
        let mut l = vec![0.0f32; 256];
        l[b'a' as usize] = 9.0;
        Ok(l)
    };
    let o = gen::Opts {
        max_new: 2,
        params: ferrum_model::sampling::Params::default(),
        seed: 1,
        stops: vec![],
        stream: true,
        stop_ids: vec![],
    };
    let decode = |ids: &[usize]| ids.iter().map(|&i| i as u8 as char).collect::<String>();
    let (text, n) = gen::run(&mut feed, &decode, 64, &[1], &o, None).unwrap();
    assert_eq!((text.as_str(), n), ("aa", 2));
}
