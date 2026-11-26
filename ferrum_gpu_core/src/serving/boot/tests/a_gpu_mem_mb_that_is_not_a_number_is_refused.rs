use super::*;

#[test]
fn a_gpu_mem_mb_that_is_not_a_number_is_refused() {
    let d = tmp("mem_nan");
    let model = gpt2_file(&d);
    let e = boot(
        &argv(&format!("--model {model} --device cpu --gpu-mem-mb lots")),
        &mut Vec::new(),
    )
    .err()
    .unwrap();
    assert!(e.contains("--gpu-mem-mb"), "{e}");
}
