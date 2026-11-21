use super::*;

#[test]
fn nvidia_smi_free_lines_are_read_with_the_same_tolerance_as_the_total_ones() {
    let out = "NVIDIA GeForce MX330, 2048, 1989\nNVIDIA A100-SXM4-40GB, 40960, 40000\nbroken line\nName, With, Commas GPU, 8192, 100\n";
    let v = parse_nvidia_smi_free(out);
    assert_eq!(
        v[0],
        ("NVIDIA GeForce MX330".to_string(), Some(2048), Some(1989))
    );
    assert_eq!(v[1].2, Some(40000));
    assert_eq!(v.last().unwrap().0, "Name, With, Commas GPU");
    assert_eq!(v.len(), 3, "the broken line is skipped");
    let na = parse_nvidia_smi_free("Some GPU, [N/A], [N/A]\nOther, 4096, N/A\n");
    assert_eq!(na[0], ("Some GPU".to_string(), None, None));
    assert_eq!(na[1], ("Other".to_string(), Some(4096), None));
    assert!(parse_nvidia_smi_free("").is_empty());
    assert!(parse_nvidia_smi_free(", 1, 2").is_empty(), "no name");
    assert!(parse_nvidia_smi_free("x, y, z").is_empty(), "not numbers");
}
