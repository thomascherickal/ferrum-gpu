use super::*;

#[test]
fn nvidia_smi_one_many_and_junk() {
    assert_eq!(
        parse_nvidia_smi("NVIDIA GeForce MX330, 2048, 100\n"),
        vec![("NVIDIA GeForce MX330".into(), Some(2048))]
    );
    let two =
        parse_nvidia_smi("A, 8192, 1\nTesla, Fancy Edition, [N/A], 3\n\nNVIDIA-SMI has failed\n");
    assert_eq!(
        two,
        vec![
            ("A".into(), Some(8192)),
            ("Tesla, Fancy Edition".into(), None)
        ]
    );
    assert!(parse_nvidia_smi("").is_empty());
}

#[test]
fn vram_bytes_to_mib() {
    assert_eq!(parse_vram_mb("8589934592\n"), Some(8192));
    assert_eq!(parse_vram_mb("n/a"), None);
}

#[test]
fn sysfs_reads_cards_only_in_order() {
    let r = tree(
        "a",
        &[
            ("card1", Some("2147483648\n")),
            ("card0", Some("4294967296\n")),
            ("card0-DP-1", Some("1\n")),
            ("card2", None),
        ],
    );
    assert_eq!(sysfs_vram_mb(&r), vec![4096, 2048]);
    let _ = std::fs::remove_dir_all(&r);
    assert!(sysfs_vram_mb(&r).is_empty()); // missing dir
}

#[test]
fn resolve_precedence() {
    let ram = Some(16 * 1024 * MIB);
    assert_eq!(
        resolve(Some(100), Some(2048), false, ram).unwrap(),
        Budget {
            bytes: 100 * MIB,
            source: Source::Override
        }
    );
    assert_eq!(
        resolve(None, Some(2048), false, ram).unwrap(),
        Budget {
            bytes: 2048 * MIB,
            source: Source::Tool
        }
    );
    assert_eq!(
        resolve(None, Some(2048), true, ram).unwrap(),
        Budget {
            bytes: 8 * 1024 * MIB,
            source: Source::HalfRam
        }
    );
    assert_eq!(resolve(None, None, true, None), None);
    assert_eq!(resolve(None, None, false, ram), None);
}
