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

fn tree(name: &str, cards: &[(&str, Option<&str>)]) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!("ferrum_gpu_sysfs_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (card, vram) in cards {
        let dev = root.join(card).join("device");
        std::fs::create_dir_all(&dev).unwrap();
        if let Some(v) = vram {
            std::fs::write(dev.join("mem_info_vram_total"), v).unwrap();
        }
    }
    root
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

#[test]
fn plan_refuses_before_allocating() {
    let mut p = BufferPlan::default();
    p.add("w", 600);
    p.add("act", 600);
    assert_eq!(p.total(), 1200);
    let e = p
        .check(
            1000,
            Some(Budget {
                bytes: 1000,
                source: Source::Override,
            }),
        )
        .unwrap_err();
    assert_eq!(
        e,
        PlanError::OverBudget {
            need: 1200,
            have: 1000
        }
    );
    assert!(e.to_string().contains("1200") && e.to_string().contains("1000"));
    assert!(p.check(1000, None).is_ok());
    assert!(p
        .check(
            1000,
            Some(Budget {
                bytes: 1200,
                source: Source::Tool
            })
        )
        .is_ok());
}

#[test]
fn plan_names_a_tensor_over_the_binding_limit() {
    let mut p = BufferPlan::default();
    p.add("tok_emb", 5000);
    let e = p.check(4096, None).unwrap_err();
    assert!(e.to_string().contains("tok_emb") && e.to_string().contains("4096"));
}
