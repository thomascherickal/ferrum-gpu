use super::*;

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

mod nvidia_smi_one_many_and_junk;
mod plan_refuses_before_allocating;
