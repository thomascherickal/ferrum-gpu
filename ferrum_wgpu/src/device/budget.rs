// Provenance: the nvidia-smi / sysfs parsing idea is from ferrum-cpu
// `ferrum_gui/src/sizing/hardware.rs` (commit 56a9d3a); rewritten here, ours from now on.
//! GPU memory budget (wgpu has no portable VRAM query) and the plan-then-allocate check.

use std::path::Path;

/// Where a budget number came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `--gpu-mem-mb` / `FERRUM_GPU_MEM_MB`.
    Override,
    /// `nvidia-smi` or sysfs.
    Tool,
    /// Half of system RAM, for integrated / unified-memory adapters.
    HalfRam,
}

/// A memory budget in bytes with its provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Bytes available for buffers.
    pub bytes: u64,
    /// Where it came from.
    pub source: Source,
}

const MIB: u64 = 1024 * 1024;

/// `name, total_mib, used_mib` lines (nvidia-smi csv, no header, no units) to `(name, total_mib)`.
/// The name is split from the right so commas inside it survive; junk lines are skipped and
/// `[N/A]` totals give `None`.
pub fn parse_nvidia_smi(out: &str) -> Vec<(String, Option<u64>)> {
    out.lines()
        .filter_map(|line| {
            let mut it = line.trim().rsplitn(3, ',');
            let _used = it.next()?;
            let total = it.next()?.trim();
            let name = it.next()?.trim();
            let mb = match total {
                "[N/A]" | "N/A" => None,
                t => Some(t.parse::<u64>().ok()?),
            };
            (!name.is_empty()).then(|| (name.to_string(), mb))
        })
        .collect()
}

/// sysfs `mem_info_vram_total` (bytes) to MiB.
pub fn parse_vram_mb(s: &str) -> Option<u64> {
    s.trim().parse::<u64>().ok().map(|b| b / MIB)
}

/// Total VRAM in MiB of every `<drm>/card<N>/device/mem_info_vram_total` found. The root is
/// injectable so tests use a temp tree; connector entries (`card0-DP-1`) are ignored.
pub fn sysfs_vram_mb(drm: &Path) -> Vec<u64> {
    let mut cards: Vec<_> = std::fs::read_dir(drm)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.strip_prefix("card")
                .is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(|e| e.path())
        .collect();
    cards.sort();
    cards
        .iter()
        .filter_map(|c| std::fs::read_to_string(c.join("device/mem_info_vram_total")).ok())
        .filter_map(|s| parse_vram_mb(&s))
        .collect()
}

/// Resolve the budget: override > tool-reported VRAM > half of RAM when `integrated` > unknown.
pub fn resolve(
    override_mb: Option<u64>,
    tool_mb: Option<u64>,
    integrated: bool,
    ram_bytes: Option<u64>,
) -> Option<Budget> {
    if let Some(mb) = override_mb {
        return Some(Budget {
            bytes: mb * MIB,
            source: Source::Override,
        });
    }
    if let Some(mb) = tool_mb.filter(|_| !integrated) {
        return Some(Budget {
            bytes: mb * MIB,
            source: Source::Tool,
        });
    }
    if integrated {
        return ram_bytes.map(|r| Budget {
            bytes: r / 2,
            source: Source::HalfRam,
        });
    }
    None
}

/// Refusal from [`BufferPlan::check`].
#[derive(Debug, PartialEq, Eq)]
pub enum PlanError {
    /// One buffer is larger than the adapter's binding limit.
    TooLarge {
        /// Buffer name.
        name: String,
        /// Its size.
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// The sum exceeds the budget.
    OverBudget {
        /// Required bytes.
        need: u64,
        /// Available bytes.
        have: u64,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::TooLarge { name, bytes, limit } => {
                write!(
                    f,
                    "buffer '{name}' needs {bytes} bytes, adapter binding limit is {limit}"
                )
            }
            PlanError::OverBudget { need, have } => {
                write!(f, "needs {need} bytes of GPU memory, budget is {have}")
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Every buffer an engine will create, listed before any is created.
#[derive(Default, Debug)]
pub struct BufferPlan {
    rows: Vec<(String, u64)>,
}

impl BufferPlan {
    /// Add a buffer to the plan.
    pub fn add(&mut self, name: &str, bytes: u64) {
        self.rows.push((name.to_string(), bytes));
    }

    /// The planned buffers as `(name, bytes)` rows, in insertion order.
    pub fn rows(&self) -> &[(String, u64)] {
        &self.rows
    }

    /// Sum of all planned bytes.
    pub fn total(&self) -> u64 {
        self.rows.iter().map(|r| r.1).sum()
    }

    /// Refuse (before anything is allocated) if any buffer exceeds `max_binding` or the sum
    /// exceeds `budget` (`None` budget = unknown, only the binding limit is enforced).
    pub fn check(&self, max_binding: u64, budget: Option<Budget>) -> Result<(), PlanError> {
        if let Some((name, bytes)) = self.rows.iter().find(|r| r.1 > max_binding) {
            return Err(PlanError::TooLarge {
                name: name.clone(),
                bytes: *bytes,
                limit: max_binding,
            });
        }
        match budget {
            Some(b) if self.total() > b.bytes => Err(PlanError::OverBudget {
                need: self.total(),
                have: b.bytes,
            }),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_one_many_and_junk() {
        assert_eq!(
            parse_nvidia_smi("NVIDIA GeForce MX330, 2048, 100\n"),
            vec![("NVIDIA GeForce MX330".into(), Some(2048))]
        );
        let two = parse_nvidia_smi(
            "A, 8192, 1\nTesla, Fancy Edition, [N/A], 3\n\nNVIDIA-SMI has failed\n",
        );
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
        let root =
            std::env::temp_dir().join(format!("ferrum_gpu_sysfs_{name}_{}", std::process::id()));
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
}
