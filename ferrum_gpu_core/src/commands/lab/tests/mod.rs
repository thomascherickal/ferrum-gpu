use super::*;
use crate::commands::run as run_command;

fn argv(words: &[&str]) -> Vec<String> {
    words.iter().map(|s| s.to_string()).collect()
}

struct NoReaders;
impl sysmem::Readers for NoReaders {
    fn nvidia_smi(&self) -> Option<String> {
        None
    }
    fn ram_bytes(&self) -> (Option<u64>, Option<u64>) {
        (None, None)
    }
    fn sysfs_vram_mb(&self) -> Vec<u64> {
        vec![]
    }
}

fn checked(error: f32) -> Report {
    use report::Check;
    Report {
        checks: vec![Check {
            name: "f32".into(),
            error,
            tolerance: 1e-5,
        }],
        ..Report::default()
    }
}

mod bad_flags_are_refused_before_any_device_is_opened;
mod the_command_prints_text_or_json_for_an_available_adapter;
