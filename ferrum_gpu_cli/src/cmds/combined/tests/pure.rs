//! The pieces of `run` that build commands, with no model and no files.

use super::w;
use crate::cmds::combined::split::Split;
use crate::cmds::combined::{check_generation_half, generate_command, train_command};

fn parts() -> Split {
    Split {
        train: w("--data c.txt --epochs 3"),
        generate: w("--prompt hi --max-new 5"),
        model: "m.gguf".into(),
    }
}

#[test]
fn the_generation_half_needs_a_prompt_or_ids() {
    assert!(check_generation_half(&w("--prompt hi")).is_ok());
    assert!(check_generation_half(&w("--ids 1")).is_ok());
    let e = check_generation_half(&w("--max-new 5")).unwrap_err();
    assert!(e.contains("needs --prompt"), "{e}");
}

#[test]
fn train_writes_to_the_model_path() {
    assert_eq!(
        train_command(&parts()),
        w("train --data c.txt --epochs 3 --out m.gguf")
    );
}

#[test]
fn generate_reads_the_model_path() {
    assert_eq!(
        generate_command(&parts()),
        w("generate --model m.gguf --prompt hi --max-new 5")
    );
}
