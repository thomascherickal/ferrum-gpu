//! llama.cpp's Vulkan build on each GPU: the same greedy text as ours for every precision, with the
//! verbose log as proof that the layers really were on the GPU (and a CPU control that they were not).

use crate::skip;
use crate::support::log::{grep, opted_in, step};
use crate::support::model::{fixture, ours, NEW_TOKENS, PROMPT};
use crate::support::proc::with_deadline;
use crate::support::tools::llamacpp_vulkan_dir;
use std::path::Path;
use std::process::Stdio;

/// What one verbose `llama-completion` run produced.
struct Run {
    text: String,
    log: String,
}

fn completion(dir: &Path, model: &Path, offload: &[&str]) -> Run {
    let o = with_deadline(&dir.join("llama-completion"), 540)
        .env("LD_LIBRARY_PATH", dir)
        .args(["-v", "-m"])
        .arg(model)
        .args(["-p", PROMPT, "-n", &NEW_TOKENS.to_string()])
        .args(["--temp", "0", "-no-cnv", "--no-warmup", "-t", "1"])
        .args(offload)
        .stdin(Stdio::null())
        .output()
        .expect("run llama-completion");
    assert!(
        o.status.success(),
        "llama-completion {offload:?}: {}",
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .last()
            .unwrap_or("")
    );
    Run {
        text: String::from_utf8_lossy(&o.stdout).trim_end().to_string(),
        log: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// `Vulkan0`, `Vulkan1`, ... as listed by `--list-devices`, with their descriptions.
fn devices(dir: &Path) -> Vec<(String, String)> {
    let o = with_deadline(&dir.join("llama-completion"), 60)
        .env("LD_LIBRARY_PATH", dir)
        .arg("--list-devices")
        .stdin(Stdio::null())
        .output()
        .expect("list devices");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    text.lines()
        .filter_map(|l| {
            let (name, rest) = l.trim().split_once(": ")?;
            name.starts_with("Vulkan")
                .then(|| (name.to_string(), rest.to_string()))
        })
        .collect()
}

/// `(offloaded, total)` from the `offloaded A/B layers to GPU` line.
fn offloaded(log: &str) -> Option<(u32, u32)> {
    let rest = log.split("offloaded ").nth(1)?;
    let frac = rest.split_whitespace().next()?;
    let (a, b) = frac.split_once('/')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

#[test]
fn llama_cpp_on_each_gpu_gives_our_text_for_every_precision() {
    if !opted_in("FERRUM_INTEROP_GPU", "runs llama.cpp on the GPUs") {
        return;
    }
    let Some(dir) = llamacpp_vulkan_dir() else {
        skip!("no Vulkan build of llama.cpp (set FERRUM_LLAMACPP_VULKAN_DIR to an unpacked llama-bNNNN-bin-ubuntu-vulkan-x64)")
    };
    step(format!("llama.cpp Vulkan build at {}", dir.display()));
    let devs = devices(&dir);
    assert!(!devs.is_empty(), "the Vulkan build lists no device");
    for (name, desc) in &devs {
        step(format!("device {name}: {desc}"));
    }
    for (dev, desc) in &devs {
        for (label, model) in fixture().all() {
            step(format!(
                "{dev} ({desc}), {label}: llama-completion -v -ngl 99 --device {dev} -t 1 on {}",
                model.display()
            ));
            let run = completion(&dir, model, &["-ngl", "99", "--device", dev]);
            for l in grep(
                &run.log,
                &[
                    "using device",
                    "offloaded",
                    "Vulkan0 model",
                    "Vulkan1 model",
                ],
            )
            .iter()
            .take(4)
            {
                step(format!("  llama.cpp: {l}"));
            }
            let (a, b) = offloaded(&run.log).expect("an `offloaded A/B layers to GPU` line");
            assert!(a > 0 && a == b, "{dev} {label}: {a}/{b} layers on the GPU");
            assert!(
                run.log.contains(&format!("assigned to device {dev}")),
                "{dev} {label}: layers not on {dev}"
            );
            let mine = ours(model);
            step(format!("  llama.cpp text: {:?}", run.text));
            step(format!("  ferrum-gpu text: {mine:?}"));
            assert_eq!(run.text, mine, "{dev} {label}");
            step(format!("  {dev} {label}: MATCH"));
        }
    }
}

#[test]
fn the_cpu_control_shows_no_layer_on_the_gpu() {
    // the assertion above must be able to fail: with -ngl 0 the same build offloads nothing
    if !opted_in("FERRUM_INTEROP_GPU", "runs llama.cpp on the GPUs") {
        return;
    }
    let Some(dir) = llamacpp_vulkan_dir() else {
        skip!("no Vulkan build of llama.cpp")
    };
    step("control: the Vulkan build with -ngl 0 (all layers on the CPU)");
    let run = completion(&dir, &fixture().f32, &["-ngl", "0"]);
    let (a, b) = offloaded(&run.log).expect("an offload line");
    step(format!(
        "  offloaded {a}/{b} layers to GPU; text {:?}",
        run.text
    ));
    assert_eq!(a, 0);
    assert!(b > 0);
    assert_eq!(run.text, ours(&fixture().f32));
}
