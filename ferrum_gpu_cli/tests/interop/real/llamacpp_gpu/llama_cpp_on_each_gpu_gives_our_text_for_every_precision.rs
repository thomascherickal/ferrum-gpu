use super::*;

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
