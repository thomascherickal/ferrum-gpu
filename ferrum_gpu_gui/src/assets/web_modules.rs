/// `web_modules!("a.js", ..)`: the `(name, source)` pairs of files under `web/`; `include_str!`
/// needs a literal path, so the macro builds it from each name.
macro_rules! web_modules {
    ($($name:literal),* $(,)?) => {
        &[$(($name, include_str!(concat!("../../web/", $name)))),*]
    };
}

/// The embedded browser modules, by file name under `/web/`.
#[rustfmt::skip]
pub const MODULES: &[(&str, &str)] = web_modules![
    "style/panels.css", "gguf.js", "webgpu.js", "llama.js", "browser.js", "selftest.js",
    "bench.js", "stop.js", "studio.js", "autotest/studio.js", "shell/api.js", "shell/state.js",
    "jobs/client.js", "jobs/panel.js", "files/paths.js", "files/picker.js", "tabs/args.js",
    "tabs/datasets.js", "tabs/prepare.js", "tabs/train_args.js", "tabs/gtrain.js",
    "tabs/flow_args.js", "tabs/finetune.js", "tabs/evaluate.js", "tabs/generate.js",
    "tabs/models.js", "tabs/runtime_args.js", "tabs/export.js", "tabs/serve.js",
    "tabs/capable.js", "tabs/gpulab.js", "tabs/lab_args.js", "tabs/capable_view.js",
    "tabs/models_view.js", "charts/livechart.js", "charts/loss.js", "shell/index.js",
    "shell/tabs.js", "shell/settings.js", "shell/tooltip.js", "shell/help.js",
    "app/autotest.js", "app/device.js", "app/dom.js", "app/generate.js", "app/report.js",
    "app/train.js", "app/ui_autotest.js", "autotest/data_stages.js", "autotest/kit.js",
    "autotest/model_stages.js", "autotest/service_stages.js", "autotest/shell_stages.js",
    "autotest/train_stages.js", "files/listing.js", "gguf/fail.js", "gguf/quant.js",
    "gguf/reader.js", "llama/config.js", "llama/forward.js", "llama/sampling.js",
    "llama/util.js", "llama/weights.js", "shell/help/data_and_training.js",
    "shell/help/lab_and_page.js", "shell/help/models_and_serving.js", "tabs/fields.js",
    "tabs/models_dom.js", "tabs/options.js", "tabs/train_sections.js",
    "webgpu/dispatch_size.js", "webgpu/kernel.js", "webgpu/readback.js",
];
