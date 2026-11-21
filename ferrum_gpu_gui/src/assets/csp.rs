/// The policy every page response carries: scripts and styles only from this origin (the page has no
/// inline script or style), connections only back to the server, no plugins, no framing. The desktop
/// app loads this same page in its window, so the policy is what keeps it a closed box.
pub const CSP: &str =
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; \
connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; \
frame-ancestors 'none'";

pub(super) const PAGE: &str = include_str!("../../web/index.html");
pub(super) const STYLE: &str = include_str!("../../web/style.css");
pub(super) const APP: &str = include_str!("../../web/app.js");

/// The embedded browser modules, by file name under `/web/`.
pub const MODULES: &[(&str, &str)] = &[
    ("gguf.js", include_str!("../../web/gguf.js")),
    ("webgpu.js", include_str!("../../web/webgpu.js")),
    ("llama.js", include_str!("../../web/llama.js")),
    ("browser.js", include_str!("../../web/browser.js")),
    ("selftest.js", include_str!("../../web/selftest.js")),
    ("bench.js", include_str!("../../web/bench.js")),
    ("stop.js", include_str!("../../web/stop.js")),
    ("studio.js", include_str!("../../web/studio.js")),
    (
        "autotest/studio.js",
        include_str!("../../web/autotest/studio.js"),
    ),
    ("shell/api.js", include_str!("../../web/shell/api.js")),
    ("shell/state.js", include_str!("../../web/shell/state.js")),
    ("jobs/client.js", include_str!("../../web/jobs/client.js")),
    ("jobs/panel.js", include_str!("../../web/jobs/panel.js")),
    ("files/paths.js", include_str!("../../web/files/paths.js")),
    ("files/picker.js", include_str!("../../web/files/picker.js")),
    ("tabs/args.js", include_str!("../../web/tabs/args.js")),
    (
        "tabs/datasets.js",
        include_str!("../../web/tabs/datasets.js"),
    ),
    ("tabs/prepare.js", include_str!("../../web/tabs/prepare.js")),
    (
        "tabs/train_args.js",
        include_str!("../../web/tabs/train_args.js"),
    ),
    ("tabs/gtrain.js", include_str!("../../web/tabs/gtrain.js")),
    (
        "tabs/flow_args.js",
        include_str!("../../web/tabs/flow_args.js"),
    ),
    (
        "tabs/finetune.js",
        include_str!("../../web/tabs/finetune.js"),
    ),
    (
        "tabs/evaluate.js",
        include_str!("../../web/tabs/evaluate.js"),
    ),
    (
        "tabs/generate.js",
        include_str!("../../web/tabs/generate.js"),
    ),
    ("tabs/models.js", include_str!("../../web/tabs/models.js")),
    (
        "tabs/runtime_args.js",
        include_str!("../../web/tabs/runtime_args.js"),
    ),
    ("tabs/export.js", include_str!("../../web/tabs/export.js")),
    ("tabs/serve.js", include_str!("../../web/tabs/serve.js")),
    ("tabs/capable.js", include_str!("../../web/tabs/capable.js")),
    ("tabs/gpulab.js", include_str!("../../web/tabs/gpulab.js")),
    (
        "tabs/lab_args.js",
        include_str!("../../web/tabs/lab_args.js"),
    ),
    (
        "tabs/capable_view.js",
        include_str!("../../web/tabs/capable_view.js"),
    ),
    (
        "tabs/models_view.js",
        include_str!("../../web/tabs/models_view.js"),
    ),
    (
        "charts/livechart.js",
        include_str!("../../web/charts/livechart.js"),
    ),
    ("charts/loss.js", include_str!("../../web/charts/loss.js")),
    ("shell/index.js", include_str!("../../web/shell/index.js")),
    ("shell/tabs.js", include_str!("../../web/shell/tabs.js")),
    (
        "shell/settings.js",
        include_str!("../../web/shell/settings.js"),
    ),
    (
        "shell/tooltip.js",
        include_str!("../../web/shell/tooltip.js"),
    ),
    ("shell/help.js", include_str!("../../web/shell/help.js")),
];
