//! Static files: the page, its script and style, the browser runtime modules (Phase 5b) and the
//! WGSL kernels. Everything is compiled in; lookups are by exact name, so no request path ever
//! reaches the file system.

use crate::http::Response;

/// The policy every page response carries: scripts and styles only from this origin (the page has no
/// inline script or style), connections only back to the server, no plugins, no framing. The desktop
/// app loads this same page in its window, so the policy is what keeps it a closed box.
pub const CSP: &str =
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; \
connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; \
frame-ancestors 'none'";

const PAGE: &str = include_str!("../web/index.html");
const STYLE: &str = include_str!("../web/style.css");
const APP: &str = include_str!("../web/app.js");

/// The embedded browser modules, by file name under `/web/`.
pub const MODULES: &[(&str, &str)] = &[
    ("gguf.js", include_str!("../web/gguf.js")),
    ("webgpu.js", include_str!("../web/webgpu.js")),
    ("llama.js", include_str!("../web/llama.js")),
    ("browser.js", include_str!("../web/browser.js")),
    ("selftest.js", include_str!("../web/selftest.js")),
    ("bench.js", include_str!("../web/bench.js")),
    ("stop.js", include_str!("../web/stop.js")),
    ("studio.js", include_str!("../web/studio.js")),
    (
        "autotest/studio.js",
        include_str!("../web/autotest/studio.js"),
    ),
    ("shell/api.js", include_str!("../web/shell/api.js")),
    ("shell/state.js", include_str!("../web/shell/state.js")),
    ("jobs/client.js", include_str!("../web/jobs/client.js")),
    ("jobs/panel.js", include_str!("../web/jobs/panel.js")),
    ("files/paths.js", include_str!("../web/files/paths.js")),
    ("files/picker.js", include_str!("../web/files/picker.js")),
    ("tabs/args.js", include_str!("../web/tabs/args.js")),
    ("tabs/datasets.js", include_str!("../web/tabs/datasets.js")),
    ("tabs/prepare.js", include_str!("../web/tabs/prepare.js")),
    (
        "tabs/train_args.js",
        include_str!("../web/tabs/train_args.js"),
    ),
    ("tabs/gtrain.js", include_str!("../web/tabs/gtrain.js")),
    ("charts/loss.js", include_str!("../web/charts/loss.js")),
    ("shell/index.js", include_str!("../web/shell/index.js")),
    ("shell/tabs.js", include_str!("../web/shell/tabs.js")),
    (
        "shell/settings.js",
        include_str!("../web/shell/settings.js"),
    ),
    ("shell/tooltip.js", include_str!("../web/shell/tooltip.js")),
    ("shell/help.js", include_str!("../web/shell/help.js")),
];

/// The response for a static `path`, or `None` if it names nothing embedded.
pub fn serve(path: &str) -> Option<Response> {
    let (ct, body): (&str, &str) = match path {
        "/" | "/index.html" => ("text/html; charset=utf-8", PAGE),
        "/style.css" => ("text/css; charset=utf-8", STYLE),
        "/app.js" => ("text/javascript; charset=utf-8", APP),
        "/shaders/prelude.wgsl" => ("text/plain; charset=utf-8", ferrum_wgpu::shaders::PRELUDE),
        p => {
            if let Some(name) = p.strip_prefix("/web/") {
                let (_, src) = MODULES.iter().find(|(n, _)| *n == name)?;
                ("text/javascript; charset=utf-8", src)
            } else {
                let name = p.strip_prefix("/shaders/")?.strip_suffix(".wgsl")?;
                (
                    "text/plain; charset=utf-8",
                    ferrum_wgpu::shaders::source(name)?,
                )
            }
        }
    };
    let mut r = Response::new(200, ct, body);
    r.headers
        .push(("Content-Security-Policy".into(), CSP.into()));
    Some(r)
}
