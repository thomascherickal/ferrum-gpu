//! Static files: the page, its script and style, the browser runtime modules (Phase 5b) and the
//! WGSL kernels. Everything is compiled in; lookups are by exact name, so no request path ever
//! reaches the file system.

use crate::http::Response;

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
    Some(Response::new(200, ct, body))
}
