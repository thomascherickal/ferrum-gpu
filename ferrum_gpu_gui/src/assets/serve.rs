use super::*;

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
