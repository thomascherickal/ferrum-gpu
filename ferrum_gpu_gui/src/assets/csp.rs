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
