//! The JSON the page reads about a job.

use super::{Job, Line};
use crate::json::Json;

fn num(n: impl Into<f64>) -> Json {
    Json::Num(n.into())
}

fn line_json(index: usize, l: &Line) -> Json {
    let mut o = vec![
        ("i".to_string(), num(index as f64)),
        ("e".to_string(), Json::Bool(l.err)),
        ("t".to_string(), Json::Str(l.text.clone())),
    ];
    if let Some(ev) = &l.ev {
        o.push(("ev".to_string(), ev.json()));
    }
    Json::Obj(o)
}

/// One line per job for the list.
pub fn summary(job: &Job) -> Json {
    let p = job.poll(usize::MAX);
    Json::obj([
        ("id", num(job.id as f64)),
        ("cmd", Json::Str(job.cmd.clone())),
        ("state", Json::Str(p.state.into())),
        ("seconds", num(p.seconds)),
    ])
}

/// A poll: state, exit status, the new lines (each with its progress event, if any) and `next`.
pub fn poll(job: &Job, since: usize) -> Json {
    let p = job.poll(since);
    let opt = |e: &Option<super::progress::Event>| e.as_ref().map_or(Json::Null, |e| e.json());
    Json::obj([
        ("id", num(job.id as f64)),
        ("cmd", Json::Str(job.cmd.clone())),
        (
            "args",
            Json::Arr(job.args.iter().cloned().map(Json::Str).collect()),
        ),
        ("state", Json::Str(p.state.into())),
        ("exit", p.exit.map_or(Json::Null, num)),
        ("seconds", num(p.seconds)),
        (
            "lines",
            Json::Arr(p.lines.iter().map(|(i, l)| line_json(*i, l)).collect()),
        ),
        ("next", num(p.next as f64)),
        ("step", opt(&p.last_step)),
        ("val", opt(&p.last_val)),
        ("eval", opt(&p.last_eval)),
    ])
}
