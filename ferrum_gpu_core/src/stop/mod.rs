//! Stop strings: reading them from a request and cutting generated text at the first one.
//!
//! Shared by the GUI and the server, so both cut text the same way.

use crate::json::Json;

/// Most stop strings a request may carry.
pub const MAX_STOPS: usize = 8;
/// Longest stop string, in bytes.
pub const MAX_STOP_LEN: usize = 64;

/// Cut `text` before the first of `stops` (empty stop strings never match).
pub fn cut_at_stop(mut text: String, stops: &[String]) -> String {
    if let Some(at) = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        text.truncate(at);
    }
    text
}

/// The `stop` member of a request: absent means none; otherwise an array of at most [`MAX_STOPS`]
/// strings of at most [`MAX_STOP_LEN`] bytes each.
pub fn parse(j: &Json) -> Result<Vec<String>, String> {
    let Some(v) = j.get("stop") else {
        return Ok(Vec::new());
    };
    let Json::Arr(items) = v else {
        return Err("'stop' must be an array of strings".into());
    };
    if items.len() > MAX_STOPS {
        return Err(format!("at most {MAX_STOPS} stop strings"));
    }
    items
        .iter()
        .map(|s| match s.as_str() {
            Some(s) if s.len() <= MAX_STOP_LEN => Ok(s.to_string()),
            Some(_) => Err(format!("a stop string is over {MAX_STOP_LEN} bytes")),
            None => Err("'stop' must be an array of strings".into()),
        })
        .collect()
}

#[cfg(test)]
mod tests;
