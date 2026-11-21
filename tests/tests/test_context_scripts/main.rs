#![forbid(unsafe_code)]
// bash scripts with unix file modes: not built on Windows (the independence guard still is)
#![cfg(unix)]
//! Tests for `scripts/statusline.sh` and `scripts/handoff_check.sh`, the machinery behind
//! CLAUDE.md rule 12 (write handoff.md at 98% context). They run the real scripts under bash with
//! jq, a stub delegate and temp directories; without jq they skip with a printed reason.
use std::io::Write;
use std::process::{Command, Stdio};

mod fixtures;
mod runners;
use fixtures::*;
use runners::*;

mod statusline_falls_back_to_a_plain_figure_when_there_is_no_delegate;
mod statusline_records_used_context_from_remaining_and_delegates_the_display;
mod the_hook_ignores_a_stale_figure_and_garbage;
