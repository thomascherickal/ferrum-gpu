use super::support::{args, block_until_cancelled, jobs_with, texts, wait, wait_for_line};
use crate::jobs::progress::Event;
use crate::jobs::runner::LineSink;
use crate::jobs::{Job, Jobs, MAX_KEPT, MAX_LINES, MAX_RUNNING};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

mod one_jobs_cancel_flag_does_not_reach_another;
mod the_command_gets_its_name_first_and_its_arguments_after;
mod the_real_command_reports_its_own_errors_as_lines;
