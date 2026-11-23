use super::support::{args, block_until_cancelled, jobs_with, texts, wait, wait_for_line};
use crate::jobs::progress::Event;
use crate::jobs::runner::{started, LineSink};
use crate::jobs::{Job, Jobs, MAX_KEPT, MAX_LINES, MAX_RUNNING};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

mod a_panic_ends_the_job_not_the_process;
mod a_thread_that_cannot_start_is_reported;
mod an_eval_result_line_is_kept_as_the_latest_eval;
mod one_jobs_cancel_flag_does_not_reach_another;
mod stop_raises_the_cancel_flag_the_command_polls_and_the_job_ends_stopped;
mod the_command_gets_its_name_first_and_its_arguments_after;
mod the_real_command_reports_its_own_errors_as_lines;
mod too_many_running_jobs_are_refused_and_room_comes_back;
