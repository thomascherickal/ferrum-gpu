use super::*;
use crate::budget::Source;
use crate::device::open_for_test;

mod a_batch_runs_commands_in_order_in_one_submit;
mod copy_f32_moves_a_slice_on_the_device_and_checks_bounds;
mod groups_cover_and_respect_the_limit;
mod wgpu_errors_map_to_readable_messages;
