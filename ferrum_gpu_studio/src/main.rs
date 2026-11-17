#![forbid(unsafe_code)]
// no console window on Windows release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ferrum_gpu_studio_lib::run();
}
