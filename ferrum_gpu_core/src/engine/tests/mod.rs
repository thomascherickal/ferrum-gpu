use super::*;
use ferrum_wgpu::device::open_for_test;

fn pattern(n: usize) -> Vec<u8> {
    b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(n)
        .copied()
        .collect()
}

fn spec(steps: usize) -> TrainSpec {
    TrainSpec {
        text: pattern(600),
        steps,
        lr: 0.01,
        batch: 4,
        context: 24,
        seed: 3,
    }
}

fn shared() -> Arc<Mutex<Progress>> {
    Arc::new(Mutex::new(Progress::default()))
}

fn train_on(backend: &dyn Backend, steps: usize) -> (Llama, Vec<f32>) {
    let progress = shared();
    let w = backend
        .train(&fresh_model("tiny", 7).unwrap(), &spec(steps), &progress)
        .unwrap();
    let losses = progress.lock().unwrap().losses.clone();
    (w, losses)
}

mod generate_with_stops_at_once_when_the_request_is_cancelled;
mod generation_respects_limits_and_the_sampler;
mod presets_are_valid_models_and_unknown_names_are_refused;
