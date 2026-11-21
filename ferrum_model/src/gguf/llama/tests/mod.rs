use super::*;
use ferrum_gguf::{GGML_F32, GGML_Q4_0, GGML_Q8_0};

fn cfg(ty: RopeType) -> Config {
    Config {
        vocab: 13,
        ctx: 12,
        dim: 32,
        heads: 4,
        kv_heads: 2,
        head_dim: 8,
        layers: 2,
        hidden: 64,
        rope_dim: 8,
        rope_base: 10000.0,
        rope_type: ty,
        eps: 1e-5,
    }
}

fn parse(bytes: Vec<u8>) -> Result<Llama, GgufError> {
    from_gguf(&Gguf::parse(bytes)?)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn set(key: &'static str, v: Value) -> impl Fn(&mut Meta) {
    move |kv| {
        kv.retain(|(k, _)| k != key);
        kv.push((key.to_string(), v.clone()));
    }
}

fn refused(m: &Llama, edit: &dyn Fn(&mut Meta), skip: &[&str]) -> String {
    parse(write(m, Quant::F32, edit, skip)).unwrap_err().0
}

mod f32_round_trip_is_exact_for_both_architectures;
mod hostile_metadata_is_refused_before_it_can_size_an_allocation;
mod missing_or_misshapen_tensors_are_named_in_the_error;
mod quantised_round_trips_stay_close_and_q8_beats_q4;
mod scaling_policy_follows_llama_cpp;
