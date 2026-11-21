use super::*;
use crate::commands::{gate_with, preflight, sysmem};

/// Open the device the flags name (or the best one) and say which it is.
pub(super) fn open_session(a: &Args, out: &mut dyn Write) -> Result<Session, String> {
    let choice = DeviceChoice::parse(a.get("--device").unwrap_or("auto"))?;
    let threads = a.num(
        "--threads",
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    )?;
    let dec = decide(
        choice,
        Workload::Decode,
        &ferrum_wgpu::Gpu::list(),
        a.get("--gpu-adapter"),
    )?;
    let mut app = Session::new(dec.choice, dec.adapter.as_deref(), threads.max(1))?;
    if dec.note.is_some() {
        app.fallback_note = dec.note;
    }
    say(
        out,
        format!("device: {} ({})", app.backend.name(), app.backend.kind()),
    );
    if let Some(n) = &app.fallback_note {
        say(out, format!("note: {n}"));
    }
    Ok(app)
}

/// Load the model file, pre-flight its buffers against the device, and wrap it as a decoder.
pub(super) fn load_decoder(
    a: &Args,
    app: &Session,
    out: &mut dyn Write,
) -> Result<Arc<dyn Decoder>, String> {
    let path = a.need("--model")?;
    let mem = a
        .get("--gpu-mem-mb")
        .map(|_| a.num("--gpu-mem-mb", 0u64))
        .transpose()?;
    if gpt2::is_gpt2(path) {
        let l = gpt2::load(path)?;
        gate_with(
            &sysmem::System,
            out,
            app,
            &preflight::gpt2_decode_buffers(&l.model),
            mem,
        )?;
        return Ok(Arc::new(Gpt2Decoder::new(app.backend.clone(), l)));
    }
    let model = crate::commands::load(path)?;
    gate_with(
        &sysmem::System,
        out,
        app,
        &preflight::decode_buffers_for(&model, app),
        mem,
    )?;
    Ok(Arc::new(ModelDecoder::new(app.backend.clone(), model)))
}
