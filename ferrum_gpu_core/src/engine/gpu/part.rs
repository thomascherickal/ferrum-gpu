use super::*;

impl Backend for GpuBackend {
    fn kind(&self) -> &'static str {
        "gpu"
    }

    fn name(&self) -> String {
        self.gpu.info().name.clone()
    }

    fn gpt2_engine<'a>(&'a self, w: &Gpt2, qat: bool) -> Result<Box<dyn Engine + 'a>, String> {
        let mut t = GpuTrainer::new(&self.gpu, w, None).map_err(|e| e.to_string())?;
        t.set_qat(qat).map_err(|e| e.to_string())?;
        Ok(Box::new(t))
    }

    fn gpt2_engine_with<'a>(
        &'a self,
        w: &Gpt2,
        qat: bool,
        o: &EngineOptions,
    ) -> Result<Box<dyn Engine + 'a>, String> {
        let mut t = GpuTrainer::new(&self.gpu, w, None).map_err(|e| e.to_string())?;
        t.set_qat(qat).map_err(|e| e.to_string())?;
        t.set_tie(o.tie).map_err(|e| e.to_string())?;
        t.set_dropout(o.dropout, o.seed);
        Ok(Box::new(t))
    }

    fn gpt2_decoder<'a>(&'a self, w: &Gpt2) -> Result<Gpt2Feed<'a>, String> {
        let mut m = GpuGpt2::new(&self.gpu, w, None).map_err(|e| e.to_string())?;
        Ok(Box::new(move |t| m.feed(t).map_err(|e| e.to_string())))
    }

    fn llama_decoder<'a>(&'a self, model: &'a Llama) -> Result<Gpt2Feed<'a>, String> {
        let mut m =
            GpuLlama::new(&self.gpu, model, self.weights, None).map_err(|e| e.to_string())?;
        Ok(Box::new(move |t| m.feed(t).map_err(|e| e.to_string())))
    }

    fn train(
        &self,
        model: &Llama,
        spec: &TrainSpec,
        progress: &Arc<Mutex<Progress>>,
    ) -> Result<Llama, String> {
        spec.validate(&model.cfg)?;
        let mut tr = GpuLlamaTrainer::new(&self.gpu, model, None).map_err(|e| e.to_string())?;
        run_training(spec, progress, &mut |ids, targets, batch, cfg| {
            tr.step(ids, targets, batch, cfg).map_err(|e| e.to_string())
        })?;
        tr.weights_host().map_err(|e| e.to_string())
    }

    fn generate(
        &self,
        model: &Llama,
        prompt: &[usize],
        opts: &GenOpts,
    ) -> Result<Vec<usize>, String> {
        let mut m =
            GpuLlama::new(&self.gpu, model, self.weights, None).map_err(|e| e.to_string())?;
        generate_with(
            &mut |t| m.feed(t).map_err(|e| e.to_string()),
            model.cfg.ctx,
            prompt,
            opts,
        )
    }
}
