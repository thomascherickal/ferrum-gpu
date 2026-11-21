use super::*;

impl App {
    /// Start a training run on a copy of the current model, holding out the last `val_fraction` of the
    /// text (`0.0` holds nothing out): the run trains on the rest, and its progress reports the loss on
    /// the held-out part before and after. Errors if there is no model, a run is already going, or the
    /// spec does not fit the model.
    pub fn start_training_validated(
        self: &Arc<Self>,
        mut spec: TrainSpec,
        val_fraction: f32,
    ) -> Result<(), String> {
        let (train, val) = {
            let (t, v) = split(&spec.text, val_fraction)?;
            (t.to_vec(), v.to_vec())
        };
        spec.text = train;
        let context = spec.context;
        let model = lock(&self.model)
            .clone()
            .ok_or("create or load a model first")?;
        let progress = {
            let mut job = lock(&self.job);
            if job.state == JobState::Running {
                return Err("a training run is already in progress".into());
            }
            spec.validate(&model.cfg)?;
            job.state = JobState::Running;
            job.progress = Arc::new(Mutex::new(Progress::default()));
            job.progress.clone()
        };
        let app = self.clone();
        std::thread::spawn(move || {
            if !val.is_empty() {
                lock(&progress).val_before = mean_loss(&model, &val, context);
            }
            let result = app.backend.train(&model, &spec, &progress);
            if let (false, Ok(w)) = (val.is_empty(), &result) {
                lock(&progress).val_after = mean_loss(w, &val, context);
            }
            let stopped = lock(&progress).stop;
            let mut job = lock(&app.job);
            job.state = match result {
                Ok(w) => {
                    app.set_model(w);
                    if stopped {
                        JobState::Stopped
                    } else {
                        JobState::Done
                    }
                }
                Err(e) => JobState::Failed(e),
            };
        });
        Ok(())
    }

    /// Replace the current model and move the version.
    pub fn set_model(&self, m: Llama) {
        *lock(&self.model) = Some(m);
        self.version.fetch_add(1, Ordering::SeqCst);
    }

    /// Ask a running job to stop after its current step.
    pub fn stop_training(&self) -> bool {
        let job = lock(&self.job);
        if job.state == JobState::Running {
            lock(&job.progress).stop = true;
            true
        } else {
            false
        }
    }
}

impl App {
    /// Name of the preset whose configuration equals the model's, if any.
    pub fn preset_of(model: &Llama) -> Option<&'static str> {
        ["tiny", "small", "medium"]
            .into_iter()
            .find(|p| preset(p).as_ref() == Some(&model.cfg))
    }
}
