use super::*;

impl Server {
    /// The environment the server and every client of it share.
    pub(crate) fn env(&self) -> Vec<(&'static str, String)> {
        let mut e = vec![
            ("HOME", self.home.display().to_string()),
            ("USERPROFILE", self.home.display().to_string()),
            ("OLLAMA_HOST", self.host.clone()),
            (
                "OLLAMA_MODELS",
                self.home.join("models").display().to_string(),
            ),
            ("OLLAMA_DEBUG", "1".into()),
        ];
        match self.mode {
            Mode::Cpu => {
                e.push(("CUDA_VISIBLE_DEVICES", "-1".into()));
                e.push(("HIP_VISIBLE_DEVICES", "-1".into()));
                e.push(("OLLAMA_LLM_LIBRARY", "cpu".into()));
            }
            Mode::Cuda => {}
            Mode::Vulkan => {
                e.push(("OLLAMA_VULKAN", "1".into()));
                e.push(("CUDA_VISIBLE_DEVICES", "-1".into()));
            }
        }
        e
    }

    /// Register `model` as `name` through `ferrum-gpu export-ollama` against this server.
    pub(crate) fn export(&self, model: &Path, name: &str) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_ferrum-gpu"));
        cmd.args(["export-ollama", "--model"])
            .arg(model)
            .args(["--name", name, "--bin"])
            .arg(&self.bin);
        for (k, v) in self.env() {
            cmd.env(k, v);
        }
        cmd.output().expect("run export-ollama")
    }

    /// Ollama's raw greedy continuation of the prompt with the registered `name`. `cpu_only` adds
    /// `num_gpu: 0` to the request; the GPU tests leave the choice to Ollama.
    pub(crate) fn raw_generate(&self, name: &str, cpu_only: bool) -> String {
        let gpu = if cpu_only { r#","num_gpu":0"# } else { "" };
        let body = format!(
            r#"{{"model":"{name}","prompt":"{PROMPT}","raw":true,"stream":false,"options":{{"temperature":0,"seed":1,"num_predict":{NEW_TOKENS},"num_thread":1{gpu}}}}}"#
        );
        let (code, reply) = http(&self.host, "POST", "/api/generate", &body).expect("reach ollama");
        assert_eq!(code, 200, "{reply}");
        let j = Json::parse(&reply).unwrap_or_else(|e| panic!("{e}: {reply}"));
        j.get("response")
            .and_then(Json::as_str)
            .expect("a response")
            .to_string()
    }

    /// `(size, size_vram)` in bytes of the loaded model `name`, from `GET /api/ps`.
    pub(crate) fn loaded_sizes(&self, name: &str) -> Option<(u64, u64)> {
        let (_, body) = http(&self.host, "GET", "/api/ps", "")?;
        let j = Json::parse(&body).ok()?;
        let Some(Json::Arr(models)) = j.get("models") else {
            return None;
        };
        let m = models.iter().find(|m| {
            m.get("name")
                .and_then(Json::as_str)
                .is_some_and(|n| n.starts_with(name))
        })?;
        let size = |k: &str| m.get(k).and_then(Json::as_f64).map(|v| v as u64);
        Some((size("size")?, size("size_vram")?))
    }

    /// The server's own log so far.
    pub(crate) fn log_text(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }
}

pub(crate) fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
