use super::*;

/// Which hardware the server may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// No GPU: `CUDA_VISIBLE_DEVICES=-1` and the CPU runner.
    Cpu,
    /// The GPU through CUDA (Ollama's default choice).
    Cuda,
    /// The GPU through Vulkan (`OLLAMA_VULKAN=1`, CUDA hidden).
    Vulkan,
}

/// A running server, ended with its process group when dropped.
pub struct Server {
    pub(super) _group: Group,
    pub host: String,
    pub home: PathBuf,
    pub bin: PathBuf,
    pub mode: Mode,
    pub log: PathBuf,
}
