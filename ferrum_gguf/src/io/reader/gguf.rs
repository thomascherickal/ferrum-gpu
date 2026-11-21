use super::*;

/// A parsed GGUF file (keeps the bytes; tensors decode on demand).
pub struct Gguf {
    /// Metadata in file order.
    pub kv: Vec<(String, Value)>,
    /// Tensor directory in file order.
    pub tensors: Vec<TensorInfo>,
    pub(crate) index: HashMap<String, usize>,
    pub(crate) data_start: usize,
    pub(crate) bytes: Vec<u8>,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gguf {{ {} keys, {} tensors, {} bytes }}",
            self.kv.len(),
            self.tensors.len(),
            self.bytes.len()
        )
    }
}

pub(crate) struct Cur<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) p: usize,
}
