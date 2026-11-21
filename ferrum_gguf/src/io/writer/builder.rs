use super::*;

impl GgufBuilder {
    /// An empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or append) a metadata entry.
    pub fn meta(&mut self, key: &str, v: Value) {
        self.kv.push((key.to_string(), v));
    }

    /// Add an F32 tensor with GGML dims (innermost first).
    pub fn tensor_f32(&mut self, name: &str, dims: &[u64], data: &[f32]) {
        let bytes = encode(GGML_F32, data).expect("F32 encodes any length");
        self.tensor_raw(name, dims, GGML_F32, bytes);
    }

    /// Add a tensor already encoded as `ggml_type`.
    pub fn tensor_raw(&mut self, name: &str, dims: &[u64], ggml_type: u32, bytes: Vec<u8>) {
        self.tensors
            .push((name.to_string(), dims.to_vec(), ggml_type, bytes));
    }

    /// Serialise the file.
    pub fn into_bytes(self) -> Vec<u8> {
        let mut head = Vec::new();
        head.extend_from_slice(&MAGIC.to_le_bytes());
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
        let mut alignment = DEFAULT_ALIGNMENT;
        for (k, v) in &self.kv {
            put_str(&mut head, k);
            head.extend_from_slice(&tag(v).to_le_bytes());
            put_value(&mut head, v);
            if let ("general.alignment", Value::U32(a)) = (k.as_str(), v) {
                alignment = u64::from(*a).max(1);
            }
        }
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (name, dims, ty, bytes) in &self.tensors {
            put_str(&mut head, name);
            head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            dims.iter()
                .for_each(|d| head.extend_from_slice(&d.to_le_bytes()));
            head.extend_from_slice(&ty.to_le_bytes());
            head.extend_from_slice(&offset.to_le_bytes());
            data.extend_from_slice(bytes);
            let padded = (bytes.len() as u64).div_ceil(alignment) * alignment;
            data.resize(data.len() + (padded - bytes.len() as u64) as usize, 0);
            offset += padded;
        }
        let pad = (head.len() as u64).div_ceil(alignment) * alignment - head.len() as u64;
        head.resize(head.len() + pad as usize, 0);
        head.extend_from_slice(&data);
        head
    }
}
