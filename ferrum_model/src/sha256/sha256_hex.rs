use super::*;

impl Sha256 {
    /// A hasher over zero bytes.
    pub fn new() -> Self {
        Sha256 {
            h: INIT,
            buf: Vec::with_capacity(128),
            len: 0,
        }
    }
    /// Hash more bytes.
    pub fn update(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        self.buf.extend_from_slice(data);
        let whole = self.buf.len() / 64 * 64;
        for b in self.buf[..whole].chunks_exact(64) {
            compress(&mut self.h, b);
        }
        self.buf.drain(..whole);
    }
    /// The digest as raw bytes.
    pub fn digest(mut self) -> [u8; 32] {
        let bits = self.len * 8;
        self.buf.push(0x80);
        while self.buf.len() % 64 != 56 {
            self.buf.push(0);
        }
        self.buf.extend(bits.to_be_bytes());
        for b in self.buf.chunks_exact(64) {
            compress(&mut self.h, b);
        }
        let mut out = [0u8; 32];
        for (o, x) in out.chunks_exact_mut(4).zip(self.h) {
            o.copy_from_slice(&x.to_be_bytes());
        }
        out
    }
    /// The digest as 64 lowercase hex characters.
    pub fn hex(self) -> String {
        self.digest().iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// The SHA-256 digest of `data` as 64 lowercase hex characters.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut s = Sha256::new();
    s.update(data);
    s.hex()
}
