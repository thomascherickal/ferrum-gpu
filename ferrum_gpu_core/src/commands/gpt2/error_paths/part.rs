use super::*;

impl FailNth {
    pub(crate) fn new(n: usize) -> Self {
        FailNth { n, calls: 0 }
    }
}

impl Write for FailNth {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.calls += 1;
        if self.calls > self.n {
            return Err(std::io::Error::other("output closed"));
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
