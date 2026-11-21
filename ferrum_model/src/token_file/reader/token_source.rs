use super::*;

impl TokenSource for TokenFile {
    fn len(&self) -> usize {
        self.tokens
    }

    fn window(&self, start: usize, len: usize) -> Result<Vec<usize>, String> {
        self.read(start, len)
    }
}
