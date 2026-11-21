use super::*;

impl ByteBpeTokenizer {
    /// Learn BPE merges by streaming `reader` (splitting on `\n\n` or [`DOC_SEP`]
    /// boundaries, or flushing after 64 KiB at a safe newline boundary) so a
    /// multi-gigabyte text file can train a tokenizer in bounded memory while
    /// preserving multi-newline runs across lines.
    pub fn train_from_reader<R: std::io::BufRead>(
        mut reader: R,
        vocab_size: usize,
    ) -> Result<Self> {
        if vocab_size <= BASE_VOCAB {
            return Err(InferError::DimMismatch(format!(
                "vocab_size {vocab_size} < {} (byte base vocabulary + EOS)",
                BASE_VOCAB + 1
            )));
        }
        let mut freq: HashMap<String, i64> = HashMap::new();
        let mut total_bytes = 0usize;
        let mut buf = String::new();
        let mut line = String::new();
        let mut add_chunk = |text: &str| {
            for part in text.split(DOC_SEP) {
                for c in pretokenize(part) {
                    if let Some(count) = freq.get_mut(c) {
                        *count += 1;
                    } else {
                        freq.insert(c.to_owned(), 1);
                    }
                }
            }
        };
        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                if !buf.is_empty() {
                    add_chunk(&buf);
                }
                break;
            }
            total_bytes += n;
            buf.push_str(&line);
            // Flush once the buffer exceeds 64 KiB and does not end in whitespace
            // (or when `line` contains `DOC_SEP`), so whitespace runs across lines
            // stay intact until followed by a non-whitespace line.
            if buf.len() >= (1 << 16) && !line.trim_end_matches(['\r', '\n']).is_empty() {
                // Keep the trailing newline in the next buffer so a blank line
                // following this line still groups `\n\n` together.
                let split = buf.len().saturating_sub(usize::from(buf.ends_with('\n')));
                add_chunk(&buf[..split]);
                buf.drain(..split);
            }
        }
        Self::train_from_word_counts(freq, total_bytes, vocab_size)
    }

    /// Learn BPE merges by streaming `path` from disk via [`train_from_reader`](Self::train_from_reader).
    pub fn train_from_file(path: &std::path::Path, vocab_size: usize) -> Result<Self> {
        let f = std::fs::File::open(path)?;
        Self::train_from_reader(std::io::BufReader::with_capacity(1 << 20, f), vocab_size)
    }
}
