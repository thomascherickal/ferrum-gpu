use super::*;

/// Write a token file to `path` in one pass with O(1) memory: `produce` pushes ids into the
/// [`TokenSink`] (a placeholder header, then the ids through a 1 MiB buffer, then the header
/// rewritten with the real counts). The file appears at `path` only when complete (temp file and
/// rename); an error from `produce` leaves any existing file untouched.
///
/// Ceiling: a `u32` vocabulary and a `u64` token count; 2 bytes per token while the vocabulary is
/// at most 32768, else 4.
pub fn write(
    path: &str,
    meta: &TokenFileMeta,
    produce: impl FnOnce(&mut TokenSink) -> Result<(), String>,
) -> Result<TokenFileStats, String> {
    write_boxed(path, meta, Box::new(produce))
}

type Producer<'a> = Box<dyn FnOnce(&mut TokenSink) -> Result<(), String> + 'a>;

/// [`write`] with the producer boxed, so its body is compiled once and not per closure type.
fn write_boxed(
    path: &str,
    meta: &TokenFileMeta,
    produce: Producer<'_>,
) -> Result<TokenFileStats, String> {
    if meta.vocab_size == 0 || meta.vocab_size > i32::MAX as usize {
        return Err(format!("vocab_size {} out of range", meta.vocab_size));
    }
    if meta.eos_id.is_some_and(|e| e as usize >= meta.vocab_size) {
        return Err("eos_id is outside the vocabulary".into());
    }
    let (ty, width) = element_type(meta.vocab_size);
    let tmp = format!("{path}.part");
    let io = |e: std::io::Error| format!("{path}: {e}");
    let mut f = std::fs::File::create(&tmp).map_err(io)?;
    let outcome = (|| -> Result<TokenFileStats, String> {
        let placeholder = header_bytes(meta, 0, 0);
        f.write_all(&placeholder).map_err(io)?;
        let mut sink = TokenSink {
            w: BufWriter::with_capacity(1 << 20, &mut f),
            vocab: meta.vocab_size,
            eos: meta.eos_id,
            wide: ty == GGML_I32,
            tokens: 0,
            eos_seen: 0,
        };
        produce(&mut sink)?;
        sink.w.flush().map_err(io)?;
        let (n, docs) = (
            sink.tokens,
            if sink.tokens == 0 {
                0
            } else {
                sink.eos_seen + 1
            },
        );
        drop(sink);
        let header = header_bytes(meta, n, docs);
        f.seek(SeekFrom::Start(0)).map_err(io)?;
        f.write_all(&header).map_err(io)?;
        f.sync_all().map_err(io)?;
        Ok(TokenFileStats {
            tokens: n,
            documents: docs,
            bytes: header.len() as u64 + n * width as u64,
        })
    })();
    drop(f);
    match outcome {
        Ok(stats) => {
            std::fs::rename(&tmp, path).map_err(io)?;
            Ok(stats)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}
