//! `tokenize`: encode a text file, or a directory of text files, into a token file.

use crate::cmds::say;
use crate::flags::Args;
use ferrum_model::sha256::sha256_hex;
use ferrum_model::token_file::{write, TokenFileMeta};
use ferrum_model::tokenizer_gguf;
use ferrum_tok::DOC_SEP;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The files to read: `path` itself, or the regular files directly inside it in path order.
fn inputs(path: &str) -> Result<Vec<PathBuf>, String> {
    let p = Path::new(path);
    if !p.is_dir() {
        return Ok(vec![p.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(p)
        .map_err(|e| format!("{path}: {e}"))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|f| f.is_file())
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("{path}: the directory holds no files"));
    }
    Ok(files)
}

/// `tokenize --data FILE|DIR --tokenizer TOK.gguf --out TOKENS.gguf [--doc-sep S]`.
pub fn tokenize(a: &Args, out: &mut dyn Write) -> Result<(), String> {
    let (data, tokp, target) = (a.need("--data")?, a.need("--tokenizer")?, a.need("--out")?);
    let tok_bytes = std::fs::read(tokp).map_err(|e| format!("--tokenizer: {tokp}: {e}"))?;
    let tok = tokenizer_gguf::from_gguf_bytes(&tok_bytes).map_err(|e| format!("{tokp}: {e}"))?;
    let sep = a.get("--doc-sep").unwrap_or(DOC_SEP);
    if sep.is_empty() {
        return Err("--doc-sep must not be empty".into());
    }
    let files = inputs(data)?;
    let meta = TokenFileMeta {
        vocab_size: tok.vocab_size(),
        eos_id: tok.eos_id().map(|e| e as u32),
        tokenizer_sha256: sha256_hex(&tok_bytes),
    };
    let mut bytes_in = 0u64;
    let stats = write(target, &meta, |sink| {
        for (i, f) in files.iter().enumerate() {
            let raw = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
            bytes_in += raw.len() as u64;
            let mut text = String::from_utf8(raw)
                .map_err(|_| format!("{}: the file is not valid UTF-8", f.display()))?;
            if sep != DOC_SEP {
                text = text.replace(sep, DOC_SEP);
            }
            if i > 0 {
                if let Some(eos) = meta.eos_id {
                    sink.push(&[eos])?;
                }
            }
            let ids: Vec<u32> = tok
                .encode_docs(&text)
                .into_iter()
                .map(|x| x as u32)
                .collect();
            sink.push(&ids)?;
        }
        Ok(())
    })?;
    say(
        out,
        format!(
            "tokenized {data} -> {target}: {} tokens, {} documents, vocab {} ({bytes_in} bytes in, {} bytes out)",
            stats.tokens, stats.documents, meta.vocab_size, stats.bytes
        ),
    )
}
