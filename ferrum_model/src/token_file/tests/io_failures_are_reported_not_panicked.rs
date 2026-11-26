use super::*;
use std::io::BufWriter;
use std::sync::Mutex;

fn written(name: &str) -> String {
    let path = tmp(name);
    write(&path, &meta(10, None), |s| s.push(&[1, 2, 3])).unwrap();
    path
}

fn handle_at(path: &str, data_start: u64) -> TokenFile {
    TokenFile {
        file: Mutex::new(std::fs::File::open(path).unwrap()),
        data_start,
        width: 2,
        tokens: 1,
        vocab_size: 10,
        documents: 1,
        eos_id: None,
        tokenizer_sha256: String::new(),
    }
}

#[test]
fn a_directory_is_not_a_token_file() {
    let e = TokenFile::open(&tmp("a_dir").replace("a_dir", "")).unwrap_err();
    assert!(e.contains("ferrum_tokfile"), "the path is named: {e}");
}

#[test]
fn a_tensor_offset_that_overflows_is_refused() {
    let e = reader::tensor_start(8, u64::MAX, "f.tok").unwrap_err();
    assert_eq!(e, "f.tok: tensor offset overflows");
}

#[test]
fn a_file_cut_short_after_opening_fails_the_read() {
    let path = written("cut.tok.gguf");
    let f = TokenFile::open(&path).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    let e = f.read(0, 3).unwrap_err();
    assert!(e.starts_with("reading the token file"), "{e}");
}

#[test]
fn an_offset_the_file_cannot_seek_to_is_an_error() {
    let f = handle_at(&written("seek.tok.gguf"), 1 << 63);
    assert!(f.read(0, 1).is_err());
}

#[test]
fn a_poisoned_handle_lock_does_not_stop_reads() {
    let path = written("poison.tok.gguf");
    let f = TokenFile::open(&path).unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = f.file.lock().unwrap();
        panic!("poison the lock");
    }));
    assert!(poisoned.is_err() && f.file.is_poisoned());
    assert_eq!(f.read(0, 3).unwrap(), vec![1, 2, 3]);
}

#[test]
fn a_sink_whose_file_rejects_writes_reports_the_error() {
    let mut ro = std::fs::File::open(written("ro.tok.gguf")).unwrap();
    let mut sink = TokenSink {
        w: BufWriter::with_capacity(1, &mut ro),
        vocab: 10,
        eos: None,
        wide: false,
        tokens: 0,
        eos_seen: 0,
    };
    assert!(sink.push(&[1]).is_err());
}
