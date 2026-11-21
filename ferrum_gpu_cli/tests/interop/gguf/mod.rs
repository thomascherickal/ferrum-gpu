//! The GGUF files we write, checked against what llama.cpp's loader requires. No tool is run: this
//! is the container (header, alignment, offsets), the metadata keys and the tensor names, shapes and
//! types of a `gpt2` model, so a regression shows up here before it shows up as a failed load.
use crate::support::model::fixture;
use ferrum_gguf::Gguf;

const F32: u32 = 0;
const Q4_0: u32 = 2;
const Q8_0: u32 = 8;

fn load(p: &std::path::Path) -> (Vec<u8>, Gguf) {
    let bytes = std::fs::read(p).unwrap();
    let g = Gguf::parse(bytes.clone()).unwrap();
    (bytes, g)
}

mod a_tied_head_file_stores_the_embedding_once;
mod every_tensor_llama_cpp_expects_is_there_with_ggml_dimension_order;
mod the_header_is_gguf_version_3;
