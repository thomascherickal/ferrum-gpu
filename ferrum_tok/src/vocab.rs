// Provenance: copied from ferrum-cpu ferrum_core/src/tokenizer/gguf_io.rs (vocabulary arrays only) at commit 0bb0f92b7e71969f11a49bd80f346140c45f2983 (2026-11-11).
// After the copy this file is ours; it is not kept in sync.
//! GGUF vocabulary arrays and standalone tokenizer files.
use crate::error::{InferError, Result};
use crate::pretokenize::byte_to_unicode;
use crate::{ByteBpeTokenizer, BASE_VOCAB};
use std::collections::HashMap;

impl ByteBpeTokenizer {
    /// `tokenizer.ggml.tokens`: each token's bytes mapped through
    /// [`byte_to_unicode`]; special tokens by name.
    pub fn gguf_tokens(&self) -> Vec<String> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| match i.checked_sub(n_text) {
                Some(s) => self.special[s].clone(),
                None => self.vocab[i].iter().map(|&b| byte_to_unicode(b)).collect(),
            })
            .collect()
    }

    /// `tokenizer.ggml.merges` in rank order: `"<left> <right>"`.
    pub fn gguf_merges(&self) -> Vec<String> {
        let toks = self.gguf_tokens();
        self.merges
            .iter()
            .map(|&(a, b)| format!("{} {}", toks[a as usize], toks[b as usize]))
            .collect()
    }

    /// `tokenizer.ggml.token_type`: 1 = normal, 3 = control (specials).
    pub fn gguf_token_types(&self) -> Vec<i32> {
        let n_text = BASE_VOCAB + self.merges.len();
        (0..self.vocab.len())
            .map(|i| if i < n_text { 1 } else { 3 })
            .collect()
    }

    /// Rebuild from the three GGUF arrays (inverse of the exporters). Expects
    /// Ferrum's layout: 256 byte tokens, then one token per merge in rank
    /// order, then control tokens. Errors on any deviation, except that the
    /// `token_type` of the text tokens (the first `256 + merges`) is not
    /// inspected: foreign GPT-2 GGUFs label them inconsistently (normal, byte,
    /// unused) and the `tokens`/`merges` arrays alone define them. Only the
    /// tokens after the merges must be control (3).
    pub fn from_gguf_vocab(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
    ) -> Result<Self> {
        let bad = |m: String| InferError::Format(format!("gguf vocab: {m}"));
        let n_text = BASE_VOCAB + merges.len();
        if tokens.len() < n_text || token_types.len() != tokens.len() {
            return Err(bad(format!(
                "{} tokens / {} types for {} merges",
                tokens.len(),
                token_types.len(),
                merges.len()
            )));
        }
        let ids: HashMap<&str, u32> = tokens[..n_text]
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i as u32))
            .collect();
        let mut tok = Self::byte_level();
        for (b, t) in tokens[..BASE_VOCAB].iter().enumerate() {
            if *t != byte_to_unicode(b as u8).to_string() {
                return Err(bad(format!("token {b} is not byte {b}")));
            }
        }
        for (i, m) in merges.iter().enumerate() {
            // llama.cpp: split at the first space searching from index 1.
            let from = m.chars().next().map_or(0, char::len_utf8);
            let (l, r) = m[from..]
                .find(' ')
                .map(|j| (&m[..from + j], &m[from + j + 1..]))
                .ok_or_else(|| bad(format!("bad merge {m:?}")))?;
            let (Some(&a), Some(&b)) = (ids.get(l), ids.get(r)) else {
                return Err(bad(format!("merge {m:?} has a part that is not a token")));
            };
            if a as usize >= BASE_VOCAB + i || b as usize >= BASE_VOCAB + i {
                return Err(bad(format!("merge {m:?} uses a later token")));
            }
            let merged = [tok.vocab[a as usize].clone(), tok.vocab[b as usize].clone()].concat();
            let want: String = merged.iter().map(|&x| byte_to_unicode(x)).collect();
            if tokens[BASE_VOCAB + i] != want {
                return Err(bad(format!("token {} is not merge {m:?}", BASE_VOCAB + i)));
            }
            tok.vocab.push(merged);
            tok.merges.push((a, b));
        }
        for (k, (name, ty)) in tokens[n_text..]
            .iter()
            .zip(&token_types[n_text..])
            .enumerate()
        {
            if *ty != 3 {
                return Err(bad(format!("token {name:?} after merges is not control")));
            }
            if tok.add_special_token(name)? != n_text + k {
                return Err(bad(format!("duplicate special token {name:?}")));
            }
        }
        tok.rebuild_ranks();
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::{small_tok, trained_tok};
    use crate::ByteBpeTokenizer;

    fn parts(t: &ByteBpeTokenizer) -> (Vec<String>, Vec<String>, Vec<i32>) {
        (t.gguf_tokens(), t.gguf_merges(), t.gguf_token_types())
    }

    #[test]
    fn arrays_round_trip_with_and_without_specials() {
        for t in [small_tok(), trained_tok(), ByteBpeTokenizer::byte_level()] {
            let (tok, mer, ty) = parts(&t);
            assert_eq!(tok.len(), t.vocab_size());
            assert_eq!(
                mer.len(),
                t.vocab_size() - 256 - (tok.len() - 256 - mer.len())
            );
            let back = ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty).unwrap();
            assert_eq!(back.encode_state(), t.encode_state());
            let text = "abab cdcd ünï low lower";
            assert_eq!(back.encode(text), t.encode(text));
        }
    }

    #[test]
    fn specials_are_control_tokens_after_the_merges() {
        let t = trained_tok();
        let (tok, mer, ty) = parts(&t);
        assert_eq!(tok.last().unwrap(), "<|endoftext|>");
        assert_eq!(*ty.last().unwrap(), 3);
        assert!(ty[..256 + mer.len()].iter().all(|&x| x == 1));
    }

    #[test]
    fn rejects_inconsistent_arrays() {
        let (tok, mer, ty) = parts(&small_tok());
        let e = |r: crate::error::Result<ByteBpeTokenizer>| r.expect_err("must fail").to_string();
        // types length differs from tokens
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty[1..])).contains("types"));
        // byte token wrong
        let mut bad = tok.clone();
        bad.swap(65, 66);
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&bad, &mer, &ty)).contains("is not byte"));
        // malformed merge
        let mut badm = mer.clone();
        badm[0] = "nospace".into();
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &badm, &ty)).contains("bad merge"));
        // merge part that is not a token
        let mut badp = mer.clone();
        badp[0] = "zzzz yyyy".into();
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&tok, &badp, &ty)).contains("not a token"));
        // merge that uses a later token
        let mut late = mer.clone();
        late.swap(0, 1);
        assert!(ByteBpeTokenizer::from_gguf_vocab(&tok, &late, &ty).is_err());
        // token after the merges that is not control
        let (t2, m2, mut y2) = parts(&trained_tok());
        *y2.last_mut().unwrap() = 1;
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&t2, &m2, &y2)).contains("not control"));
        // duplicate special token
        let (mut t3, m3, mut y3) = parts(&trained_tok());
        t3.push("<|endoftext|>".into());
        y3.push(3);
        assert!(e(ByteBpeTokenizer::from_gguf_vocab(&t3, &m3, &y3)).contains("duplicate"));
    }

    #[test]
    fn foreign_text_token_types_are_not_inspected() {
        let (tok, mer, mut ty) = parts(&small_tok());
        for x in ty.iter_mut().take(256 + mer.len()) {
            *x = 6; // another writer's label for text tokens; specials stay control
        }
        let r = ByteBpeTokenizer::from_gguf_vocab(&tok, &mer, &ty);
        assert!(r.is_ok(), "{:?}", r.err());
    }
}
