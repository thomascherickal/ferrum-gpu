use super::*;
use crate::gpt2::Config;
use crate::matmul::rand_vec;

fn cfg(dim: usize) -> Config {
    Config {
        vocab: 16,
        ctx: 8,
        dim,
        heads: 2,
        layers: 2,
        hidden: dim * 2,
    }
}

#[test]
fn eligibility_needs_whole_blocks() {
    assert!(eligible(32) && eligible(64) && eligible(4096));
    assert!(!eligible(0) && !eligible(31) && !eligible(33) && !eligible(8));
}

#[test]
fn a_snapped_block_is_a_fixed_point_with_at_most_127_distinct_levels() {
    let blk = rand_vec(32, 1);
    let once = snap_block(&blk);
    assert_ne!(once, blk, "random values are not already on the grid");
    assert_eq!(snap_block(&once), once, "idempotent");
    let step = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs())) / 127.0;
    for (a, b) in blk.iter().zip(&once) {
        assert!((a - b).abs() <= step * 0.51 + 1e-6);
    }
    assert_eq!(snap_block(&[0.0; 32]), vec![0.0; 32]);
}

#[test]
fn rows_and_cols_snap_the_right_blocks() {
    // rows: two blocks of 32 along the flat data.
    let mut r = rand_vec(64, 2);
    let want: Vec<f32> = r.chunks(32).flat_map(snap_block).collect();
    snap_rows(&mut r, 32);
    assert_eq!(r, want);
    // cols: a [64, 3] matrix, blocks run down each column.
    let mut m = rand_vec(64 * 3, 3);
    let orig = m.clone();
    snap_cols(&mut m, 64, 3);
    for j in 0..3 {
        for b in 0..2 {
            let col: Vec<f32> = (0..32).map(|k| orig[(b * 32 + k) * 3 + j]).collect();
            let want = snap_block(&col);
            for (k, w) in want.iter().enumerate() {
                assert_eq!(m[(b * 32 + k) * 3 + j], *w, "col {j} block {b} elem {k}");
            }
        }
    }
}

#[test]
fn ineligible_shapes_are_left_untouched() {
    let mut a = rand_vec(40, 4);
    let o = a.clone();
    snap_rows(&mut a, 8);
    snap_cols(&mut a, 8, 5);
    snap_cols(&mut a, 0, 5);
    assert_eq!(a, o);
}

#[test]
fn fake_quantize_touches_only_what_a_q8_0_export_quantises() {
    let w = Gpt2::random(cfg(32), 5);
    let q = w.fake_quantize();
    assert_ne!(q.tok, w.tok);
    assert_ne!(q.head, w.head);
    assert_ne!(q.blocks[1].w2, w.blocks[1].w2);
    assert_eq!(q.pos, w.pos, "position embeddings stay F32");
    assert_eq!(q.blocks[0].bq, w.blocks[0].bq);
    assert_eq!(q.blocks[0].ln1_g, w.blocks[0].ln1_g);
    assert_eq!(q.lnf_b, w.lnf_b);
    assert_eq!(q.fake_quantize(), q, "idempotent");
    // A width that is not a multiple of 32 is stored F32, so nothing changes.
    let small = Gpt2::random(cfg(8), 6);
    assert_eq!(small.fake_quantize(), small);
}

#[test]
fn the_snapped_values_are_exactly_what_a_q8_0_gguf_would_hold() {
    // Property QAT depends on: quantising the snapped model again loses nothing, because the
    // snap used the very same encoder (tested end to end in ferrum_model's export test).
    let w = Gpt2::random(cfg(32), 7).fake_quantize();
    let mut again = w.clone();
    snap_rows(&mut again.tok, 32);
    assert_eq!(again.tok, w.tok);
}
