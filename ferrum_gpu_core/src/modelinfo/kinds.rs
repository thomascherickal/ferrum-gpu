//! The tensor types of a GGUF file: their names and how many bytes a number of elements takes.
//! The ids are ggml's. Only the layout is needed here (sizes for display), not the codecs.

/// `(id, name, elements per block, bytes per block)`.
const TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (30, "BF16", 1, 2),
    (39, "MXFP4", 32, 17),
];

/// The name of a tensor type, or `type N` for one this table does not know.
pub fn name(id: u32) -> String {
    TYPES
        .iter()
        .find(|t| t.0 == id)
        .map_or_else(|| format!("type {id}"), |t| t.1.to_string())
}

/// Bytes `elements` values of type `id` take, or `None` for an unknown type or a count that is not a
/// whole number of blocks.
pub fn bytes_of(id: u32, elements: u64) -> Option<u64> {
    let (_, _, per, bytes) = TYPES.iter().find(|t| t.0 == id)?;
    elements
        .is_multiple_of(*per)
        .then(|| elements / per * bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_types_have_names_and_sizes() {
        assert_eq!(name(0), "F32");
        assert_eq!(name(8), "Q8_0");
        assert_eq!(name(14), "Q6_K");
        assert_eq!(name(30), "BF16");
        assert_eq!(bytes_of(0, 10), Some(40));
        assert_eq!(bytes_of(1, 10), Some(20));
        assert_eq!(bytes_of(2, 64), Some(36), "two Q4_0 blocks");
        assert_eq!(bytes_of(8, 32), Some(34));
        assert_eq!(bytes_of(12, 512), Some(288));
    }

    #[test]
    fn unknown_types_and_partial_blocks_have_no_size() {
        assert_eq!(name(999), "type 999");
        assert_eq!(bytes_of(999, 32), None);
        assert_eq!(bytes_of(2, 33), None, "not a whole block");
        assert_eq!(bytes_of(2, 0), Some(0));
    }

    #[test]
    fn every_listed_type_is_distinct() {
        let mut ids: Vec<u32> = TYPES.iter().map(|t| t.0).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), TYPES.len());
        for (_, n, per, bytes) in TYPES {
            assert!(!n.is_empty() && *per > 0 && *bytes > 0);
        }
    }
}
