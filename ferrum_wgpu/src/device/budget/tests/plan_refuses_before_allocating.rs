use super::*;

#[test]
fn plan_refuses_before_allocating() {
    let mut p = BufferPlan::default();
    p.add("w", 600);
    p.add("act", 600);
    assert_eq!(p.total(), 1200);
    let e = p
        .check(
            1000,
            Some(Budget {
                bytes: 1000,
                source: Source::Override,
            }),
        )
        .unwrap_err();
    assert_eq!(
        e,
        PlanError::OverBudget {
            need: 1200,
            have: 1000
        }
    );
    assert!(e.to_string().contains("1200") && e.to_string().contains("1000"));
    assert!(p.check(1000, None).is_ok());
    assert!(p
        .check(
            1000,
            Some(Budget {
                bytes: 1200,
                source: Source::Tool
            })
        )
        .is_ok());
}

#[test]
fn plan_names_a_tensor_over_the_binding_limit() {
    let mut p = BufferPlan::default();
    p.add("tok_emb", 5000);
    let e = p.check(4096, None).unwrap_err();
    assert!(e.to_string().contains("tok_emb") && e.to_string().contains("4096"));
}
