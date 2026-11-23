use super::*;

#[test]
fn a_lock_whose_holder_panicked_is_still_usable() {
    let m = Mutex::new(1);
    let held = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut g = lock(&m);
        *g = 2;
        panic!("poison the lock");
    }));
    assert!(held.is_err() && m.is_poisoned());
    assert_eq!(*lock(&m), 2);
}
