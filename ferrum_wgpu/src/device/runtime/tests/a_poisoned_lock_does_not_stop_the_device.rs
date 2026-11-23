use super::*;
use std::sync::Mutex;

/// Make `m` poisoned, as if a thread had panicked while holding it.
fn poison<T>(m: &Mutex<T>) {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _held = m.lock().unwrap();
        panic!("poison the lock");
    }));
    assert!(r.is_err() && m.is_poisoned());
}

const COUNT: &str = "struct U { v: u32, p0: u32, p1: u32, p2: u32 }
    @group(0) @binding(0) var<storage, read_write> o: array<f32>;
    @group(0) @binding(1) var<uniform> u: U;
    @compute @workgroup_size(1) fn main() { o[0] = f32(u.v); }";

#[test]
fn a_poisoned_lock_does_not_stop_the_device() {
    let Some(g) = open_for_test() else { return };
    poison(&g.health);
    poison(&g.pipelines);
    poison(&g.batch);
    g.check().unwrap();
    assert!(!g.batching());
    let p = g.pipeline("test-poisoned-locks", COUNT).unwrap();
    let out = g.zeros(1).unwrap();
    g.begin_batch().unwrap();
    g.dispatch(&p, &[&out], &[9], (1, 1)).unwrap();
    g.end_batch(true).unwrap();
    assert_eq!(g.read(&out).unwrap(), vec![9.0]);
    // The error callback and the device-lost callback also write through the poisoned lock.
    let _ = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::MAX / 2,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(g.check().unwrap_err().to_string().contains("GPU error"));
    g.destroy();
    crate::device::wait_for_loss(&g);
    assert!(g.check().unwrap_err().to_string().contains("device lost"));
}
