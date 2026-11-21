use super::*;

#[test]
fn a_lost_device_surfaces_through_every_engine_method() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let w = Gpt2::init(c, 43);
    let mut tr = GpuTrainer::new(&g, &w, None).unwrap();
    let snap = Engine::snapshot(&tr).unwrap();
    g.destroy();
    for _ in 0..200 {
        let _ = g.device.poll(wgpu::PollType::Poll);
        if g.check().is_err() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let (ids, targets) = batch_data(&c, 1, 3, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    assert!(Engine::step(&mut tr, &ids, &targets, 1, &sc).is_err());
    assert!(Engine::snapshot(&tr).is_err());
    assert!(Engine::restore(&mut tr, &snap).is_err());
    assert_eq!(Engine::steps(&tr), 0);
}

#[test]
fn the_engine_view_agrees_with_the_inherent_methods() {
    let Some(g) = open_for_test() else { return };
    let c = cfg();
    let mut tr = GpuTrainer::new(&g, &Gpt2::init(c, 44), None).unwrap();
    let (ids, targets) = batch_data(&c, 2, 4, 0);
    let sc = StepConfig {
        adam: Adam::new(0.01),
        clip: 0.0,
    };
    let s = Engine::step(&mut tr, &ids, &targets, 2, &sc).unwrap();
    assert!(s.loss.is_finite());
    assert_eq!(Engine::steps(&tr), 1);
    let snap = Engine::snapshot(&tr).unwrap();
    assert_eq!(snap.steps, 1);
    assert_eq!(snap.weights, tr.weights_host().unwrap());
    Engine::restore(&mut tr, &snap).unwrap();
    assert_eq!(Engine::steps(&tr), 1);
}
