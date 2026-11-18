use crate::jobs::progress::{parse_line, Event};

#[test]
fn a_step_line_is_read_in_full() {
    let e =
        parse_line("step 20/200 loss 3.1234 grad_norm 0.512 tokens/s 4800 elapsed 4.2s eta 38s");
    assert_eq!(
        e,
        Some(Event::Step {
            step: 20,
            total: 200,
            loss: 3.1234,
            grad_norm: 0.512,
            tokens_per_s: 4800.0,
            elapsed: 4.2,
            eta: 38.0
        })
    );
}

#[test]
fn a_validation_line_says_whether_it_was_the_best() {
    assert_eq!(
        parse_line("epoch 3 val loss 2.9876 (best)"),
        Some(Event::Val {
            epoch: 3,
            loss: 2.9876,
            best: true
        })
    );
    assert_eq!(
        parse_line("epoch 4 val loss 3.1000"),
        Some(Event::Val {
            epoch: 4,
            loss: 3.1,
            best: false
        })
    );
}

#[test]
fn other_lines_are_not_events() {
    for l in [
        "",
        "device: gpu",
        "step",
        "step x/y loss 1",
        "step 1/2 loss 1.0", // missing fields
        "step 1/2 loss abc grad_norm 1 tokens/s 1 elapsed 1s eta 1s",
        "epoch 3 train loss 1.0",
        "epoch x val loss 1.0",
        "epoch 3 val loss y",
        "trained 200 steps, final loss 0.5",
    ] {
        assert_eq!(parse_line(l), None, "{l:?}");
    }
}

#[test]
fn events_become_the_json_the_page_reads() {
    let step =
        parse_line("step 1/2 loss 1.5 grad_norm 0.25 tokens/s 10 elapsed 1.0s eta 1s").unwrap();
    let j = step.json().to_string();
    for needle in [
        "\"kind\":\"step\"",
        "\"step\":1",
        "\"total\":2",
        "\"loss\":1.5",
        "\"tokens_per_s\":10",
    ] {
        assert!(j.contains(needle), "{needle} in {j}");
    }
    let val = parse_line("epoch 2 val loss 0.5 (best)")
        .unwrap()
        .json()
        .to_string();
    assert!(
        val.contains("\"kind\":\"val\"") && val.contains("\"best\":true"),
        "{val}"
    );
}

#[test]
fn an_eval_result_line_is_read_and_becomes_json() {
    let e = parse_line("tokens: 4096 loss: 2.3456 perplexity: 10.442");
    assert_eq!(
        e,
        Some(Event::Eval {
            tokens: 4096,
            loss: 2.3456,
            perplexity: 10.442,
            bits_per_byte: None
        })
    );
    let j = e.unwrap().json().to_string();
    for needle in [
        "\"kind\":\"eval\"",
        "\"tokens\":4096",
        "\"perplexity\":10.442",
    ] {
        assert!(j.contains(needle), "{needle} in {j}");
    }
    for l in [
        "tokens: x loss: 1 perplexity: 2",
        "tokens: 5 loss: y perplexity: 2",
        "tokens: 5 loss: 1 perplexity: z",
        "tokens: 5 loss: 1",
        "tokens: 5 bad: 1 perplexity: 2",
    ] {
        assert_eq!(parse_line(l), None, "{l:?}");
    }
}

#[test]
fn an_eval_line_with_bits_per_byte_carries_it_and_a_bad_value_is_ignored() {
    let e = parse_line("tokens: 10 loss: 1.5 perplexity: 4.481 bits/byte: 0.8125").unwrap();
    assert_eq!(
        e,
        Event::Eval {
            tokens: 10,
            loss: 1.5,
            perplexity: 4.481,
            bits_per_byte: Some(0.8125)
        }
    );
    assert!(e.json().to_string().contains("\"bits_per_byte\":0.8125"));
    // without it the field is null, and a garbled value does not lose the rest of the result
    let none = parse_line("tokens: 10 loss: 1.5 perplexity: 4.481")
        .unwrap()
        .json()
        .to_string();
    assert!(none.contains("\"bits_per_byte\":null"), "{none}");
    let bad = parse_line("tokens: 10 loss: 1.5 perplexity: 4.481 bits/byte: x");
    assert!(matches!(
        bad,
        Some(Event::Eval {
            bits_per_byte: None,
            ..
        })
    ));
}
