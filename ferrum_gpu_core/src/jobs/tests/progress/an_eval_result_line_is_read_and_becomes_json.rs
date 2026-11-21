use super::*;

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
