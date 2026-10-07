// Shared by the int8 and binary acceptance tests (included with `include!`).

/// Mean over queries of the share of the true top 10 found among the codec's top
/// `candidates`, and the mean absolute estimate error with the mean absolute score.
fn codec_quality<C: Codec>(
    codec: &C,
    metric: Metric,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    candidates: usize,
) -> (f64, f64, f64) {
    let codes: Vec<Vec<u8>> = base
        .iter()
        .map(|v| {
            let mut out = vec![0u8; codec.code_len()];
            codec.encode(v, &mut out);
            out
        })
        .collect();
    let mut recall = 0.0;
    let mut err = 0.0;
    let mut magnitude = 0.0;
    for q in queries {
        let prepared = codec.prepare(q);
        let estimates: Vec<f32> = codes.iter().map(|c| codec.score(&prepared, c)).collect();
        assert!(estimates.iter().all(|e| e.is_finite()), "finite estimates");
        let truth = true_top(metric, base, q, 10);
        let found = top_by(base.len(), candidates, |i| estimates[i]);
        recall += overlap(&truth, &found);
        for (i, v) in base.iter().enumerate() {
            let s = canonical::score(oracle_metric(metric), q, v);
            err += (estimates[i] - s).abs() as f64;
            magnitude += s.abs() as f64;
        }
    }
    let nq = queries.len() as f64;
    let pairs = nq * base.len() as f64;
    (recall / nq, err / pairs, magnitude / pairs)
}

fn symmetric_and_deterministic<C: Codec + PartialEq + std::fmt::Debug>(
    train: impl Fn(&[f32]) -> C,
    from_bytes: impl Fn(&[u8]) -> mhr_vector::Result<C>,
    base: &[Vec<f32>],
) {
    let sample: Vec<f32> = base.concat();
    let a = train(&sample);
    let b = train(&sample);
    assert_eq!(a, b, "training is deterministic");
    assert_eq!(a.to_bytes(), b.to_bytes());
    let back = from_bytes(&a.to_bytes()).unwrap();
    assert_eq!(back, a, "from_bytes(to_bytes) round-trips");
    assert!(from_bytes(&[1, 2, 3]).is_err(), "garbage parameters are refused");
    let enc = |v: &[f32]| {
        let mut out = vec![0u8; a.code_len()];
        a.encode(v, &mut out);
        out
    };
    for xs in base.iter().take(30) {
        let x = enc(xs);
        assert_eq!(x, enc(xs), "encoding is deterministic");
        for ys in base.iter().take(30) {
            let y = enc(ys);
            let s1 = a.score_codes(&x, &y);
            let s2 = a.score_codes(&y, &x);
            assert!(s1.is_finite());
            assert_eq!(s1.to_bits(), s2.to_bits(), "score_codes is symmetric");
        }
    }
}

fn degenerate_inputs_do_not_break<C: Codec>(train: impl Fn(&[f32], usize) -> C) {
    // One vector; identical vectors; a constant dimension.
    let dim = 24;
    let one: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).sin()).collect();
    let normalized = canonical::normalize(&one).unwrap();
    for sample in [normalized.clone(), [normalized.clone(), normalized.clone(), normalized.clone()].concat()] {
        let c = train(&sample, dim);
        let mut code = vec![0u8; c.code_len()];
        c.encode(&normalized, &mut code);
        let q = c.prepare(&normalized);
        assert!(c.score(&q, &code).is_finite());
        assert!(c.score_codes(&code, &code).is_finite());
    }
}
