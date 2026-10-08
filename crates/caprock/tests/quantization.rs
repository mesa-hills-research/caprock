//! Codes: the f32 codec, codec dispatch and training samples.
#![cfg(feature = "internals")]

mod common;

use caprock::CodeKind;
use caprock::quant::{AnyCodec, Codec, F32Codec, MAX_TRAINING_VECTORS, training_sample};
use caprock::util::SplitMix64;
use caprock_oracle::canonical;
use common::{METRICS, oracle_metric, stored_vectors};

#[test]
fn f32_codes_are_the_vectors_and_score_canonically() {
    for metric in METRICS {
        let dim = 37;
        let codec = F32Codec { metric, dim };
        assert_eq!(codec.code_len(), 4 * dim);
        assert_eq!(codec.kind(), CodeKind::F32);
        assert!(codec.to_bytes().is_empty());
        let vectors = stored_vectors(metric, dim, 50, 1);
        let codes: Vec<Vec<u8>> = vectors
            .iter()
            .map(|v| {
                let mut out = vec![0u8; codec.code_len()];
                codec.encode(v, &mut out);
                assert_eq!(
                    out,
                    v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()
                );
                out
            })
            .collect();
        for (qi, q) in vectors.iter().enumerate().take(10) {
            let prepared = codec.prepare(q);
            for (i, v) in vectors.iter().enumerate() {
                let want = canonical::score(oracle_metric(metric), q, v);
                assert_eq!(codec.score(&prepared, &codes[i]).to_bits(), want.to_bits());
                assert_eq!(
                    codec.score_codes(&codes[qi], &codes[i]).to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

#[test]
fn any_codec_dispatches() {
    let dim = 16;
    let sample: Vec<f32> = stored_vectors(caprock::Metric::Dot, dim, 30, 2).concat();
    let any = AnyCodec::train(CodeKind::F32, caprock::Metric::Dot, dim, &sample, 0).unwrap();
    assert!(matches!(any, AnyCodec::F32(_)));
    assert_eq!(any.kind(), CodeKind::F32);
    assert_eq!(any.code_len(), 4 * dim);
    let mut out = vec![0u8; any.code_len()];
    any.encode(&sample[..dim], &mut out);
    assert_eq!(&out[..4], &sample[0].to_le_bytes());
    let back =
        AnyCodec::from_bytes(CodeKind::F32, caprock::Metric::Dot, dim, &any.to_bytes()).unwrap();
    assert_eq!(back, any);
}

#[test]
fn training_samples() {
    assert_eq!(training_sample(0, 1), Vec::<usize>::new());
    assert_eq!(training_sample(5, 1), vec![0, 1, 2, 3, 4]);
    assert_eq!(
        training_sample(MAX_TRAINING_VECTORS, 1).len(),
        MAX_TRAINING_VECTORS
    );
    let n = MAX_TRAINING_VECTORS * 3 + 17;
    let s = training_sample(n, 99);
    let want: Vec<usize> = SplitMix64::new(99)
        .sample(n as u64, MAX_TRAINING_VECTORS)
        .into_iter()
        .map(|x| x as usize)
        .collect();
    assert_eq!(s, want);
    assert_eq!(s, training_sample(n, 99));
}
