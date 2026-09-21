//! Pippenger's affine wNAF vs projective wNAF MSM
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test batch_affine_vs_projective -- --ignored --nocapture`
//! and with `--features ark-ec/parallel` for the parallel numbers.

use ark_ec::{
    scalar_mul::{sw_pippenger::msm_batch_affine_bigint, variable_base::msm_bigint_wnaf},
    CurveGroup, VariableBaseMSM,
};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn batch_affine_vs_projective() {
    #[cfg(feature = "parallel")]
    println!("parallel");
    #[cfg(not(feature = "parallel"))]
    println!("serial");
    let bit_sizes: Vec<u32> = std::env::var("MSM_SIZES")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![9, 10, 11, 12, 13, 14, 15, 16]);
    for bit_size in bit_sizes {
        compare(1 << bit_size, bit_size);
    }
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

fn compare(n: usize, bit_size: u32) {
    let rng = &mut test_rng();
    let bases: Vec<GAffine> = G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();

    let expected: G = msm_bigint_wnaf(&bases, &bigints);
    assert_eq!(msm_batch_affine_bigint::<PallasConfig>(&bases, &bigints), expected);
    assert_eq!(G::msm_unchecked(&bases, &scalars), expected);

    let reps = 100;
    let mut ratios = Vec::with_capacity(reps);
    let (mut min_old, mut min_new) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = core::hint::black_box(msm_bigint_wnaf::<G>(&bases, &bigints));
        let old = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = core::hint::black_box(msm_batch_affine_bigint::<PallasConfig>(&bases, &bigints));
        let new = t.elapsed().as_secs_f64();
        ratios.push(old / new);
        min_old = min_old.min(old);
        min_new = min_new.min(new);
    }
    println!(
        "n = 2^{bit_size:<2}: projective wnaf {:8.2} ms, batch affine {:8.2} ms (min), {:.2}x min, {:.2}x median, {:.2}x worst",
        min_old * 1e3,
        min_new * 1e3,
        min_old / min_new,
        median(ratios.clone()),
        ratios.iter().cloned().fold(f64::MAX, f64::min),
    );
}
