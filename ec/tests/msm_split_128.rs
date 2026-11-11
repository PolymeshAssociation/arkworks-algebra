//! `msm_signed` runs the 65 to 128 bit scalars as their own MSM beside full-width ones once there
//! are at least `max(2^10, 2^9 * threads)` of them, so a parallel build checks it on a two-thread
//! pool.

use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_pallas::{Fr, Projective};
use ark_std::{test_rng, vec::Vec};

fn split_128_matches_naive() {
    let rng = &mut test_rng();
    let n = 1100;
    let bases =
        Projective::normalize_batch(&(0..n).map(|_| Projective::rand(rng)).collect::<Vec<_>>());
    // 1045 scalars of 65 to 128 bits, half negated, and 55 full-width ones.
    let scalars: Vec<Fr> = (0..n)
        .map(|i| {
            if i % 20 == 0 {
                Fr::rand(rng)
            } else {
                let v = Fr::from(u128::rand(rng) | 1 << 127);
                if i % 2 == 0 { v } else { -v }
            }
        })
        .collect();
    let naive: Projective = bases.iter().zip(&scalars).map(|(b, s)| *b * s).sum();
    assert_eq!(Projective::msm(&bases, &scalars).unwrap(), naive);
}

#[test]
fn split_128_beside_full_width() {
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(split_128_matches_naive);
    #[cfg(not(feature = "parallel"))]
    split_128_matches_naive();
}
