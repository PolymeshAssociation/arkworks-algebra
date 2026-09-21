//! The routed small-MSM path against the bucket algorithm it bypasses.
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test msm_small_bench -- --ignored --nocapture`
//! Add `--features=parallel` for measuring `MIN_PARALLEL_SCALARS`

use ark_ec::{scalar_mul::glv::eisenstein::eisenstein_msm, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

const REPS: usize = 11;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn routed_vs_buckets() {
    for n in [1usize, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512] {
        compare(n);
    }
    println!();
    for n in [1usize, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 256] {
        straus(n);
    }
    println!();
    for n in [256usize, 1024, 4096, 16384, 32768, 49152, 65536, 131072] {
        into_bigint_conversion(n);
    }
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

fn compare(n: usize) {
    let rng = &mut test_rng();
    let bases: Vec<GAffine> = G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
    assert_eq!(
        G::msm_unchecked(&bases, &scalars),
        G::msm_bigint(&bases, &bigints)
    );

    let rounds = (1 << 13) / n + 1;
    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        // The bucket path, reached directly so the routing cannot intercept it.
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(G::msm_bigint(&bases, &bigints));
        }
        let old = t.elapsed().as_secs_f64() / rounds as f64;
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(G::msm_unchecked(&bases, &scalars));
        }
        let new = t.elapsed().as_secs_f64() / rounds as f64;
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    println!(
        "n = {n:4}: buckets {:8.2} us, routed {:8.2} us, {:.2}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        median(ratios)
    );
}

/// The Straus ladder over the joint Eisenstein digits against the bucket algorithm, at sizes
/// the routing does not yet send to it.
fn straus(n: usize) {
    let rng = &mut test_rng();
    let bases: Vec<GAffine> = G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let bigints: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
    let expected = G::msm_bigint(&bases, &bigints);
    assert_eq!(
        eisenstein_msm::<PallasConfig>(&bases, &scalars).unwrap(),
        expected
    );

    let rounds = (1 << 13) / n + 1;
    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(G::msm_bigint(&bases, &bigints));
        }
        let old = t.elapsed().as_secs_f64() / rounds as f64;
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(eisenstein_msm::<PallasConfig>(&bases, &scalars));
        }
        let new = t.elapsed().as_secs_f64() / rounds as f64;
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    println!(
        "n = {n:4}: buckets {:8.2} us, straus {:8.2} us, {:.2}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        median(ratios)
    );
}

/// `msm_unchecked`'s scalar conversion, serial against `par_iter`. Measures the crossover, so
/// `MIN_PARALLEL_SCALARS` can be set from it.
fn into_bigint_conversion(n: usize) {
    let rng = &mut test_rng();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let rounds = (1 << 16) / n + 1;
    let serial = |_: ()| {
        scalars
            .iter()
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>()
    };
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(serial(()));
        }
        times.push(t.elapsed().as_secs_f64() / rounds as f64);
    }
    let t_serial = median(times);

    #[cfg(feature = "parallel")]
    let t_par = {
        use rayon::prelude::*;
        let mut times = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let t = Instant::now();
            for _ in 0..rounds {
                let _ = core::hint::black_box(
                    scalars
                        .par_iter()
                        .map(|s| s.into_bigint())
                        .collect::<Vec<_>>(),
                );
            }
            times.push(t.elapsed().as_secs_f64() / rounds as f64);
        }
        median(times)
    };
    #[cfg(not(feature = "parallel"))]
    let t_par = f64::NAN;

    println!(
        "into_bigint({n:6}): serial {:8.2} us, par_iter {:8.2} us, {:.2}x",
        t_serial * 1e6,
        t_par * 1e6,
        t_serial / t_par
    );
}
