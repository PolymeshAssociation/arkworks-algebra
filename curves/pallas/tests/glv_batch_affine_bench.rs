//! Full-width MSM over GLV halves against full-width windows.
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test glv_batch_affine_bench -- --ignored --nocapture`
//! and with `--features parallel` for the parallel numbers (`MSM_THREADS=1,4,16` picks the pools).
//! `MSM_SIZES=64,256,4096` picks the sizes.

use ark_ec::{
    scalar_mul::{
        sw_pippenger::{msm_batch_affine_bigint, msm_batch_affine_glv_bigint, BATCH_AFFINE_MIN_POINTS},
        variable_base::msm_bigint_wnaf,
    },
    CurveGroup,
};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

type BigInt = <Fr as PrimeField>::BigInt;

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

/// The full-width path before the split: batch-affine windows from `BATCH_AFFINE_MIN_POINTS`,
/// the projective wNAF below.
fn full_width(bases: &[GAffine], bigints: &[BigInt]) -> G {
    if bases.len() >= BATCH_AFFINE_MIN_POINTS {
        msm_batch_affine_bigint::<PallasConfig>(bases, bigints)
    } else {
        msm_bigint_wnaf(bases, bigints)
    }
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn glv_batch_affine_vs_full_width() {
    let sizes = env_list(
        "MSM_SIZES",
        &[64, 96, 128, 192, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    );
    let threads = if cfg!(feature = "parallel") {
        env_list("MSM_THREADS", &[1, 4, 8, 16])
    } else {
        vec![1]
    };
    let max_n = *sizes.iter().max().unwrap();
    let rng = &mut test_rng();
    let bases: Vec<GAffine> =
        G::normalize_batch(&(0..max_n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    const SETS: usize = 3;
    let bigints: Vec<Vec<BigInt>> = (0..SETS)
        .map(|_| (0..max_n).map(|_| Fr::rand(rng).into_bigint()).collect())
        .collect();
    for t in threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap();
        println!(
            "{} threads {t}",
            if cfg!(feature = "parallel") { "parallel" } else { "serial" }
        );
        for &n in &sizes {
            pool.install(|| compare(&bases[..n], &bigints, n));
        }
    }
}

fn compare(bases: &[GAffine], bigints: &[Vec<BigInt>], n: usize) {
    for k in bigints {
        assert_eq!(
            msm_batch_affine_glv_bigint::<PallasConfig>(bases, &k[..n]).unwrap(),
            full_width(bases, &k[..n])
        );
    }
    let reps = ((1usize << 19) / n).clamp(8, 200);
    let (mut old_t, mut new_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..reps {
        let k = &bigints[r % bigints.len()][..n];
        let time = |glv: bool| {
            let t = Instant::now();
            if glv {
                let _ = core::hint::black_box(msm_batch_affine_glv_bigint::<PallasConfig>(bases, k));
            } else {
                let _ = core::hint::black_box(full_width(bases, k));
            }
            t.elapsed().as_secs_f64() * 1e3
        };
        let (old, new) = if r % 2 == 0 {
            let old = time(false);
            (old, time(true))
        } else {
            let new = time(true);
            (time(false), new)
        };
        old_t.push(old);
        new_t.push(new);
        ratios.push(old / new);
    }
    let min = |v: &[f64]| v.iter().cloned().fold(f64::MAX, f64::min);
    let (old_min, new_min) = (min(&old_t), min(&new_t));
    println!(
        "  n = {n:>6}: full width {:9.3} ms, GLV halves {:9.3} ms (minima), {:.3}x on minima, {:.3}x median ratio",
        old_min,
        new_min,
        old_min / new_min,
        median(ratios),
    );
}
