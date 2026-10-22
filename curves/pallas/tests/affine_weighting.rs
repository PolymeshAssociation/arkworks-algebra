//! Affine bucket weighting against the `xyzz` running sum.
//!
//! Benches run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test affine_weighting -- --ignored --nocapture`
//! and with `--features parallel` for the parallel numbers (`MSM_THREADS=1,4,16` picks the pools).
//! `BUCKETS=...` and `MSM_SIZES=...` pick the sizes.

use ark_ec::{
    scalar_mul::fixed_base::FixedBaseMSM,
    scalar_mul::sw_pippenger::{
        bucket_sum_for_bench, msm_batch_affine_bigint, msm_batch_affine_glv_bigint,
        AFFINE_WEIGHTING,
    },
    AffineRepr, CurveGroup,
};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{rand::Rng, test_rng, vec::Vec};
use std::{sync::atomic::Ordering, time::Instant};

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

fn naive(buckets: &[GAffine]) -> G {
    buckets
        .iter()
        .enumerate()
        .map(|(i, b)| *b * Fr::from(i as u64 + 1))
        .sum()
}

#[test]
fn affine_bucket_sum_matches_running_sum() {
    let rng = &mut test_rng();
    let pool: Vec<GAffine> = G::normalize_batch(&(0..8).map(|_| G::rand(rng)).collect::<Vec<_>>());
    for len in (1..=300).chain([511, 512, 513, 1024, 2048, 4097]) {
        for density in [0u32, 1, 2, 3] {
            let buckets: Vec<GAffine> = (0..len)
                .map(|_| match (density, rng.gen_range(0..8u32)) {
                    // Random distinct points with some empty buckets.
                    (0, 0) => GAffine::zero(),
                    (0, _) => G::rand(rng).into_affine(),
                    // Mostly empty.
                    (1, 0) => G::rand(rng).into_affine(),
                    (1, _) => GAffine::zero(),
                    // Few distinct points, so pairs and tree levels hit doublings.
                    (2, k) => pool[k as usize % 3],
                    // A point, its negation and the identity, so sums cancel.
                    (_, k) => match k % 3 {
                        0 => pool[0],
                        1 => -pool[0],
                        _ => GAffine::zero(),
                    },
                })
                .collect();
            let expected = bucket_sum_for_bench::<PallasConfig>(&buckets, false);
            assert_eq!(
                bucket_sum_for_bench::<PallasConfig>(&buckets, true),
                expected,
                "len {len} density {density}"
            );
            if len <= 64 {
                assert_eq!(expected, naive(&buckets));
            }
        }
    }
}

#[test]
fn msm_with_affine_weighting_matches() {
    let rng = &mut test_rng();
    for n in [256usize, 1000, 4096] {
        let bases: Vec<GAffine> =
            G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let bigints: Vec<BigInt> = scalars.iter().map(|s| s.into_bigint()).collect();
        let expected: G = bases.iter().zip(&scalars).map(|(b, s)| *b * s).sum();
        assert_eq!(
            msm_batch_affine_bigint::<PallasConfig>(&bases, &bigints),
            expected
        );
        assert_eq!(
            msm_batch_affine_glv_bigint::<PallasConfig>(&bases, &bigints).unwrap(),
            expected
        );
    }
}

#[test]
fn fixed_base_with_affine_weighting_matches() {
    let rng = &mut test_rng();
    for (n, cs) in [(37usize, &[4usize, 8][..]), (600, &[6, 10, 13][..]), (4100, &[8, 12, 16][..])] {
        let mut bases: Vec<GAffine> =
            G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
        // Repeated and opposite bases, so bucket points collide and cancel.
        bases[1] = bases[0];
        bases[2] = -bases[0];
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let bigints: Vec<BigInt> = scalars.iter().map(|s| s.into_bigint()).collect();
        let expected: G = bases.iter().zip(&scalars).map(|(b, s)| *b * s).sum();
        for &c in cs {
            let table = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, c);
            for affine in [true, false] {
                AFFINE_WEIGHTING.store(affine, Ordering::Relaxed);
                assert_eq!(table.msm_bigint(&bigints), expected, "n {n} c {c} affine {affine}");
            }
        }
    }
    AFFINE_WEIGHTING.store(true, Ordering::Relaxed);
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn bucket_sum_bench() {
    let sizes = env_list(
        "BUCKETS",
        &[64, 128, 256, 512, 1024, 2048, 4096, 8192, 32768],
    );
    let rng = &mut test_rng();
    const SETS: usize = 4;
    for &len in &sizes {
        let sets: Vec<Vec<GAffine>> = (0..SETS)
            .map(|_| G::normalize_batch(&(0..len).map(|_| G::rand(rng)).collect::<Vec<_>>()))
            .collect();
        let reps = ((1usize << 18) / len).clamp(20, 400);
        let (mut old_t, mut new_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..reps {
            let b = &sets[r % SETS];
            let time = |affine: bool| {
                let t = Instant::now();
                let _ = core::hint::black_box(bucket_sum_for_bench::<PallasConfig>(b, affine));
                t.elapsed().as_secs_f64() * 1e6
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
        println!(
            "  buckets = {len:>6}: running {:9.2} us, affine {:9.2} us (medians), {:.3}x median ratio",
            median(old_t),
            median(new_t),
            median(ratios),
        );
    }
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn msm_bench() {
    let sizes = env_list(
        "MSM_SIZES",
        &[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    );
    let threads = if cfg!(feature = "parallel") {
        env_list("MSM_THREADS", &[1, 4, 16])
    } else {
        vec![1]
    };
    let max_n = *sizes.iter().max().unwrap();
    let rng = &mut test_rng();
    let bases: Vec<GAffine> =
        G::normalize_batch(&(0..max_n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    const SETS: usize = 4;
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
            if cfg!(feature = "parallel") {
                "parallel"
            } else {
                "serial"
            }
        );
        for &n in &sizes {
            pool.install(|| {
                compare("full width", n, &bigints, |k| {
                    msm_batch_affine_bigint::<PallasConfig>(&bases[..n], k)
                });
                compare("GLV", n, &bigints, |k| {
                    msm_batch_affine_glv_bigint::<PallasConfig>(&bases[..n], k).unwrap()
                });
            });
        }
    }
    AFFINE_WEIGHTING.store(true, Ordering::Relaxed);
}

fn compare(label: &str, n: usize, bigints: &[Vec<BigInt>], f: impl Fn(&[BigInt]) -> G) {
    let reps = ((1usize << 19) / n).clamp(10, 200);
    let (mut old_t, mut new_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..reps {
        let k = &bigints[r % bigints.len()][..n];
        let time = |affine: bool| {
            AFFINE_WEIGHTING.store(affine, Ordering::Relaxed);
            let t = Instant::now();
            let _ = core::hint::black_box(f(k));
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
    println!(
        "  {label:>10} n = {n:>6}: running {:9.3} ms, affine {:9.3} ms (medians), {:.3}x median ratio",
        median(old_t),
        median(new_t),
        median(ratios),
    );
}

/// BGMW evaluation, running sum against affine weighting, at every window in `MSM_C`. Sizes are
/// `2^log + 2`, the Bulletproofs verifier's generator counts.
#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn fixed_base_bench() {
    let logs = env_list("MSM_LOGS", &[10, 12, 13, 14, 16]);
    let cs = env_list("MSM_C", &[10, 11, 12, 13, 14, 15, 16, 17, 18]);
    let threads = if cfg!(feature = "parallel") {
        env_list("MSM_THREADS", &[16])
    } else {
        vec![1]
    };
    let rng = &mut test_rng();
    const SETS: usize = 4;
    for t in threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        println!(
            "{} threads {t}",
            if cfg!(feature = "parallel") { "parallel" } else { "serial" }
        );
        pool.install(|| {
            for &log in &logs {
                let n = (1usize << log) + 2;
                let bases: Vec<GAffine> =
                    G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
                let bigints: Vec<Vec<BigInt>> = (0..SETS)
                    .map(|_| (0..n).map(|_| Fr::rand(rng).into_bigint()).collect())
                    .collect();
                let picked = model_window(n);
                println!("  n = 2^{log} + 2: best_window {picked}");
                for &c in &cs {
                    let table = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, c);
                    let reps = ((1usize << 21) / n).clamp(9, 41);
                    let (mut old_t, mut new_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
                    for r in 0..reps {
                        let k = &bigints[r % SETS];
                        let time = |affine: bool| {
                            AFFINE_WEIGHTING.store(affine, Ordering::Relaxed);
                            let t = Instant::now();
                            let _ = core::hint::black_box(table.msm_bigint(k));
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
                    println!(
                        "    c = {c:>2} buckets = {:>6}: running {:8.3} ms, affine {:8.3} ms (medians), {:.3}x median ratio",
                        table.num_buckets(),
                        median(old_t),
                        median(new_t),
                        median(ratios),
                    );
                }
            }
        });
    }
    AFFINE_WEIGHTING.store(true, Ordering::Relaxed);
}

/// `fixed_base::best_window` for Pallas's 255-bit scalars.
fn model_window(n: usize) -> usize {
    (2..=20usize)
        .min_by_key(|&c| n * 255usize.div_ceil(c) + (1usize << c))
        .unwrap()
}
