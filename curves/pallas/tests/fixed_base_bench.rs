//! Wall-clock and table-size comparison: BGMW fixed-base MSM vs the variable-base paths, at the
//! Bulletproofs verifier's fixed-generator sizes (`2n + 2` for `n` in the DART range).
//!
//! Gate 1 of FIXED_BASE_MSM_PLAN.md. `fixed/full_width` compares against
//! `VariableBaseMSM::msm_bigint_full_width` — the B3 sorted-bucket batch-affine path that
//! `msm_check` calls today. `fixed/unchecked` compares against `msm_unchecked` (the full
//! `msm_signed` routing). Ratios > 1 mean fixed-base is faster.
//!
//! Serial:  cargo test --release --manifest-path curves/pallas/Cargo.toml --test fixed_base_bench -- --ignored --nocapture
//! Parallel: add `--features parallel` (build ec with parallel) to the arkworks-algebra workspace.
//!
//! Env: MSM_LOGS=13,14,15 (sizes n = 2^log + 2); MSM_C=13,14,15,16,17 (explicit windows to sweep).

use ark_ec::{scalar_mul::fixed_base::FixedBaseMSM, CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::test_rng;
use std::time::{Duration, Instant};

fn min_time(reps: usize, mut f: impl FnMut() -> G) -> f64 {
    let _ = f();
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed());
        let _ = std::hint::black_box(r);
    }
    best.as_secs_f64()
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn fixed_base_msm_vs_variable() {
    let rng = &mut test_rng();
    let logs: Vec<u32> = std::env::var("MSM_LOGS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![13, 14, 15]);
    let cs: Vec<usize> = std::env::var("MSM_C")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();

    let threads = {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads()
        }
        #[cfg(not(feature = "parallel"))]
        {
            1usize
        }
    };
    println!(
        "size_of::<Affine> = {} bytes; rayon threads = {threads}; ratios > 1 mean fixed-base faster",
        core::mem::size_of::<GAffine>()
    );

    for log in logs {
        // The fixed part of the verifier MSM is 2*padded_n + 2.
        let n = (1usize << log) + 2;
        let bases =
            G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
        let scalars = (0..n).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
        let bigints = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
        let reps = if log >= 15 { 9 } else { 15 };

        let windows = if cs.is_empty() {
            vec![None]
        } else {
            cs.iter().map(|&c| Some(c)).collect::<Vec<_>>()
        };

        for wopt in windows {
            let t = Instant::now();
            let pc = match wopt {
                Some(c) => FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, c),
                None => FixedBaseMSM::<PallasConfig>::new(&bases),
            };
            let build = t.elapsed();

            assert_eq!(
                pc.msm(&scalars),
                G::msm_bigint_full_width(&bases, &bigints),
                "BGMW result mismatch at n=2^{log}+2, c={}",
                pc.window()
            );

            let mut r_fw: Vec<f64> = Vec::with_capacity(reps);
            let mut r_un: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let fw = {
                    let t = Instant::now();
                    let r = G::msm_bigint_full_width(&bases, &bigints);
                    let _ = std::hint::black_box(r);
                    t.elapsed().as_secs_f64()
                };
                let un = {
                    let t = Instant::now();
                    let r = G::msm_unchecked(&bases, &scalars);
                    let _ = std::hint::black_box(r);
                    t.elapsed().as_secs_f64()
                };
                let fb = min_time(1, || pc.msm(&scalars));
                r_fw.push(fw / fb);
                r_un.push(un / fb);
            }
            // Compare on the best fixed-base time: recompute each path's min separately.
            let fb_min = min_time(reps, || pc.msm(&scalars));
            let fw_min = min_time(reps, || G::msm_bigint_full_width(&bases, &bigints));
            let un_min = min_time(reps, || G::msm_unchecked(&bases, &scalars));

            println!(
                "n=2^{log}+2  c={c:<2} W={w:<2} buckets={buck:<6} table={len:>9} pts ({mib:6.1} MiB)  build {build:>8.2?} | fixed {fb:8.3}ms  full_width {fw:8.3}ms ({rfw:.3}x)  unchecked {un:8.3}ms ({run:.3}x)",
                c = pc.window(),
                w = pc.num_windows(),
                buck = pc.num_buckets(),
                len = pc.table_len(),
                mib = pc.table_bytes() as f64 / (1024.0 * 1024.0),
                fb = fb_min * 1e3,
                fw = fw_min * 1e3,
                un = un_min * 1e3,
                rfw = fw_min / fb_min,
                run = un_min / fb_min,
            );
            let _ = (&r_fw, &r_un);
        }
    }
}
