//! Window (`c`) comparison for `FixedBaseMSM`, validating against the variable-base baseline.
//! For each base count it builds the table at several `c`, times the fixed-base evaluation
//! against `msm_bigint_full_width` (the B3 sorted-bucket path `msm_check` uses), and marks the window
//! `new()` picks.
//!
//! Serial:   cargo test --release --manifest-path curves/pallas/Cargo.toml --test fixed_base_window_bench -- --ignored --nocapture
//! Parallel: Use `--features parallel` (ec/parallel) as well.
//!
//! Env: MSM_SIZES=10,12,13,14,16 (base counts n = 2^size); MSM_C=8,10,..,18 (windows to compare).

use ark_ec::scalar_mul::fixed_base::FixedBaseMSM;
use ark_ec::short_weierstrass::{Projective, SWCurveConfig};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ark_std::test_rng;
use std::time::{Duration, Instant};

/// Minimum time to compute the function
fn min_time<P: SWCurveConfig>(reps: usize, mut f: impl FnMut() -> Projective<P>) -> f64 {
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

fn compare<P: SWCurveConfig>(name: &str, sizes: &[u32], c: &[usize]) {
    let rng = &mut test_rng();
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
    println!("\n== {name}  (threads={threads}; ratio = variable-base / fixed-base, >1 = fixed faster) ==");
    for &size in sizes {
        let n = 1usize << size;
        let bases =
            Projective::<P>::normalize_batch(&(0..n).map(|_| Projective::<P>::rand(rng)).collect::<Vec<_>>());
        let scalars: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();
        let big: Vec<_> = scalars.iter().map(|s| s.into_bigint()).collect();
        let reps = if size >= 16 { 11 } else { 25 };

        let new_c = FixedBaseMSM::<P>::new(&bases).window();
        let fw = min_time::<P>(reps, || Projective::<P>::msm_bigint_full_width(&bases, &big));
        println!("  n=2^{size} ({n} bases)   variable-base {:.3} ms   new()->c={new_c}", fw * 1e3);
        for &c_i in c {
            let t = Instant::now();
            let pc = FixedBaseMSM::<P>::new_given_window_size(&bases, c_i);
            let build = t.elapsed();
            let ev = min_time::<P>(reps, || pc.msm_bigint(&big));
            println!(
                "    c={c_i:<2} W={w:<2} buckets={b:<6} table={mib:6.1} MiB  precompute {build:>8.1?}  eval {ev:7.3} ms  {ratio:.3}x{mark}",
                w = pc.num_windows(),
                b = pc.num_buckets(),
                mib = pc.table_bytes() as f64 / (1024.0 * 1024.0),
                ev = ev * 1e3,
                ratio = fw / ev,
                mark = if c_i == new_c { "  <- new()" } else { "" },
            );
        }
    }
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn fixed_base_window_comparison() {
    let sizes: Vec<u32> = std::env::var("MSM_SIZES")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 12, 13, 14, 16]);
    let cs: Vec<usize> = std::env::var("MSM_C")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![10, 11, 12, 13, 14, 15, 16, 17, 18]);
    compare::<ark_pallas::PallasConfig>("Pallas", &sizes, &cs);
    compare::<ark_vesta::VestaConfig>("Vesta", &sizes, &cs);
}
