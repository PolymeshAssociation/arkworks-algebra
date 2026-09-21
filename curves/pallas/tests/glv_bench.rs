//! The Eisenstein joint recoding against the joint sparse form it replaced.
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test glv_bench -- --ignored --nocapture`

use ark_ec::{
    scalar_mul::glv::{
        eisenstein::{glv_mul_same_scalar, Decomposed, Table},
        jsf_mul_affine_projective, jsf_mul_projective, GLVConfig,
    },
    AffineRepr, CurveGroup,
};
use ark_ff::{AdditiveGroup, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

const REPS: usize = 11;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn eisenstein_vs_jsf() {
    single_projective();
    single_affine();
    for n in [8usize, 32, 256] {
        same_scalar(n);
    }
    for n in [3usize, 4, 8, 16, 32, 64, 256] {
        table_build(n);
    }
    single_narrow();
    decompose("random 255-bit  ", &(0..512).map(|_| Fr::rand(&mut test_rng())).collect::<Vec<_>>());
    decompose("64-bit          ", &(1..=512u64).map(Fr::from).collect::<Vec<_>>());
    decompose("powers_of_base  ", &powers_of_base());
    powers_of_base_msm();
    digit_density();
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

/// Runs `old` and `new` alternately and reports the median ratio.
fn compare(label: &str, per: usize, mut old: impl FnMut(), mut new: impl FnMut()) {
    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        let t = Instant::now();
        old();
        let o = t.elapsed().as_secs_f64() / per as f64;
        let t = Instant::now();
        new();
        let n = t.elapsed().as_secs_f64() / per as f64;
        ratios.push(o / n);
        t_old += o;
        t_new += n;
    }
    println!(
        "{label}: old {:7.2} us, new {:7.2} us, {:.2}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        median(ratios)
    );
}

fn single_projective() {
    let rng = &mut test_rng();
    let n = 200;
    let points: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    assert!(points
        .iter()
        .zip(&scalars)
        .all(|(p, k)| PallasConfig::glv_mul_projective(*p, *k) == jsf_mul_projective::<PallasConfig>(*p, *k)));
    compare(
        "single mul, projective base ",
        n,
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(jsf_mul_projective::<PallasConfig>(*p, *k));
            }
        },
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(PallasConfig::glv_mul_projective(*p, *k));
            }
        },
    );
}

fn single_affine() {
    let rng = &mut test_rng();
    let n = 200;
    let points: Vec<GAffine> =
        G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    compare(
        "single mul, affine base     ",
        n,
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(jsf_mul_affine_projective::<PallasConfig>(*p, *k));
            }
        },
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(PallasConfig::glv_mul_affine_projective(*p, *k));
            }
        },
    );
}

/// One scalar against many bases: the batch amortizes the table inversion over `8n` points.
fn same_scalar(n: usize) {
    let rng = &mut test_rng();
    let points: Vec<GAffine> =
        G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let k = Fr::rand(rng);
    let rounds = 4096 / n + 1;
    compare(
        &format!("same scalar, n = {n:<4}      "),
        n * rounds,
        || {
            for _ in 0..rounds {
                for p in &points {
                    let _ = core::hint::black_box(jsf_mul_affine_projective::<PallasConfig>(*p, k));
                }
            }
        },
        || {
            for _ in 0..rounds {
                let _ = core::hint::black_box(glv_mul_same_scalar::<PallasConfig>(&points, k));
            }
        },
    );
}

/// [`Table::batch`]'s affine addition chain against the projective build it replaces. Five
/// batched inversions over `n` denominators against seven projective additions per point and a
/// shared normalization over `8n` entries.
fn table_build(n: usize) {
    let rng = &mut test_rng();
    let points: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let rounds = 4096 / n + 1;
    let affine = Table::<PallasConfig>::batch(&points);
    let projective = Table::<PallasConfig>::batch_projective(&points);
    let k = Decomposed::<PallasConfig>::new(Fr::rand(rng)).expect("half-width bound holds");
    assert!(affine
        .iter()
        .zip(&projective)
        .all(|(a, b)| a.mul_decomposed(&k) == b.mul_decomposed(&k)));
    compare(
        &format!("table build, n = {n:<4}     "),
        n * rounds,
        || {
            for _ in 0..rounds {
                let _ = core::hint::black_box(Table::<PallasConfig>::batch_projective(&points));
            }
        },
        || {
            for _ in 0..rounds {
                let _ = core::hint::black_box(Table::<PallasConfig>::batch(&points));
            }
        },
    );
}

/// One scalar multiplication on a 64-bit scalar, where `Decomposed::new`'s narrow path skips the
/// lattice reduction. The old arm is the body these entry points had: decompose, then recode.
fn single_narrow() {
    let rng = &mut test_rng();
    let n = 200;
    let points: Vec<GAffine> =
        G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let scalars: Vec<Fr> = (1..=n as u64).map(Fr::from).collect();
    let old = |p: &GAffine, k: &Fr| {
        let ((sa, a), (sb, b)) = PallasConfig::scalar_decomposition(*k);
        Decomposed::<PallasConfig>::new_given_halves(sa, a, sb, b)
            .map(|d| Table::<PallasConfig>::new(&p.into_group()).mul_decomposed(&d))
    };
    for (p, k) in points.iter().zip(&scalars) {
        assert_eq!(
            old(p, k).unwrap(),
            PallasConfig::glv_mul_affine_projective(*p, *k)
        );
    }
    compare(
        "single mul, 64-bit scalar   ",
        n,
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(old(p, k));
            }
        },
        || {
            for (p, k) in points.iter().zip(&scalars) {
                let _ = core::hint::black_box(PallasConfig::glv_mul_affine_projective(*p, *k));
            }
        },
    );
}

/// The `powers_of_base` shape dart-bp commits against: `{2^(48i)}` for `i` in `0..6`. The first
/// three fit one GLV half and take the narrow path; the rest decompose.
fn powers_of_base() -> Vec<Fr> {
    (0..6)
        .map(|i| {
            let mut k = Fr::from(1u64);
            for _ in 0..48 * i {
                k.double_in_place();
            }
            k
        })
        .collect()
}

/// [`Decomposed::new`]'s narrow-scalar path against the lattice reduction it skips. The old arm
/// is the body `new` had: decompose, then recode.
fn decompose(label: &str, scalars: &[Fr]) {
    let n = scalars.len();
    let rounds = 4096 / n + 1;
    for k in scalars {
        let ((sa, a), (sb, b)) = PallasConfig::scalar_decomposition(*k);
        assert_eq!(
            Decomposed::<PallasConfig>::new(*k).map(|d| d.len()).is_some(),
            Decomposed::<PallasConfig>::new_given_halves(sa, a, sb, b).is_some(),
        );
    }
    compare(
        &format!("decompose, {label}"),
        n * rounds,
        || {
            for _ in 0..rounds {
                for k in scalars {
                    let ((sa, a), (sb, b)) = PallasConfig::scalar_decomposition(*k);
                    let _ = core::hint::black_box(Decomposed::<PallasConfig>::new_given_halves(
                        sa, a, sb, b,
                    ));
                }
            }
        },
        || {
            for _ in 0..rounds {
                for k in scalars {
                    let _ = core::hint::black_box(Decomposed::<PallasConfig>::new(*k));
                }
            }
        },
    );
}

/// The whole small multi scalar multiplication on the `powers_of_base` shape, against the bucket
/// algorithm it is routed away from.
fn powers_of_base_msm() {
    use ark_ec::{scalar_mul::glv::eisenstein::eisenstein_msm, VariableBaseMSM};
    let rng = &mut test_rng();
    let scalars = powers_of_base();
    let n = scalars.len();
    let bases: Vec<GAffine> = G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
    let expected = G::msm_unchecked(&bases, &scalars);
    assert_eq!(eisenstein_msm::<PallasConfig>(&bases, &scalars), Some(expected));
    let rounds = 256;
    compare(
        "powers_of_base msm, n = 6   ",
        n * rounds,
        || {
            for _ in 0..rounds {
                let _ = core::hint::black_box(G::msm_unchecked(&bases, &scalars));
            }
        },
        || {
            for _ in 0..rounds {
                let _ = core::hint::black_box(eisenstein_msm::<PallasConfig>(&bases, &scalars));
            }
        },
    );
}

/// Absolute per-call minima for the same-scalar batch path, split into its two phases: the
/// table build ([`Table::batch`], the target of the effective-coordinate rewrite) and the
/// shared batch-affine ladder ([`Table::mul_decomposed_batch`], the target of the reusable
/// inversion scratch). Minima over repetitions, never mean or median, and a warmup, because
/// this laptop's per-rep noise on these sizes swamps a mean.
#[test]
#[ignore = "absolute timing; run explicitly with --release --nocapture"]
fn same_scalar_phases() {
    println!("same-scalar phases (us per call, min of {ABS_REPS} reps):");
    println!("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}", "n", "build", "ladder", "end2end", "b+l");
    for n in [32usize, 128, 512, 2048] {
        phases(n);
    }
}

const ABS_REPS: usize = 15;

/// Minimum, over [`ABS_REPS`] repetitions after a warmup, of the wall time of one `f()` call in
/// microseconds. `f` runs `inner` sub-iterations so the timer sees a interval well above its
/// resolution; the reported figure is per sub-iteration.
fn bench_min(inner: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..ABS_REPS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() / inner as f64);
    }
    best * 1e6
}

fn phases(n: usize) {
    let rng = &mut test_rng();
    let proj: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let affine: Vec<GAffine> = G::normalize_batch(&proj);
    let k = Fr::rand(rng);
    let d = Decomposed::<PallasConfig>::new(k).expect("half-width bound holds");
    let tables = Table::<PallasConfig>::batch(&proj);

    let rounds = (8192 / n).max(4);
    let build = bench_min(rounds, || {
        for _ in 0..rounds {
            let _ = core::hint::black_box(Table::<PallasConfig>::batch(&proj));
        }
    });
    let ladder = bench_min(rounds, || {
        for _ in 0..rounds {
            let _ = core::hint::black_box(Table::mul_decomposed_batch(&tables, &d));
        }
    });
    let e2e = bench_min(rounds, || {
        for _ in 0..rounds {
            let _ = core::hint::black_box(glv_mul_same_scalar::<PallasConfig>(&affine, k));
        }
    });
    println!("{n:>6}  {build:>10.2}  {ladder:>10.2}  {e2e:>10.2}  {:>10.2}", build + ladder);
}

/// The digit counts the cost model rests on.
fn digit_density() {
    let rng = &mut test_rng();
    let (mut columns, mut nonzero, mut max_columns, mut max_nonzero) = (0usize, 0usize, 0usize, 0usize);
    let samples = 20_000;
    for _ in 0..samples {
        let d = Decomposed::<PallasConfig>::new(Fr::rand(rng)).expect("half-width bound holds");
        let len = d.len();
        let nz = (0..len).filter(|&i| d.digit(i) != 0).count();
        columns += len;
        nonzero += nz;
        max_columns = max_columns.max(len);
        max_nonzero = max_nonzero.max(nz);
    }
    println!(
        "digits over {samples} scalars: {:.1} columns (max {max_columns}), {:.1} nonzero (max {max_nonzero})",
        columns as f64 / samples as f64,
        nonzero as f64 / samples as f64,
    );
    let table_bytes = core::mem::size_of::<Table<PallasConfig>>();
    println!("Table = {table_bytes} bytes, Decomposed = {} bytes", core::mem::size_of::<Decomposed<PallasConfig>>());
}
