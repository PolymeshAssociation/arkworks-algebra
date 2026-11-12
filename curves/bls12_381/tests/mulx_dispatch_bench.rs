//! Cost of the run-time CPUID check that selects the x86-64 MULX/ADX Montgomery kernel. Compare a
//! build that reaches the kernel through the check with one that folds it away, the repository's
//! `+bmi2,+adx`, in separate target directories:
//! `RUSTFLAGS="" cargo test --release --target-dir target-cpuid --manifest-path curves/bls12_381/Cargo.toml --features asm --test mulx_dispatch_bench -- --ignored --nocapture`
//! and the same command without `RUSTFLAGS=""` and with `--target-dir target-folded`. The rows
//! are the same as `curves/bn254/tests/mulx_dispatch_bench.rs`.

use ark_bls12_381::{Bls12_381 as Curve, Fq, Fr, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup, VariableBaseMSM};
use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_std::test_rng;
use std::{hint::black_box, time::Instant};

const SAMPLES: usize = 1024;
const INNER: usize = 50_000;
const REPS: usize = 15;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Median ns per call over `REPS` repetitions of `n` calls of `f`.
fn time(n: usize, mut f: impl FnMut(usize)) -> f64 {
    median(
        (0..REPS)
            .map(|_| {
                let t = Instant::now();
                for i in 0..n {
                    f(i);
                }
                t.elapsed().as_nanos() as f64 / n as f64
            })
            .collect(),
    )
}

/// Throughput over independent operations and latency over one dependent chain.
fn field_rows<F: Field + UniformRand>(name: &str) {
    let mut rng = test_rng();
    let xs: Vec<F> = (0..SAMPLES).map(|_| F::rand(&mut rng)).collect();
    let ys: Vec<F> = (0..SAMPLES).map(|_| F::rand(&mut rng)).collect();
    let ops: [(&str, fn(F, F) -> F); 2] = [("mul", |x, y| x * y), ("square", |x, _| x.square())];
    for (op_name, op) in ops {
        let thr = time(INNER, |i| {
            black_box(op(xs[i % SAMPLES], ys[i % SAMPLES]));
        });
        let mut acc = xs[0];
        let lat = time(INNER, |i| acc = op(acc, ys[i % SAMPLES]));
        black_box(acc);
        println!("  {name} {op_name:<7} thr {thr:>9.2} ns   lat {lat:>9.2} ns");
    }
}

fn curve_rows() {
    let mut rng = test_rng();
    let ps: Vec<G1Projective> = (0..SAMPLES).map(|_| G1Projective::rand(&mut rng)).collect();
    let add = time(INNER, |i| {
        let _ = black_box(ps[i % SAMPLES] + ps[(i + 1) % SAMPLES]);
    });
    println!("  G1 add          {add:>12.1} ns");
    let double = time(INNER, |i| {
        let _ = black_box(ps[i % SAMPLES].double());
    });
    println!("  G1 double       {double:>12.1} ns");

    let bases = G1Projective::normalize_batch(&ps[..]);
    let bases: Vec<_> = bases.iter().cycle().take(1 << 12).copied().collect();
    let scalars: Vec<Fr> = (0..bases.len()).map(|_| Fr::rand(&mut rng)).collect();
    let msm = time(10, |_| {
        let _ = black_box(G1Projective::msm(&bases, &scalars).unwrap());
    });
    println!("  G1 msm 2^12     {msm:>12.1} ns");

    // Rotating inputs, so the branch predictor cannot memorize one pairing.
    let g1 = G1Projective::normalize_batch(&ps[..16]);
    let g2: Vec<_> = (0..16)
        .map(|_| G2Projective::rand(&mut rng).into_affine())
        .collect();
    let pairing = time(200, |i| {
        let _ = black_box(Curve::pairing(g1[i % 16], g2[i % 16]));
    });
    println!("  pairing         {pairing:>12.1} ns");
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn mulx_dispatch() {
    println!(
        "target_arch {} | asm feature {} | MULX kernel {} | BMI2 and ADX at compile time {}",
        std::env::consts::ARCH,
        cfg!(feature = "asm"),
        ark_ff::mulx_kernel_active(),
        cfg!(all(target_feature = "bmi2", target_feature = "adx")),
    );
    field_rows::<Fq>("Fq");
    field_rows::<Fr>("Fr");
    curve_rows();
}
