//! Timings of every operation the assembly kernels serve. Multiply and square are compared against
//! [`Fp::mul_portable`] and [`Fp::square_portable`] in one binary; addition, subtraction, doubling
//! and the curve operations are compared by running this test from a build with the `asm` feature
//! and one without.
//!
//! Throughput rows run independent operations over a 1024-entry table with a `black_box` per
//! result; latency rows run one dependent chain. Kernel order alternates per repetition and each
//! figure is the median over repetitions.
//!
//! Run with `--release`, whose profile has no debug assertions, since the portable kernels would
//! pay overflow checks the assembly cannot:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --features asm --test asm_backend_bench -- --ignored --nocapture`

use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{AdditiveGroup, Field, UniformRand};
use ark_pallas::{Affine, Fq, Fr, Projective};
use ark_std::test_rng;
use std::{hint::black_box, time::Instant};

const SAMPLES: usize = 1024;
const INNER: usize = 50_000;
const REPS: usize = 21;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn asm_vs_portable() {
    println!(
        "target_arch {} | asm feature {} | pasta kernels {} | x86-64 MULX kernel {} | debug_assertions {}",
        std::env::consts::ARCH,
        cfg!(feature = "asm"),
        ark_ff::pasta_asm_compiled(),
        ark_ff::mulx_kernel_active(),
        cfg!(debug_assertions),
    );
    field_ops();
    curve_ops();
}

fn field_ops() {
    let mut rng = test_rng();
    let xs: Vec<Fq> = (0..SAMPLES).map(|_| Fq::rand(&mut rng)).collect();
    let ys: Vec<Fq> = (0..SAMPLES).map(|_| Fq::rand(&mut rng)).collect();

    macro_rules! throughput {
        ($op:expr) => {{
            let t = Instant::now();
            for i in 0..INNER {
                let j = i % SAMPLES;
                black_box($op(xs[j], ys[j]));
            }
            t.elapsed().as_nanos() as f64 / INNER as f64
        }};
    }
    macro_rules! latency {
        ($op:expr) => {{
            let mut acc = xs[0];
            let t = Instant::now();
            for i in 0..INNER {
                acc = $op(acc, ys[i % SAMPLES]);
            }
            black_box(acc);
            t.elapsed().as_nanos() as f64 / INNER as f64
        }};
    }

    type Op = fn(Fq, Fq) -> Fq;
    // Multiply and square have a portable oracle in this binary; the others are compared across
    // builds.
    let ops: [(&str, Op, Option<Op>); 5] = [
        ("mul", |x, y| x * y, Some(|x, y| x.mul_portable(&y))),
        (
            "square",
            |x, _| x.square(),
            Some(|x, _| x.square_portable()),
        ),
        ("add", |x, y| x + y, None),
        ("sub", |x, y| x - y, None),
        ("double", |x, _| x.double(), None),
    ];
    println!("\nFq, ns/op        dispatched   portable   ratio");
    for (name, dispatched, portable) in ops {
        let (mut td, mut tp, mut ld, mut lp) = (vec![], vec![], vec![], vec![]);
        for rep in 0..REPS {
            if rep % 2 == 1 {
                if let Some(portable) = portable {
                    tp.push(throughput!(portable));
                    lp.push(latency!(portable));
                }
            }
            td.push(throughput!(dispatched));
            ld.push(latency!(dispatched));
            if rep % 2 == 0 {
                if let Some(portable) = portable {
                    tp.push(throughput!(portable));
                    lp.push(latency!(portable));
                }
            }
        }
        for (kind, d, p) in [("thr", td, tp), ("lat", ld, lp)] {
            let d = median(d);
            if p.is_empty() {
                println!("{name:>7} {kind}    {d:>9.2}");
            } else {
                let p = median(p);
                println!("{name:>7} {kind}    {d:>9.2}  {p:>9.2}  {:>6.3}x", p / d);
            }
        }
    }
    for (x, y) in xs.iter().zip(&ys) {
        assert_eq!(*x * y, x.mul_portable(y));
        assert_eq!(x.square(), x.square_portable());
    }
}

fn curve_ops() {
    let mut rng = test_rng();
    let ps: Vec<Projective> = (0..SAMPLES).map(|_| Projective::rand(&mut rng)).collect();
    let qs: Vec<Affine> = (0..SAMPLES).map(|_| Affine::rand(&mut rng)).collect();
    let ks: Vec<Fr> = (0..SAMPLES).map(|_| Fr::rand(&mut rng)).collect();

    macro_rules! time {
        ($n:expr, $body:expr) => {{
            let mut v = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t = Instant::now();
                for i in 0..$n {
                    let _ = black_box($body(i % SAMPLES));
                }
                v.push(t.elapsed().as_nanos() as f64 / $n as f64);
            }
            median(v)
        }};
    }
    println!("\nPallas, ns/op");
    println!("  mixed add   {:>12.1}", time!(INNER, |i| ps[i] + qs[i]));
    println!(
        "  add         {:>12.1}",
        time!(INNER, |i| ps[i] + ps[(i + 1) % SAMPLES])
    );
    println!(
        "  double      {:>12.1}",
        time!(INNER, |i: usize| ps[i].double())
    );
    println!("  scalar mul  {:>12.1}", time!(2_000, |i| qs[i] * ks[i]));
    for log_n in [8u32, 12, 16] {
        let n = 1usize << log_n;
        let bases: Vec<Affine> = Projective::normalize_batch(
            &(0..n)
                .map(|_| Projective::rand(&mut rng))
                .collect::<Vec<_>>(),
        );
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let reps = (1usize << 20 >> log_n).clamp(3, 200);
        let v: Vec<f64> = (0..REPS.min(7))
            .map(|_| {
                let t = Instant::now();
                for _ in 0..reps {
                    let _ = black_box(Projective::msm(&bases, &scalars).unwrap());
                }
                t.elapsed().as_nanos() as f64 / reps as f64
            })
            .collect();
        println!("  msm 2^{log_n:<3}   {:>12.1}", median(v));
    }
}
