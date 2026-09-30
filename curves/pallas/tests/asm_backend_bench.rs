//! Timings of every operation the assembly kernels serve. Multiply and square are compared against
//! [`Fp::mul_portable`] and [`Fp::square_portable`] in one binary; addition, subtraction, doubling
//! and the curve operations are compared by running this test from a build with the `asm` feature
//! and one without.
//!
//! Throughput rows run independent operations over a 1024-entry table with a `black_box` per
//! result; latency rows run one dependent chain. Kernel order alternates per repetition and each
//! figure is the median over repetitions.
//!
//! Run with the test profile's debug assertions off, since the portable kernels would pay overflow
//! checks the assembly cannot:
//! `CARGO_PROFILE_TEST_DEBUG_ASSERTIONS=false cargo test --release --manifest-path curves/pallas/Cargo.toml --features asm --test asm_backend_bench -- --ignored --nocapture`

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
        "target_arch {} | asm feature {} | pasta kernels {} | x86-64 generic kernel {} | debug_assertions {}",
        std::env::consts::ARCH,
        cfg!(feature = "asm"),
        ark_ff::pasta_asm_available(),
        ark_ff::asm_backend_active(),
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
    let ops: [(&str, Op, Op); 5] = [
        ("mul", |x, y| x * y, |x, y| x.mul_portable(&y)),
        ("square", |x, _| x.square(), |x, _| x.square_portable()),
        ("add", |x, y| x + y, |x, y| x + y),
        ("sub", |x, y| x - y, |x, y| x - y),
        ("double", |x, _| x.double(), |x, _| x.double()),
    ];
    println!("\nFq, ns/op        dispatched   portable   ratio");
    for (name, dispatched, portable) in ops {
        let (mut td, mut tp, mut ld, mut lp) = (vec![], vec![], vec![], vec![]);
        for rep in 0..REPS {
            if rep % 2 == 0 {
                td.push(throughput!(dispatched));
                tp.push(throughput!(portable));
                ld.push(latency!(dispatched));
                lp.push(latency!(portable));
            } else {
                tp.push(throughput!(portable));
                td.push(throughput!(dispatched));
                lp.push(latency!(portable));
                ld.push(latency!(dispatched));
            }
        }
        for (kind, d, p) in [
            ("thr", median(td), median(tp)),
            ("lat", median(ld), median(lp)),
        ] {
            println!("{name:>7} {kind}    {d:>9.2}  {p:>9.2}  {:>6.3}x", p / d);
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
