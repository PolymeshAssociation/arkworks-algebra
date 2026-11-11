//! Canonical against loose field operations on Pallas `Fq`.
//!
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml [--features asm] --test loose_ops_bench -- --ignored --nocapture`

use ark_ff::{AdditiveGroup, Field, Loose, UniformRand};
use ark_pallas::Fq;
use ark_std::test_rng;
use std::{hint::black_box, time::Instant};

const REPS: usize = 21;
const INNER: usize = 200_000;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn pair(label: &str, mut canonical: impl FnMut(), mut loose: impl FnMut()) {
    let (mut c, mut l) = (Vec::new(), Vec::new());
    for r in 0..REPS {
        let mut time = |f: &mut dyn FnMut()| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e9 / INNER as f64
        };
        if r % 2 == 0 {
            c.push(time(&mut canonical));
            l.push(time(&mut loose));
        } else {
            l.push(time(&mut loose));
            c.push(time(&mut canonical));
        }
    }
    let (c, l) = (median(c), median(l));
    println!("{label:<22} canonical {c:6.2} ns  loose {l:6.2} ns  {:.3}x", c / l);
}

#[test]
#[ignore = "timing; run explicitly with --release --nocapture"]
fn loose_ops() {
    println!("asm {}", cfg!(feature = "asm"));
    let rng = &mut test_rng();
    let a = Fq::rand(rng);
    let b = Fq::rand(rng);
    let (la, lb) = (Loose::from(a), Loose::from(b));
    pair(
        "mul latency",
        || {
            let mut x = a;
            for _ in 0..INNER {
                x *= &b;
            }
            black_box(x);
        },
        || {
            let mut x = la;
            for _ in 0..INNER {
                x *= &lb;
            }
            black_box(x.reduce());
        },
    );
    pair(
        "mul throughput",
        || {
            let mut x = [a, b, a + b, a.double()];
            for _ in 0..INNER / 4 {
                for v in &mut x {
                    *v *= &b;
                }
            }
            black_box(x);
        },
        || {
            let mut x = [la, lb, la + lb, la.double()];
            for _ in 0..INNER / 4 {
                for v in &mut x {
                    *v *= &lb;
                }
            }
            black_box(x.map(|v| v.reduce()));
        },
    );
    pair(
        "square latency",
        || {
            let mut x = a;
            for _ in 0..INNER {
                x.square_in_place();
            }
            black_box(x);
        },
        || {
            let mut x = la;
            for _ in 0..INNER {
                x = x.square();
            }
            black_box(x.reduce());
        },
    );
    pair(
        "add latency",
        || {
            let mut x = a;
            for _ in 0..INNER {
                x += &b;
            }
            black_box(x);
        },
        || {
            let mut x = la;
            for _ in 0..INNER {
                x += &lb;
            }
            black_box(x.reduce());
        },
    );
    pair(
        "sub latency",
        || {
            let mut x = a;
            for _ in 0..INNER {
                x -= &b;
            }
            black_box(x);
        },
        || {
            let mut x = la;
            for _ in 0..INNER {
                x -= &lb;
            }
            black_box(x.reduce());
        },
    );
    pair(
        "mul+reduce latency",
        || {
            let mut x = a;
            for _ in 0..INNER {
                x *= &b;
            }
            black_box(x);
        },
        || {
            let mut x = a;
            for _ in 0..INNER {
                x = (Loose::from(x) * &lb).reduce();
            }
            black_box(x);
        },
    );
}
