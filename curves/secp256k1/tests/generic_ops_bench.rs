use ark_ec::{AdditiveGroup, CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_secp256k1::{Affine, Projective, Fr};
use ark_std::test_rng;
use std::{hint::black_box, time::Instant};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn row(label: &str, per: usize, mut f: impl FnMut()) {
    f();
    let mut t = Vec::new();
    for _ in 0..15 {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64() * 1e9 / per as f64);
    }
    println!("ROW {label} {:.2}", median(t));
}

#[test]
#[ignore]
fn generic_ops() {
    let rng = &mut test_rng();
    let proj: Vec<Projective> = (0..1024).map(|_| Projective::rand(rng)).collect();
    let aff: Vec<Affine> = Projective::normalize_batch(&proj);
    let inner = 100_000;
    row("madd_ns", inner, || {
        let mut acc = proj[0];
        for i in 0..inner {
            acc += &aff[i & 1023];
        }
        black_box(acc);
    });
    row("add_ns", inner, || {
        let mut acc = proj[0];
        for i in 0..inner {
            acc += &proj[i & 1023];
        }
        black_box(acc);
    });
    row("double_ns", inner, || {
        let mut acc = proj[0];
        for _ in 0..inner {
            acc.double_in_place();
        }
        black_box(acc);
    });
    let s: Vec<Fr> = (0..1 << 12).map(|_| Fr::rand(rng)).collect();
    let b: Vec<Affine> = (0..1 << 12).map(|i| aff[i % 1024]).collect();
    for log_n in [5usize, 8, 12] {
        let n = 1 << log_n;
        let rounds = (1 << 13) / n;
        row(&format!("msm_2^{log_n}_ns_per_point"), rounds * n, || {
            for _ in 0..rounds {
                black_box(Projective::msm_unchecked(&b[..n], &s[..n]));
            }
        });
    }
}
