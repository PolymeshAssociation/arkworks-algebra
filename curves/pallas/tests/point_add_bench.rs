//! Point-addition timings, compared across builds by running this test from each and taking
//! per-row medians across interleaved runs.
//!
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml [--features asm] --test point_add_bench -- --ignored --nocapture`

use ark_ec::{scalar_mul::glv::GLVConfig, AdditiveGroup, CurveGroup, VariableBaseMSM};
use ark_ff::UniformRand;
use ark_pallas::{Affine, Fr, PallasConfig, Projective};
use ark_std::test_rng;
use std::{hint::black_box, time::Instant};

const SAMPLES: usize = 1024;
const REPS: usize = 15;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn row(label: &str, per: usize, mut f: impl FnMut()) {
    f();
    let mut t = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64() * 1e9 / per as f64);
    }
    println!("ROW {label} {:.2}", median(t));
}

#[test]
#[ignore = "timing; run explicitly with --release --nocapture"]
fn point_add() {
    println!(
        "asm {} debug_assertions {}",
        cfg!(feature = "asm"),
        cfg!(debug_assertions)
    );
    let rng = &mut test_rng();
    let proj: Vec<Projective> = (0..SAMPLES).map(|_| Projective::rand(rng)).collect();
    let aff: Vec<Affine> = Projective::normalize_batch(&proj);
    let scalars: Vec<Fr> = (0..SAMPLES).map(|_| Fr::rand(rng)).collect();

    let inner = 200_000;
    row("madd_latency_ns", inner, || {
        let mut acc = proj[0];
        for i in 0..inner {
            acc += &aff[i & (SAMPLES - 1)];
        }
        black_box(acc);
    });
    row("madd_throughput_ns", inner, || {
        let mut acc = [proj[0], proj[1], proj[2], proj[3]];
        for i in (0..inner).step_by(4) {
            for l in 0..4 {
                acc[l] += &aff[(i + l) & (SAMPLES - 1)];
            }
        }
        black_box(acc);
    });
    row("add_latency_ns", inner, || {
        let mut acc = proj[0];
        for i in 0..inner {
            acc += &proj[i & (SAMPLES - 1)];
        }
        black_box(acc);
    });
    row("add_throughput_ns", inner, || {
        let mut acc = [proj[0], proj[1], proj[2], proj[3]];
        for i in (0..inner).step_by(4) {
            for l in 0..4 {
                acc[l] += &proj[(i + l) & (SAMPLES - 1)];
            }
        }
        black_box(acc);
    });
    row("double_latency_ns", inner, || {
        let mut acc = proj[0];
        for _ in 0..inner {
            acc.double_in_place();
        }
        black_box(acc);
    });
    let muls = 2048;
    row("glv_mul_projective_ns", muls, || {
        for i in 0..muls {
            let j = i & (SAMPLES - 1);
            black_box(PallasConfig::glv_mul_projective(proj[j], scalars[j]));
        }
    });
    row("glv_mul_affine_ns", muls, || {
        for i in 0..muls {
            let j = i & (SAMPLES - 1);
            black_box(PallasConfig::glv_mul_affine_projective(aff[j], scalars[j]));
        }
    });
    let big: Vec<Affine> = (0..1 << 12).map(|i| aff[i % SAMPLES]).collect();
    let big_s: Vec<Fr> = (0..1 << 12).map(|_| Fr::rand(rng)).collect();
    for log_n in [3usize, 5, 6, 8, 10, 12] {
        let n = 1 << log_n;
        let rounds = (1 << 14) / n;
        row(&format!("msm_2^{log_n}_ns_per_point"), rounds * n, || {
            for r in 0..rounds {
                let off = (r * 7) % (big.len() - n + 1);
                black_box(Projective::msm_unchecked(
                    &big[off..off + n],
                    &big_s[off..off + n],
                ));
            }
        });
    }
}
