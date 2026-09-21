//! The fixed-window `pow` on a dense exponent: secp256k1's `p = 3 mod 4` square root raises to
//! `(p + 1) / 4`, which has almost every bit set, the opposite of Pasta's sparse trace.
//!
//! Run with:
//! `cargo test --release --manifest-path curves/secp256k1/Cargo.toml --test pow_bench -- --ignored --nocapture`

use ark_ff::{pow_binary, BigInteger, Field, PrimeField, UniformRand};
use ark_secp256k1::Fq;
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn dense_exponent() {
    const REPS: usize = 11;
    let rng = &mut test_rng();

    // (p + 1) / 4, the exponent `SqrtPrecomputation::Case3Mod4` uses.
    let mut exp = Fq::MODULUS;
    exp.add_with_carry(&<Fq as PrimeField>::BigInt::from(1u64));
    exp.div2();
    exp.div2();
    let bits = ark_ff::significant_bits(exp.as_ref());
    let weight: u32 = exp.as_ref().iter().map(|l| l.count_ones()).sum();

    let bases: Vec<Fq> = (0..200).map(|_| Fq::rand(rng)).collect();
    assert!(bases.iter().all(|a| a.pow(exp) == pow_binary(a, exp.as_ref())));

    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        let t = Instant::now();
        for a in &bases {
            core::hint::black_box(pow_binary(a, exp.as_ref()));
        }
        let old = t.elapsed().as_secs_f64() / bases.len() as f64;
        let t = Instant::now();
        for a in &bases {
            core::hint::black_box(a.pow(exp));
        }
        let new = t.elapsed().as_secs_f64() / bases.len() as f64;
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    ratios.sort_by(f64::total_cmp);
    println!(
        "secp256k1 sqrt exponent ({bits} bits, Hamming weight {weight}): binary {:.2} us, pow {:.2} us, {:.2}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        ratios[REPS / 2]
    );

    let squares: Vec<Fq> = (0..200).map(|_| Fq::rand(rng).square()).collect();
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for a in &squares {
            core::hint::black_box(a.sqrt().unwrap());
        }
        times.push(t.elapsed().as_secs_f64() / squares.len() as f64);
    }
    times.sort_by(f64::total_cmp);
    println!("secp256k1 sqrt: {:.2} us", times[REPS / 2] * 1e6);
}
