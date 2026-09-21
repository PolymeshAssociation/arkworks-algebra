//! Fixed-window `pow` against the binary square-and-multiply it replaces.
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test pow_bench -- --ignored --nocapture`

use ark_ff::{pow_binary, pow_windowed, Field, PrimeField, UniformRand};
use ark_pallas::Fq;
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

const REPS: usize = 11;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn pow() {
    for bits in [32usize, 64, 96, 128, 129, 191, 222, 255] {
        crossover(bits);
    }
    sqrt_exponent();
    sqrt();
    legendre();
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

/// Where the 14-multiply table starts paying for itself. All-ones exponents are the binary
/// path's worst case, random ones its average.
fn crossover(bits: usize) {
    let rng = &mut test_rng();
    let limbs = bits.div_ceil(64);
    // Normalize a limb vector to exactly `bits` significant bits: clear the limbs above bit
    // `bits`, then set bit `bits - 1` so `significant_bits(exp) == bits` whatever the random
    // draw. The `bits % 64 != 0` guard skips the high-limb mask when `bits` fills whole limbs,
    // where `1u64 << 64` would overflow the shift (and no masking is needed).
    let mask = |e: &mut Vec<u64>| {
        if bits % 64 != 0 {
            e[limbs - 1] &= (1u64 << (bits % 64)) - 1;
        }
        e[(bits - 1) / 64] |= 1u64 << ((bits - 1) % 64);
    };
    // Three shapes bracket the dispatch decision at this length: all bits set is binary's worst
    // case and the window's best, random is the average, one bit set is the window's worst case
    // (its 14-multiply table can never pay off).
    let mut ones = vec![u64::MAX; limbs];
    mask(&mut ones);
    let mut random: Vec<u64> = (0..limbs).map(|_| u64::rand(rng)).collect();
    mask(&mut random);
    let mut sparse = vec![0u64; limbs];
    mask(&mut sparse);

    let bases: Vec<Fq> = (0..200).map(|_| Fq::rand(rng)).collect();
    let mut out = String::new();
    for (label, exp) in [("ones", &ones), ("random", &random), ("top bit only", &sparse)] {
        // The paths must agree before either is timed.
        assert!(bases
            .iter()
            .all(|a| pow_windowed(a, exp, bits) == pow_binary(a, exp)));
        let mut ratios = Vec::with_capacity(REPS);
        let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
        for _ in 0..REPS {
            // Time both back-to-back within one rep so their ratio cancels clock drift; `black_box`
            // stops the optimizer hoisting the pow out of the loop or dropping it as dead.
            let t = Instant::now();
            for a in &bases {
                core::hint::black_box(pow_binary(a, exp));
            }
            let old = t.elapsed().as_secs_f64() / bases.len() as f64;
            let t = Instant::now();
            for a in &bases {
                core::hint::black_box(pow_windowed(a, exp, bits));
            }
            let new = t.elapsed().as_secs_f64() / bases.len() as f64;
            ratios.push(old / new);
            t_old += old;
            t_new += new;
        }
        // Hamming weight is the quantity the dispatch keys on; `dispatch` prints the path
        // `Field::pow` would actually pick here, which should match the faster measured time. The
        // median over reps rejects timing outliers.
        let weight: u32 = exp.iter().map(|l| l.count_ones()).sum();
        out += &format!(
            " | {label} (weight {weight}, dispatch {}) binary {:5.2} us, window {:5.2} us, {:.2}x",
            if ark_ff::use_pow_windowed(exp) { "window" } else { "binary" },
            t_old / REPS as f64 * 1e6,
            t_new / REPS as f64 * 1e6,
            median(ratios)
        );
    }
    println!("{bits:3} bits{out}");
}

/// The exact exponentiation Sarkar's square root spends most of its time in.
fn sqrt_exponent() {
    let rng = &mut test_rng();
    let exp = Fq::TRACE_MINUS_ONE_DIV_TWO;
    let bits = ark_ff::significant_bits(exp.as_ref());
    let weight: u32 = exp.as_ref().iter().map(|l| l.count_ones()).sum();
    let dispatch = if ark_ff::use_pow_windowed(exp.as_ref()) { "window" } else { "binary" };
    let bases: Vec<Fq> = (0..200).map(|_| Fq::rand(rng)).collect();
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
    println!(
        "sqrt exponent ({bits} bits, Hamming weight {weight}, dispatch {dispatch}): binary {:.2} us, pow {:.2} us, {:.2}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        median(ratios)
    );
}

/// The consumer the item targets: point decompression's square root.
fn sqrt() {
    let rng = &mut test_rng();
    let squares: Vec<Fq> = (0..200).map(|_| Fq::rand(rng).square()).collect();
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for a in &squares {
            core::hint::black_box(a.sqrt().unwrap());
        }
        times.push(t.elapsed().as_secs_f64() / squares.len() as f64);
    }
    println!("sqrt (Sarkar): {:.2} us", median(times) * 1e6);
}

fn legendre() {
    let rng = &mut test_rng();
    let exp = Fq::MODULUS_MINUS_ONE_DIV_TWO;
    let weight: u32 = exp.as_ref().iter().map(|l| l.count_ones()).sum();
    println!(
        "legendre exponent: {} bits, Hamming weight {weight}",
        ark_ff::significant_bits(exp.as_ref())
    );
    let values: Vec<Fq> = (0..200).map(|_| Fq::rand(rng)).collect();
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for a in &values {
            core::hint::black_box(a.legendre());
        }
        times.push(t.elapsed().as_secs_f64() / values.len() as f64);
    }
    println!("legendre:      {:.2} us", median(times) * 1e6);
}
