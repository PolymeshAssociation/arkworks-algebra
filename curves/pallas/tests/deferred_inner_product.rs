//! Deferred inner product against the naive sum.
//!
//! Run the timing test with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test deferred_inner_product -- --ignored --nocapture`

use ark_ff::{BigInt, Field, MontAccumulator, UniformRand, Zero};
use ark_pallas::{fq::FqConfig, fr::FrConfig, Fq, Fr};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

/// Internal representations that stressed the accumulator's carry fold in Zakura's
/// `deferred_field_tests!`; its `Fp` is this crate's `Fq` and its `Fq` is this crate's `Fr`.
const FQ_PAIR: ([u64; 4], [u64; 4]) = (
    [
        0x0361524c2cc0f859,
        0xae68690a78bc7175,
        0xe66cd36e68ef8f5f,
        0x3fa6524a713b7e05,
    ],
    [
        0x7a1c5e3b9d204f61,
        0xc48e0b71a2d5f389,
        0xd9f247a0856c13be,
        0x3d8a19f5e6c7b042,
    ],
);

const FR_PAIR: ([u64; 4], [u64; 4]) = (
    [
        0x31d0b6640589f877,
        0xf87f43fdf6062541,
        0xb7d6467b2f5a522a,
        0x3eb025240950fd13,
    ],
    [
        0x5e9a3c71f8b20d46,
        0xa3d1e6f504879c2b,
        0xcb45a8d2e1f36790,
        0x3c47d2a8b10e5f93,
    ],
);

#[test]
fn adversarial_pairs() {
    check::<Fq>(FQ_PAIR);
    check::<Fr>(FR_PAIR);
}

fn check<F: Field + From<BigInt<4>>>(pair: ([u64; 4], [u64; 4]))
where
    F: NewUnchecked,
{
    let a = F::new_unchecked(BigInt(pair.0));
    let b = F::new_unchecked(BigInt(pair.1));
    for len in [1usize, 2, 99, 100, 101, 10_000] {
        let a = vec![a; len];
        let b = vec![b; len];
        let naive: F = a.iter().zip(&b).map(|(a, b)| *a * b).sum();
        assert_eq!(F::inner_product(&a, &b), naive, "len {len}");
    }
}

#[test]
fn accumulator_matches_inner_product() {
    let rng = &mut test_rng();
    let a: Vec<Fq> = (0..257).map(|_| Fq::rand(rng)).collect();
    let b: Vec<Fq> = (0..257).map(|_| Fq::rand(rng)).collect();
    let mut acc = MontAccumulator::<FqConfig, 4>::ZERO;
    for (a, b) in a.iter().zip(&b) {
        acc.mul_accumulate(a, b);
    }
    assert_eq!(acc.reduce(), Fq::inner_product(&a, &b));
    assert!(MontAccumulator::<FqConfig, 4>::ZERO.reduce().is_zero());
    assert!(MontAccumulator::<FrConfig, 4>::ZERO.reduce().is_zero());
}

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn timing() {
    for n in [2, 8, 64, 256, 1024, 4096, 16384] {
        bench(n);
    }
}

fn bench(n: usize) {
    let rng = &mut test_rng();
    let a: Vec<Fq> = (0..n).map(|_| Fq::rand(rng)).collect();
    let b: Vec<Fq> = (0..n).map(|_| Fq::rand(rng)).collect();
    let naive = || -> Fq { a.iter().zip(&b).map(|(a, b)| *a * b).sum() };
    assert_eq!(Fq::inner_product(&a, &b), naive());

    let reps = 11;
    let rounds = (1 << 20) / n + 1;
    let mut ratios = Vec::with_capacity(reps);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..rounds {
            core::hint::black_box(naive());
        }
        let old = t.elapsed().as_secs_f64() / (rounds * n) as f64;
        let t = Instant::now();
        for _ in 0..rounds {
            core::hint::black_box(Fq::inner_product(&a, &b));
        }
        let new = t.elapsed().as_secs_f64() / (rounds * n) as f64;
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    ratios.sort_by(f64::total_cmp);
    println!(
        "n = {n:5}: naive {:5.2} ns/term, deferred {:5.2} ns/term, {:.2}x",
        t_old / reps as f64 * 1e9,
        t_new / reps as f64 * 1e9,
        ratios[reps / 2]
    );
}

/// `Fp::new_unchecked` is inherent, not on a trait; this exposes it to the generic helper.
trait NewUnchecked: Sized {
    fn new_unchecked(b: BigInt<4>) -> Self;
}

impl NewUnchecked for Fq {
    fn new_unchecked(b: BigInt<4>) -> Self {
        Self::new_unchecked(b)
    }
}

impl NewUnchecked for Fr {
    fn new_unchecked(b: BigInt<4>) -> Self {
        Self::new_unchecked(b)
    }
}
