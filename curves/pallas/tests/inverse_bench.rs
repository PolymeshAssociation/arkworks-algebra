//! Divstep (safegcd) inversion against the BEA it replaced.
//! 
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test inverse_bench -- --ignored --nocapture`

use ark_ec::CurveGroup;
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Fq, Fr, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

const VALUES: usize = 100_000;
const REPS: usize = 11;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn inverse() {
    single::<Fq>("Pallas Fq");
    single::<Fr>("Pallas Fr");
    batch::<Fq>("Pallas Fq", 8);
    batch::<Fq>("Pallas Fq", 256);
    normalize_batch(8);
    normalize_batch(256);
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

/// One `inverse` against one `bea_inverse`, over the same values.
fn single<F: PrimeField>(name: &str)
where
    F: BeaInverse,
{
    let rng = &mut test_rng();
    let values: Vec<F> = (0..VALUES).map(|_| F::rand(rng)).collect();
    assert!(values
        .iter()
        .all(|a| a.inverse().unwrap() == a.bea().unwrap()));

    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        let t = Instant::now();
        let acc = values.iter().fold(F::one(), |acc, a| acc + a.bea().unwrap());
        let old = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let acc2 = values
            .iter()
            .fold(F::one(), |acc, a| acc + a.inverse().unwrap());
        let new = t.elapsed().as_secs_f64();
        assert_eq!(acc, acc2);
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    println!(
        "{name} single: bea {:.0} ns, divstep {:.0} ns, {:.2}x",
        t_old / (REPS * VALUES) as f64 * 1e9,
        t_new / (REPS * VALUES) as f64 * 1e9,
        median(ratios)
    );
}

/// Montgomery's trick over `n` elements, with only the single inversion swapped.
fn batch<F: PrimeField>(name: &str, n: usize)
where
    F: BeaInverse,
{
    let rng = &mut test_rng();
    let src: Vec<F> = (0..n).map(|_| F::rand(rng)).collect();
    let reps = REPS.max(1);
    let rounds = 20_000 / n.max(1) + 1;

    let run = |bea: bool| {
        let mut v = src.clone();
        let t = Instant::now();
        for _ in 0..rounds {
            batch_inv(&mut v, bea);
        }
        t.elapsed().as_secs_f64() / (rounds * n) as f64
    };
    let mut a = src.clone();
    let mut b = src.clone();
    batch_inv(&mut a, false);
    batch_inv(&mut b, true);
    assert_eq!(a, b);

    let mut ratios = Vec::with_capacity(reps);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..reps {
        let old = run(true);
        let new = run(false);
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    println!(
        "{name} batch({n}): bea {:.0} ns/elem, divstep {:.0} ns/elem, {:.2}x",
        t_old / reps as f64 * 1e9,
        t_new / reps as f64 * 1e9,
        median(ratios)
    );
}

/// [`ark_ff::fields::serial_batch_inversion_and_mul`] with `coeff = 1`, with the single
/// inversion selectable.
fn batch_inv<F: PrimeField + BeaInverse>(v: &mut [F], bea: bool) {
    let mut prod = Vec::with_capacity(v.len());
    let mut tmp = F::one();
    for f in v.iter().filter(|f| !f.is_zero()) {
        tmp *= f;
        prod.push(tmp);
    }
    tmp = if bea {
        tmp.bea().unwrap()
    } else {
        tmp.inverse().unwrap()
    };
    for (f, s) in v
        .iter_mut()
        .rev()
        .filter(|f| !f.is_zero())
        .zip(prod.into_iter().rev().skip(1).chain(Some(F::one())))
    {
        let new_tmp = tmp * *f;
        *f = tmp * &s;
        tmp = new_tmp;
    }
}

/// The consumer this pays off in: one inversion per call, whatever `n` is.
fn normalize_batch(n: usize) {
    let rng = &mut test_rng();
    let points: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let rounds = 20_000 / n + 1;
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = G::normalize_batch(&points);
        }
        times.push(t.elapsed().as_secs_f64() / rounds as f64);
    }
    println!(
        "normalize_batch({n}): {:.2} us/call",
        median(times) * 1e6
    );
}

/// Access to the replaced BEA path through the `PrimeField` bound.
trait BeaInverse: Sized {
    fn bea(&self) -> Option<Self>;
}

impl BeaInverse for Fq {
    fn bea(&self) -> Option<Self> {
        self.bea_inverse()
    }
}

impl BeaInverse for Fr {
    fn bea(&self) -> Option<Self> {
        self.bea_inverse()
    }
}
