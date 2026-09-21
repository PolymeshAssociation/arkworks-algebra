//! Multi-lane batch inversion against the single-chain Montgomery trick it replaced
//!
//! Run with:
//! `cargo test --release --manifest-path curves/pallas/Cargo.toml --test batch_inversion_bench -- --ignored --nocapture`

use ark_ec::CurveGroup;
use ark_ff::{
    serial_batch_inversion_and_mul, serial_batch_inversion_and_mul_lanes,
    serial_batch_inversion_and_mul_single_chain, Field, One, UniformRand, Zero,
};
use ark_pallas::{Affine, Fq, Projective as G};
use ark_std::{test_rng, vec::Vec};
use std::time::Instant;

const REPS: usize = 11;

#[test]
#[ignore = "timing comparison; run explicitly with --release --nocapture"]
fn batch_inversion() {
    let inv = single_inversion();
    println!("one Fq::inverse: {:.0} ns", inv * 1e9);
    mul_ceiling();
    for n in [4, 8, 16, 32, 64, 131, 256, 1024, 4096, 16384] {
        lanes(n, inv);
    }
    println!();
    for n in [8usize, 32, 131, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        entry_point(n);
    }
    println!();
    for n in [8usize, 131, 1024, 16384] {
        library_normalize_batch(n);
    }
    normalize_batch(1 << 10);
    normalize_batch(1 << 14);
}

fn median(mut r: Vec<f64>) -> f64 {
    r.sort_by(f64::total_cmp);
    r[r.len() / 2]
}

/// The one inversion every variant pays, so the gate can be quoted with it excluded.
fn single_inversion() -> f64 {
    let rng = &mut test_rng();
    let values: Vec<Fq> = (0..100_000).map(|_| Fq::rand(rng)).collect();
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        let acc = values.iter().fold(Fq::one(), |acc, a| acc + a.inverse().unwrap());
        times.push(t.elapsed().as_secs_f64() / values.len() as f64);
        assert!(!acc.is_zero());
    }
    median(times)
}

fn lanes(n: usize, inv: f64) {
    let rng = &mut test_rng();
    let src: Vec<Fq> = (0..n).map(|_| Fq::rand(rng)).collect();
    let coeff = Fq::rand(rng);
    let rounds = (1 << 18) / n + 1;

    let mut expected = src.clone();
    serial_batch_inversion_and_mul_single_chain(&mut expected, &coeff);
    for f in [
        serial_batch_inversion_and_mul_lanes::<Fq, 2> as fn(&mut [Fq], &Fq),
        serial_batch_inversion_and_mul_lanes::<Fq, 4>,
        serial_batch_inversion_and_mul_lanes::<Fq, 8>,
        serial_batch_inversion_and_mul,
    ] {
        let mut v = src.clone();
        f(&mut v, &coeff);
        assert_eq!(v, expected, "n = {n}");
    }

    let run = |f: fn(&mut [Fq], &Fq)| {
        let mut v = src.clone();
        let t = Instant::now();
        for _ in 0..rounds {
            f(&mut v, &coeff);
        }
        t.elapsed().as_secs_f64() / (rounds * n) as f64
    };

    let mut per: Vec<Vec<f64>> = vec![Vec::with_capacity(REPS); 4];
    for _ in 0..REPS {
        per[0].push(run(serial_batch_inversion_and_mul_single_chain));
        per[1].push(run(serial_batch_inversion_and_mul_lanes::<Fq, 2>));
        per[2].push(run(serial_batch_inversion_and_mul_lanes::<Fq, 4>));
        per[3].push(run(serial_batch_inversion_and_mul_lanes::<Fq, 8>));
    }
    let t: Vec<f64> = per.into_iter().map(median).collect();
    // The single inversion is one fixed cost per call, whatever the lane count.
    let ex = |x: f64| x - inv / n as f64;
    println!(
        "n = {n:5}: single-chain {:6.1} ns/elem | 2 lanes {:.2}x | 4 lanes {:.2}x | 8 lanes {:.2}x \
         || excluding the inversion: {:.2}x / {:.2}x / {:.2}x",
        t[0] * 1e9,
        t[0] / t[1],
        t[0] / t[2],
        t[0] / t[3],
        ex(t[0]) / ex(t[1]),
        ex(t[0]) / ex(t[2]),
        ex(t[0]) / ex(t[3]),
    );
}

/// The floor the Montgomery trick can reach: 3 multiplies per element at throughput, against
/// the 2 per element the single chain puts on the dependency path at latency.
fn mul_ceiling() {
    let rng = &mut test_rng();
    let y = Fq::rand(rng);
    let n = 3_000_000usize;
    let mut t = [0f64; 2];
    for (slot, chains) in t.iter_mut().zip([1usize, 4]) {
        let mut acc: Vec<Fq> = (0..chains).map(|_| Fq::rand(rng)).collect();
        let mut best = f64::MAX;
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..n / chains {
                for a in acc.iter_mut() {
                    *a *= y;
                }
            }
            best = best.min(start.elapsed().as_secs_f64() / (n / chains * chains) as f64);
        }
        assert!(!acc[0].is_zero());
        *slot = best;
    }
    println!(
        "Fq mul: {:.2} ns latency, {:.2} ns throughput, ratio {:.2}. The trick spends 3 muls per \
         element and puts 2 of them on the dependency chain, so the single chain's floor is \
         max(2 x latency, 3 x throughput) = {:.1} ns/elem and every lane count shares the same \
         {:.1} ns/elem throughput floor.",
        t[0] * 1e9,
        t[1] * 1e9,
        t[0] / t[1],
        (2.0 * t[0]).max(3.0 * t[1]) * 1e9,
        3.0 * t[1] * 1e9,
    );
}

/// The public entry point, which under `parallel` splits the slice into one chunk per thread and
/// gives each its own inversion: at 16 threads `batch_inversion(131)` is 16 inversions of 8, not
/// one of 131. That shrinks what A4's lanes have to work with and multiplies what A2's cheaper
/// inversion is worth, so the two items move in opposite directions when the feature is on.
fn entry_point(n: usize) {
    let rng = &mut test_rng();
    let src: Vec<Fq> = (0..n).map(|_| Fq::rand(rng)).collect();
    let rounds = (1 << 18) / n + 1;
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let mut v = src.clone();
        let t = Instant::now();
        for _ in 0..rounds {
            ark_ff::batch_inversion(&mut v);
        }
        times.push(t.elapsed().as_secs_f64() / (rounds * n) as f64);
    }
    // Mirrors `batch_inversion_and_mul`'s own rule; rayon defaults to `available_parallelism`.
    let chunk = if cfg!(feature = "parallel") && n >= 4096 {
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        (n / threads).max(256)
    } else {
        n
    };
    println!(
        "batch_inversion({n:5}) entry point: {:6.1} ns/elem, chunk {chunk} => {} inversion(s)",
        median(times) * 1e9,
        n.div_ceil(chunk),
    );
}

fn library_normalize_batch(n: usize) {
    let rng = &mut test_rng();
    let points: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let rounds = (1 << 16) / n + 1;
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        for _ in 0..rounds {
            let _ = core::hint::black_box(G::normalize_batch(&points));
        }
        times.push(t.elapsed().as_secs_f64() / rounds as f64);
    }
    println!(
        "G::normalize_batch({n:5}): {:9.2} us/call",
        median(times) * 1e6
    );
}

/// [`Projective::normalize_batch`]'s body with the batch inversion selectable, so the consumer
/// ratio is measured rather than derived.
fn normalize_batch(n: usize) {
    let rng = &mut test_rng();
    let points: Vec<G> = (0..n).map(|_| G::rand(rng)).collect();
    let rounds = (1 << 16) / n + 1;

    let run = |single: bool| -> (f64, Vec<Affine>) {
        let t = Instant::now();
        let mut out = Vec::new();
        for _ in 0..rounds {
            let mut z_s: Vec<Fq> = points.iter().map(|g| g.z).collect();
            if single {
                serial_batch_inversion_and_mul_single_chain(&mut z_s, &Fq::one());
            } else {
                serial_batch_inversion_and_mul(&mut z_s, &Fq::one());
            }
            out = points
                .iter()
                .zip(z_s)
                .map(|(g, z)| match g.is_zero() {
                    true => Affine::identity(),
                    false => {
                        let z2 = z.square();
                        Affine::new_unchecked(g.x * z2, g.y * z2 * z)
                    },
                })
                .collect();
        }
        (t.elapsed().as_secs_f64() / rounds as f64, out)
    };
    assert_eq!(run(true).1, run(false).1);
    assert_eq!(run(false).1, G::normalize_batch(&points));

    let mut ratios = Vec::with_capacity(REPS);
    let (mut t_new, mut t_old) = (0.0f64, 0.0f64);
    for _ in 0..REPS {
        let old = run(true).0;
        let new = run(false).0;
        ratios.push(old / new);
        t_old += old;
        t_new += new;
    }
    println!(
        "normalize_batch({n}): single-chain {:.1} us, 2 lanes {:.1} us, {:.3}x",
        t_old / REPS as f64 * 1e6,
        t_new / REPS as f64 * 1e6,
        median(ratios)
    );
}
