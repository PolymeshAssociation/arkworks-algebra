//! `SeleneConfig`'s SWU map against a copy of the config that keeps the default two-root
//! `sqrt_or_zeta_sqrt`. `map_matches_two_root_default` checks both give the same points;
//! `timing` compares `hash_to_curve`, run with:
//!   cargo nextest run --release -p ark-selene --run-ignored all -E 'test(timing)' --no-capture
use ark_algebra_test_templates::sha2::Sha256;
use ark_ec::{
    hashing::{
        curve_maps::swu::{SWUConfig, SWUMap},
        map_to_curve_hasher::{MapToCurve, MapToCurveBasedHasher},
        HashToCurve,
    },
    models::CurveConfig,
    short_weierstrass::{Affine, Projective, SWCurveConfig},
};
use ark_ff::field_hashers::DefaultFieldHasher;
use ark_selene::{Fq, Fr, SeleneConfig, G_GENERATOR_X, G_GENERATOR_Y};
use ark_std::{test_rng, UniformRand};
use core::hint::black_box;
use std::time::Instant;

/// `SeleneConfig` with the default `sqrt_or_zeta_sqrt`.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
struct TwoRoots;

impl CurveConfig for TwoRoots {
    type BaseField = Fq;
    type ScalarField = Fr;
    const COFACTOR: &'static [u64] = <SeleneConfig as CurveConfig>::COFACTOR;
    const COFACTOR_INV: Fr = <SeleneConfig as CurveConfig>::COFACTOR_INV;
}

impl SWCurveConfig for TwoRoots {
    const COEFF_A: Fq = <SeleneConfig as SWCurveConfig>::COEFF_A;
    const COEFF_B: Fq = <SeleneConfig as SWCurveConfig>::COEFF_B;
    const GENERATOR: Affine<Self> = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);
    type ZeroFlag = ();

    #[inline(always)]
    fn mul_by_a(elem: Fq) -> Fq {
        <SeleneConfig as SWCurveConfig>::mul_by_a(elem)
    }
}

impl SWUConfig for TwoRoots {
    const ZETA: Fq = <SeleneConfig as SWUConfig>::ZETA;
}

#[test]
fn map_matches_two_root_default() {
    let mut rng = test_rng();
    let inputs = [Fq::from(0u64), Fq::from(1u64), -Fq::from(1u64)]
        .into_iter()
        .chain((0..2000).map(|_| Fq::rand(&mut rng)));
    for u in inputs {
        let p = SWUMap::<SeleneConfig>::map_to_curve(u).unwrap();
        let q = SWUMap::<TwoRoots>::map_to_curve(u).unwrap();
        assert_eq!((p.x, p.y), (q.x, q.y), "u = {u}");
    }
}

fn bench<F: FnMut()>(name: &str, iters: usize, rounds: usize, mut f: F) {
    for _ in 0..(iters / 4 + 1) {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    println!("  {name:24} {:>10.3} us/iter", best / 1000.0);
}

#[test]
#[ignore]
fn timing() {
    let one_root = MapToCurveBasedHasher::<
        Projective<SeleneConfig>,
        DefaultFieldHasher<Sha256, 128>,
        SWUMap<SeleneConfig>,
    >::new(b"selene-timing")
    .unwrap();
    let two_roots = MapToCurveBasedHasher::<
        Projective<TwoRoots>,
        DefaultFieldHasher<Sha256, 128>,
        SWUMap<TwoRoots>,
    >::new(b"selene-timing")
    .unwrap();
    let mut rng = test_rng();
    let us: Vec<Fq> = (0..256).map(|_| Fq::rand(&mut rng)).collect();

    println!("\n=== Selene SWU timing (best of rounds, alternating) ===");
    for _ in 0..3 {
        let mut ctr = 0u64;
        bench("h2c two roots (default)", 3000, 12, || {
            ctr += 1;
            let _ = black_box(two_roots.hash(&ctr.to_le_bytes()));
        });
        let mut ctr = 0u64;
        bench("h2c one root (override)", 3000, 12, || {
            ctr += 1;
            let _ = black_box(one_root.hash(&ctr.to_le_bytes()));
        });
        let mut i = 0usize;
        bench("map two roots (default)", 5000, 12, || {
            i = (i + 1) % us.len();
            let _ = black_box(SWUMap::<TwoRoots>::map_to_curve(black_box(us[i])));
        });
        let mut i = 0usize;
        bench("map one root (override)", 5000, 12, || {
            i = (i + 1) % us.len();
            let _ = black_box(SWUMap::<SeleneConfig>::map_to_curve(black_box(us[i])));
        });
    }
}
