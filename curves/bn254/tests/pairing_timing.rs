//! Lightweight wall-clock timing for the pairing paths, run with:
//!   cargo nextest run --release --run-ignored all -E 'test(timing)' --no-capture
//! Reports the best (least-noisy) round's us/iter. Not a correctness test.
use ark_bn254::{Bn254, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_std::{test_rng, vec::Vec, UniformRand};
use core::hint::black_box;
use std::time::Instant;

type G2Prep = <Bn254 as Pairing>::G2Prepared;
type G1Prep = <Bn254 as Pairing>::G1Prepared;

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
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    println!("  {name:24} {:>10.3} us/iter", best / 1000.0);
}

#[test]
#[ignore]
fn timing() {
    let mut rng = test_rng();
    let n = 64usize;
    let g1: Vec<_> = (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
    let g2: Vec<_> = (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
    let g2_prep: Vec<G2Prep> = g2.iter().map(|q| (*q).into()).collect();
    let g1_prep: Vec<G1Prep> = g1.iter().map(|p| (*p).into()).collect();
    let ml: Vec<_> = g1_prep
        .iter()
        .zip(&g2_prep)
        .map(|(p, q)| Bn254::multi_miller_loop([p.clone()], [q.clone()]))
        .collect();

    println!("\n=== Bn254 pairing timing (best of rounds) ===");
    {
        use ark_algebra_test_templates::Sha256;
        use ark_ec::{
            hashing::{
                curve_maps::svdw::SVDWMap, map_to_curve_hasher::MapToCurveBasedHasher, HashToCurve,
            },
            AffineRepr,
        };
        use ark_ff::field_hashers::DefaultFieldHasher;
        let g1h = MapToCurveBasedHasher::<
            G1Projective,
            DefaultFieldHasher<Sha256, 128>,
            SVDWMap<ark_bn254::g1::Config>,
        >::new(b"BN254G1_XMD:SHA-256_SVDW_RO_")
        .unwrap();
        let g2h = MapToCurveBasedHasher::<
            G2Projective,
            DefaultFieldHasher<Sha256, 128>,
            SVDWMap<ark_bn254::g2::Config>,
        >::new(b"BN254G2_XMD:SHA-256_SVDW_RO_")
        .unwrap();
        let mut ctr = 0u64;
        bench("h2c_g1", 3000, 12, || {
            ctr += 1;
            let _ = black_box(g1h.hash(&ctr.to_le_bytes()));
        });
        bench("h2c_g2", 3000, 12, || {
            ctr += 1;
            let _ = black_box(g2h.hash(&ctr.to_le_bytes()));
        });
        // A point outside the order-`r` subgroup.
        let q = loop {
            let x = ark_bn254::Fq2::rand(&mut rng);
            if let Some(q) = ark_bn254::G2Affine::get_point_from_x_unchecked(x, false) {
                break q;
            }
        };
        bench("g2_clear_cofactor", 3000, 12, || { black_box(black_box(q).clear_cofactor()); });
        bench("g2_mul_by_cofactor", 3000, 12, || { black_box(black_box(q).mul_by_cofactor()); });
    }
    {
        use ark_ec::{scalar_mul::double_and_add, PrimeGroup};
        use ark_ff::PrimeField;
        let s = <Bn254 as Pairing>::ScalarField::rand(&mut rng);
        let sb = s.into_bigint();
        let q = G2Projective::rand(&mut rng);
        bench("g2_mul_glv", 3000, 12, || { let _ = black_box(black_box(q).mul_bigint(sb)); });
        bench("g2_mul_double_add", 3000, 12, || { let _ = black_box(double_and_add(&black_box(q), sb)); });
    }
    {
        let gt = Bn254::pairing(g1[0], g2[0]);
        bench("gt_in_group_fast", 3000, 12, || { let _ = black_box(black_box(gt).is_in_group()); });
        bench("gt_in_group_naive", 800, 12, || { let _ = black_box(black_box(gt).is_in_group_naive()); });
    }
    {
        use ark_ec::pairing::Pairing;
        use ark_ff::{CyclotomicMultSubgroup, PrimeField, UniformRand as _};
        let gt0 = Bn254::pairing(g1[0], g2[0]).0;
        let s = ark_bn254::Fr::rand(&mut rng).into_bigint();
        let sr: &[u64] = s.as_ref();
        bench("gt_scalar_mul_gls", 2000, 12, || { let _ = black_box(<Bn254 as Pairing>::gt_exp(black_box(&gt0), sr)); });
        bench("gt_scalar_mul_wnaf", 2000, 12, || { let _ = black_box(black_box(gt0).cyclotomic_exp(sr)); });
    }
    let mut i = 0usize;
    bench("g2_prep", 2000, 12, || {
        i = (i + 1) % n;
        black_box(G2Prep::from(black_box(g2[i])));
    });
    bench("miller_1pair", 800, 12, || {
        i = (i + 1) % n;
        let _ = black_box(Bn254::multi_miller_loop([g1_prep[i].clone()], [g2_prep[i].clone()]));
    });
    bench("final_exp", 500, 12, || {
        i = (i + 1) % n;
        black_box(Bn254::final_exponentiation(black_box(ml[i])));
    });
    bench("pairing_1pair", 300, 12, || {
        i = (i + 1) % n;
        let _ = black_box(Bn254::pairing(g1[i], g2[i]));
    });
    {
        // Groth16 shape: the proof's `B` fresh, the verifying key's `gamma` and `delta`
        // normalized, all in one Miller loop.
        let mut norm = g2_prep.clone();
        norm.iter_mut().for_each(G2Prep::normalize_lines);
        bench("miller_3_fresh", 400, 12, || {
            i = (i + 1) % (n - 2);
            let _ = black_box(Bn254::multi_miller_loop(
                g1_prep[i..i + 3].to_vec(),
                g2_prep[i..i + 3].to_vec(),
            ));
        });
        bench("miller_3_mixed", 400, 12, || {
            i = (i + 1) % (n - 2);
            let _ = black_box(Bn254::multi_miller_loop(
                g1_prep[i..i + 3].to_vec(),
                [g2_prep[i].clone(), norm[i + 1].clone(), norm[i + 2].clone()],
            ));
        });
    }
    for k in [1usize, 2, 4, 5, 8, 9] {
        bench(&format!("multipairing_{k}"), 150, 8, || {
            i = (i + 1) % (n - k);
            let _ = black_box(Bn254::multi_pairing(g1[i..i + k].to_vec(), g2[i..i + k].to_vec()));
        });
    }
}
