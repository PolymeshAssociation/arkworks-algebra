//! Lightweight wall-clock timing for the pairing paths, run with:
//!   cargo nextest run --release --run-ignored all -E 'test(timing)' --no-capture
//! Reports the best (least-noisy) round's us/iter. Not a correctness test.
use ark_bls12_381::{Bls12_381, Fq, Fq12, Fq2, G1Projective, G2Projective};
use ark_ec::{pairing::Pairing, CurveGroup};
use ark_std::{test_rng, vec::Vec, UniformRand};
use core::hint::black_box;
use std::time::Instant;

type G2Prep = <Bls12_381 as Pairing>::G2Prepared;
type G1Prep = <Bls12_381 as Pairing>::G1Prepared;

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
    let g1: Vec<_> = (0..n)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();
    let g2: Vec<_> = (0..n)
        .map(|_| G2Projective::rand(&mut rng).into_affine())
        .collect();
    let g2_prep: Vec<G2Prep> = g2.iter().map(|q| (*q).into()).collect();
    let g1_prep: Vec<G1Prep> = g1.iter().map(|p| (*p).into()).collect();
    let ml: Vec<_> = g1_prep
        .iter()
        .zip(&g2_prep)
        .map(|(p, q)| Bls12_381::multi_miller_loop([p.clone()], [q.clone()]))
        .collect();

    println!("\n=== Bls12_381 pairing timing (best of rounds) ===");
    {
        use ark_ff::Field;
        let a = Fq::rand(&mut rng);
        let a2 = Fq2::rand(&mut rng);
        let a12 = Fq12::rand(&mut rng);
        bench("fq_inverse", 20000, 12, || {
            black_box(black_box(a).inverse());
        });
        bench("fq2_inverse", 20000, 12, || {
            black_box(black_box(a2).inverse());
        });
        bench("fq12_inverse", 5000, 12, || {
            black_box(black_box(a12).inverse());
        });
        // squares of random elements are QRs, so sqrt succeeds (measures the common path)
        let aq = a.square();
        let a2q = a2.square();
        bench("fq_sqrt", 5000, 12, || {
            black_box(black_box(aq).sqrt());
        });
        bench("fq2_sqrt", 3000, 12, || {
            black_box(black_box(a2q).sqrt());
        });
        use ark_ff::CyclotomicMultSubgroup;
        // put a12 into the cyclotomic subgroup so cyclotomic_square is valid
        let cyc = Bls12_381::final_exponentiation(ark_ec::pairing::MillerLoopOutput(a12))
            .unwrap()
            .0;
        bench("fq12_square", 20000, 12, || {
            black_box(black_box(a12).square());
        });
        bench("fq12_cyclo_square", 20000, 12, || {
            black_box(black_box(cyc).cyclotomic_square());
        });
        bench("fq12_mul", 20000, 12, || {
            black_box(black_box(a12) * black_box(cyc));
        });
        bench("fq12_frobenius1", 20000, 12, || {
            let mut t = a12;
            t.frobenius_map_in_place(1);
            black_box(t);
        });
        bench("exp_by_x", 2000, 12, || {
            black_box(
                <ark_bls12_381::Config as ark_ec::bls12::Bls12Config>::exp_by_x(black_box(cyc)),
            );
        });
    }
    {
        use ark_ec::scalar_mul::double_and_add;
        use ark_ff::PrimeField;
        let s = <Bls12_381 as Pairing>::ScalarField::rand(&mut rng);
        let sb = s.into_bigint();
        let q = G2Projective::rand(&mut rng);
        bench("g2_mul_gls", 3000, 12, || {
            let _ = black_box(black_box(q) * black_box(s));
        });
        bench("g2_mul_double_add", 3000, 12, || {
            let _ = black_box(double_and_add(&black_box(q), sb));
        });
    }
    {
        use ark_ec::pairing::Pairing;
        use ark_ff::{CyclotomicMultSubgroup, PrimeField, UniformRand as _};
        let gt0 = Bls12_381::pairing(g1[0], g2[0]).0;
        let s = ark_bls12_381::Fr::rand(&mut rng).into_bigint();
        let sr: &[u64] = s.as_ref();
        bench("gt_scalar_mul_gls", 2000, 12, || {
            let _ = black_box(<Bls12_381 as Pairing>::gt_exp(black_box(&gt0), sr));
        });
        bench("gt_scalar_mul_wnaf", 2000, 12, || {
            let _ = black_box(black_box(gt0).cyclotomic_exp(sr));
        });
    }
    {
        let gt = Bls12_381::pairing(g1[0], g2[0]);
        bench("gt_in_group_fast", 3000, 12, || {
            let _ = black_box(black_box(gt).is_in_group());
        });
        bench("gt_in_group_naive", 800, 12, || {
            let _ = black_box(black_box(gt).is_in_group_naive());
        });
    }
    let mut i = 0usize;
    bench("g2_prep", 2000, 12, || {
        i = (i + 1) % n;
        black_box(G2Prep::from(black_box(g2[i])));
    });
    bench("miller_1pair", 800, 12, || {
        i = (i + 1) % n;
        let _ = black_box(Bls12_381::multi_miller_loop(
            [g1_prep[i].clone()],
            [g2_prep[i].clone()],
        ));
    });
    bench("final_exp", 500, 12, || {
        i = (i + 1) % n;
        black_box(Bls12_381::final_exponentiation(black_box(ml[i])));
    });
    bench("pairing_1pair", 300, 12, || {
        i = (i + 1) % n;
        let _ = black_box(Bls12_381::pairing(g1[i], g2[i]));
    });
    {
        use ark_ec::bls12::{Bls12, G2PreparedFixed};
        let fixed: Vec<G2PreparedFixed<ark_bls12_381::Config>> =
            g2.iter().map(|q| (*q).try_into().unwrap()).collect();
        for k in [2usize, 4] {
            let pk: Vec<_> = g1[0..k].to_vec();
            let fk = &fixed[0..k];
            let gp: Vec<_> = g2[0..k].to_vec();
            let g1p: Vec<G1Prep> = g1_prep[0..k].to_vec();
            let g2p: Vec<G2Prep> = g2_prep[0..k].to_vec();
            bench(&format!("miller_fixed_{k}"), 400, 12, || {
                let _ = black_box(Bls12::<ark_bls12_381::Config>::multi_miller_loop_fixed(
                    pk.iter().copied(),
                    fk,
                ));
            });
            bench(&format!("miller_std_{k}"), 400, 12, || {
                let _ = black_box(Bls12_381::multi_miller_loop(
                    g1p.iter().cloned(),
                    g2p.iter().cloned(),
                ));
            });
            let _ = gp;
        }
    }
    {
        // Groth16 shape: the proof's `B` fresh, the verifying key's `gamma` and `delta`
        // normalized, all in one Miller loop.
        let mut norm = g2_prep.clone();
        norm.iter_mut().for_each(G2Prep::normalize_lines);
        bench("miller_3_fresh", 400, 12, || {
            i = (i + 1) % (n - 2);
            let _ = black_box(Bls12_381::multi_miller_loop(
                g1_prep[i..i + 3].to_vec(),
                g2_prep[i..i + 3].to_vec(),
            ));
        });
        bench("miller_3_mixed", 400, 12, || {
            i = (i + 1) % (n - 2);
            let _ = black_box(Bls12_381::multi_miller_loop(
                g1_prep[i..i + 3].to_vec(),
                [g2_prep[i].clone(), norm[i + 1].clone(), norm[i + 2].clone()],
            ));
        });
    }
    for k in [1usize, 2, 4, 5, 8, 9] {
        bench(&format!("multipairing_{k}"), 150, 8, || {
            i = (i + 1) % (n - k);
            let _ = black_box(Bls12_381::multi_pairing(
                g1[i..i + k].to_vec(),
                g2[i..i + k].to_vec(),
            ));
        });
    }
}
