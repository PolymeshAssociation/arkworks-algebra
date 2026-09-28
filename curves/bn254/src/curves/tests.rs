use ark_algebra_test_templates::*;
use ark_ff::fields::Field;

use crate::{Bn254, G1Projective, G2Projective};

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<Bn254>; msm);
test_pairing!(pairing; crate::Bn254);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);

#[test]
fn test_g2_prepared_neg() {
    use ark_ec::{pairing::Pairing, CurveGroup};
    use ark_std::{test_rng, UniformRand, Zero};
    let mut rng = test_rng();
    for _ in 0..10 {
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G2Projective::rand(&mut rng).into_affine();
        let q_prep: <Bn254 as Pairing>::G2Prepared = q.into();
        let neg_prep = -q_prep.clone();
        assert_eq!(Bn254::pairing(p, -q), Bn254::pairing(p, neg_prep.clone()));
        let sum = Bn254::pairing(p, q_prep) + Bn254::pairing(p, neg_prep);
        assert!(sum.is_zero());
    }
}

#[test]
fn test_normalized_lines_mixed_miller_loop() {
    use ark_ec::{bn::G2Prepared, pairing::Pairing, CurveGroup};
    use ark_ff::One;
    use ark_std::{test_rng, vec::Vec, UniformRand};
    type Prep = G2Prepared<crate::Config>;
    let mut rng = test_rng();
    // Up to 9 pairs, so parallel builds split the loop into several chunks.
    for n in [1usize, 2, 3, 5, 9] {
        let ps: Vec<crate::G1Affine> =
            (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<crate::G2Affine> =
            (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let expected = Bn254::multi_pairing(ps.iter().copied(), qs.iter().copied());
        for mask in [0usize, 1, 0b01010, 0b10101, usize::MAX] {
            let preps: Vec<Prep> = qs
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let mut prep = Prep::from(*q);
                    if (mask >> i) & 1 == 1 {
                        prep.normalize_lines();
                        assert!(prep.ell_coeffs.iter().all(|c| c.0.is_one()));
                    }
                    prep
                })
                .collect();
            let ml = Bn254::multi_miller_loop(ps.iter().copied(), preps);
            assert_eq!(
                Bn254::final_exponentiation(ml).unwrap(),
                expected,
                "n = {n}, mask = {mask:b}"
            );
        }
    }
}

#[test]
fn test_normalized_lines_negation_and_raw_fallbacks() {
    use ark_ec::{bn::G2Prepared, pairing::Pairing, CurveGroup};
    use ark_ff::{One, Zero};
    use ark_std::{test_rng, UniformRand};
    type Prep = G2Prepared<crate::Config>;
    let mut rng = test_rng();
    let fe = |p: crate::G1Affine, q: Prep| {
        Bn254::final_exponentiation(Bn254::multi_miller_loop([p], [q])).unwrap()
    };
    let p = G1Projective::rand(&mut rng).into_affine();
    let q = G2Projective::rand(&mut rng).into_affine();
    let mut normalized = Prep::from(q);
    normalized.normalize_lines();

    // Normalizing twice changes nothing, and negation keeps the lines normalized.
    let mut twice = normalized.clone();
    twice.normalize_lines();
    assert_eq!(twice, normalized);
    let neg = -normalized.clone();
    assert!(neg.ell_coeffs.iter().all(|c| c.0.is_one()));
    assert_eq!(fe(p, neg), Bn254::pairing(-p, q));

    // A line with a zero `P.y` coefficient stays raw, including a Frobenius line.
    for k in [17, normalized.ell_coeffs.len() - 1] {
        let mut raw = Prep::from(q);
        raw.ell_coeffs[k].0 = crate::Fq2::zero();
        let mut partly = raw.clone();
        partly.normalize_lines();
        assert_eq!(partly.ell_coeffs[k], raw.ell_coeffs[k]);
        assert!(partly.ell_coeffs[k - 1].0.is_one());
        assert_eq!(fe(p, partly), fe(p, raw));
    }

    // `P.y = 0` has no `1/P.y`, so its pair runs on raw lines.
    let p0 = crate::G1Affine::new_unchecked(crate::Fq::rand(&mut rng), crate::Fq::zero());
    assert_eq!(fe(p0, normalized), fe(p0, Prep::from(q)));
}

#[test]
fn test_gt_membership_fast_matches_naive() {
    use ark_ec::{
        pairing::{Pairing, PairingOutput},
        CurveGroup,
    };
    use ark_ff::{Field, UniformRand};
    use ark_std::test_rng;
    let mut rng = test_rng();
    for _ in 0..20 {
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G2Projective::rand(&mut rng).into_affine();
        let gt = Bn254::pairing(p, q);
        assert!(gt.is_in_group());
        assert!(gt.is_in_group_naive());

        let f = crate::Fq12::rand(&mut rng);
        let rand_out = PairingOutput::<Bn254>(f);
        assert_eq!(rand_out.is_in_group(), rand_out.is_in_group_naive());
        assert!(!rand_out.is_in_group());

        let mut r = f;
        r.frobenius_map_in_place(6);
        r *= f.inverse().unwrap();
        let mut r_p2 = r;
        r_p2.frobenius_map_in_place(2);
        r *= r_p2;
        let cyc = PairingOutput::<Bn254>(r);
        assert_eq!(cyc.is_in_group(), cyc.is_in_group_naive());
        assert!(!cyc.is_in_group());
    }
}

#[test]
fn test_multi_pairing_various_n() {
    use ark_ec::{
        pairing::{Pairing, PairingOutput},
        CurveGroup,
    };
    use ark_std::{test_rng, vec::Vec, UniformRand, Zero};
    let mut rng = test_rng();
    for n in 1..=6usize {
        let ps: Vec<_> = (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<_> = (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let multi = Bn254::multi_pairing(ps.iter().copied(), qs.iter().copied());
        let prod = ps
            .iter()
            .zip(&qs)
            .map(|(p, q)| Bn254::pairing(*p, *q))
            .fold(PairingOutput::zero(), |acc, x| acc + x);
        assert_eq!(multi, prod, "n = {n}");
    }
}

#[test]
fn test_gt_exp_matches_generic() {
    use ark_ec::pairing::Pairing;
    use ark_ff::{CyclotomicMultSubgroup, PrimeField, UniformRand};
    use ark_std::test_rng;
    let mut rng = test_rng();
    let gt = {
        use ark_ec::CurveGroup;
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G2Projective::rand(&mut rng).into_affine();
        Bn254::pairing(p, q).0
    };
    for _ in 0..10 {
        let s = crate::Fr::rand(&mut rng);
        let via_gls = <Bn254 as Pairing>::gt_exp(&gt, s.into_bigint().as_ref());
        let via_generic = gt.cyclotomic_exp(s.into_bigint().as_ref());
        assert_eq!(via_gls, via_generic);
    }
    for s in [crate::Fr::from(0u64), crate::Fr::from(1u64), crate::Fr::from(2u64), -crate::Fr::from(1u64)] {
        let via_gls = <Bn254 as Pairing>::gt_exp(&gt, s.into_bigint().as_ref());
        let via_generic = gt.cyclotomic_exp(s.into_bigint().as_ref());
        assert_eq!(via_gls, via_generic, "s = {s}");
    }
}

#[test]
fn test_exp_by_x_chain_matches_generic() {
    use ark_ec::{bn::BnConfig, pairing::Pairing, CurveGroup};
    use ark_ff::{CyclotomicMultSubgroup, Field, UniformRand};
    let mut rng = ark_std::test_rng();
    for _ in 0..8 {
        let g = Bn254::pairing(
            G1Projective::rand(&mut rng).into_affine(),
            G2Projective::rand(&mut rng).into_affine(),
        )
        .0;
        // f^((p^6 - 1)(p^2 + 1)) is cyclotomic but (almost surely) not in GT.
        let f = crate::Fq12::rand(&mut rng);
        let mut cyc = f;
        cyc.frobenius_map_in_place(6);
        cyc *= f.inverse().unwrap();
        let mut cyc_p2 = cyc;
        cyc_p2.frobenius_map_in_place(2);
        cyc *= cyc_p2;
        for h in [g, cyc] {
            assert_eq!(crate::Config::exp_by_x(h), h.cyclotomic_exp(crate::Config::X));
        }
    }
}
