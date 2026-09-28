use crate::{Bls12_377, G1Projective, G2Projective};
use ark_algebra_test_templates::*;

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<Bls12_377>; msm);
test_pairing!(pairing; crate::Bls12_377);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);
test_h2c!(g1_h2c; "./src/curves/tests"; "BLS12377G1"; crate::g1::Config; crate::Fq; crate::Fq; 1);
test_h2c!(g2_hc2; "./src/curves/tests"; "BLS12377G2"; crate::g2::Config; crate::Fq2; crate::Fq; 2);

#[cfg(test)]
mod test {
    use ark_ec::{
        hashing::{
            curve_maps::wb::{WBConfig, WBMap},
            map_to_curve_hasher::MapToCurveBasedHasher,
            HashToCurve,
        },
        short_weierstrass::Projective,
    };
    use ark_ff::field_hashers::DefaultFieldHasher;
    use ark_std::Zero;

    /// make a simple hash
    fn wb_hash_arbitrary_string_to_curve<WBCurve: WBConfig>() {
        use sha2::Sha256;
        let test_wb_to_curve_hasher = MapToCurveBasedHasher::<
            Projective<WBCurve>,
            DefaultFieldHasher<Sha256, 128>,
            WBMap<WBCurve>,
        >::new(&[1])
        .unwrap();

        let hash_result = test_wb_to_curve_hasher.hash(b"if you stick a Babel fish in your ear you can instantly understand anything said to you in any form of language.").expect("fail to hash the string to curve");

        assert!(
            hash_result.x != WBCurve::BaseField::zero()
                && hash_result.y != WBCurve::BaseField::zero(),
            "we assume that not both a and b coefficienst are zero for the test curve"
        );

        assert!(
            hash_result.is_on_curve(),
            "hash results into a point off the curve"
        );
    }

    #[test]
    fn wb_hash_arbitrary_string_to_g1() {
        wb_hash_arbitrary_string_to_curve::<crate::g1::Config>();
    }

    #[test]
    fn wb_hash_arbitrary_string_to_g2() {
        wb_hash_arbitrary_string_to_curve::<crate::g2::Config>();
    }
}

#[test]
fn test_g1_subgroup_check() {
    // Cofactor (x - 1)^2 / 3 = 2^92 * 3 * 7^2 * 13^2 * 499^2.
    ark_algebra_test_templates::subgroup::test_subgroup_check::<crate::g1::Config>(
        &[2, 3, 7, 13, 499],
        8,
    );
}

#[test]
fn test_g2_subgroup_check() {
    ark_algebra_test_templates::subgroup::test_subgroup_check::<crate::g2::Config>(&[], 8);
}

#[test]
fn test_exp_by_x_and_twist_hooks() {
    use ark_ec::{bls12::Bls12Config, pairing::Pairing, short_weierstrass::SWCurveConfig};
    use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, UniformRand};
    let mut rng = ark_std::test_rng();
    for _ in 0..8 {
        let c = crate::Fq2::rand(&mut rng);
        assert_eq!(
            crate::Config::mul_by_3b_twist(c),
            <crate::g2::Config as SWCurveConfig>::COEFF_B * (c.double() + c)
        );
        let f = Bls12_377::pairing(G1Projective::rand(&mut rng), G2Projective::rand(&mut rng)).0;
        assert_eq!(crate::Config::exp_by_x(f), f.cyclotomic_exp(crate::Config::X));
    }
}

#[test]
fn test_fixed_q_miller_loop_dtwist() {
    use ark_ec::bls12::{Bls12, G2PreparedFixed};
    use ark_ec::{pairing::Pairing, CurveGroup};
    use ark_std::{test_rng, vec::Vec, UniformRand};
    let mut rng = test_rng();
    for n in 1..=4usize {
        let ps: Vec<crate::G1Affine> =
            (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<crate::G2Affine> =
            (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let fixed: Vec<G2PreparedFixed<crate::Config>> =
            qs.iter().map(|q| (*q).into()).collect();
        let ml_fixed = Bls12::<crate::Config>::multi_miller_loop_fixed(ps.iter().copied(), &fixed);
        let ml_std = Bls12_377::multi_miller_loop(ps.iter().copied(), qs.iter().copied());
        assert_eq!(
            Bls12_377::final_exponentiation(ml_fixed).unwrap(),
            Bls12_377::final_exponentiation(ml_std).unwrap(),
            "n = {n}"
        );
    }
}

#[test]
fn test_normalized_lines_mixed_miller_loop_dtwist() {
    use ark_ec::{bls12::G2Prepared, pairing::Pairing, CurveGroup};
    use ark_ff::One;
    use ark_std::{test_rng, vec::Vec, UniformRand};
    type Prep = G2Prepared<crate::Config>;
    let mut rng = test_rng();
    for n in [1usize, 3, 5] {
        let ps: Vec<crate::G1Affine> =
            (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<crate::G2Affine> =
            (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let expected = Bls12_377::multi_pairing(ps.iter().copied(), qs.iter().copied());
        for mask in [0usize, 0b01010, 0b10101, usize::MAX] {
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
            let ml = Bls12_377::multi_miller_loop(ps.iter().copied(), preps);
            assert_eq!(
                Bls12_377::final_exponentiation(ml).unwrap(),
                expected,
                "n = {n}, mask = {mask:b}"
            );
        }
    }
    let (p, q) = (
        G1Projective::rand(&mut rng).into_affine(),
        G2Projective::rand(&mut rng).into_affine(),
    );
    let mut normalized = Prep::from(q);
    normalized.normalize_lines();
    let neg = -normalized;
    assert!(neg.ell_coeffs.iter().all(|c| c.0.is_one()));
    assert_eq!(
        Bls12_377::final_exponentiation(Bls12_377::multi_miller_loop([p], [neg])).unwrap(),
        Bls12_377::pairing(-p, q)
    );
}
