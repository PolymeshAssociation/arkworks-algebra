use ark_algebra_test_templates::*;
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{fields::Field, One, UniformRand, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use ark_std::{rand::Rng, test_rng, vec};

use crate::{Bls12_381, Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<Bls12_381>; msm);
test_pairing!(pairing; crate::Bls12_381);
test_h2c!(g1_h2c; "./src/curves/tests"; "BLS12381G1"; crate::g1::Config; crate::Fq; crate::Fq; 1);
test_h2c!(g2_hc2; "./src/curves/tests"; "BLS12381G2"; crate::g2::Config; crate::Fq2; crate::Fq; 2);

#[test]
fn test_g1_endomorphism_beta() {
    assert!(crate::g1::BETA.pow([3u64]).is_one());
}

#[test]
fn test_g1_subgroup_membership_via_endomorphism() {
    let mut rng = test_rng();
    let generator = G1Projective::rand(&mut rng).into_affine();
    assert!(generator.is_in_correct_subgroup_assuming_on_curve());
}

#[test]
fn test_g1_subgroup_non_membership_via_endomorphism() {
    let mut rng = test_rng();
    loop {
        let x = Fq::rand(&mut rng);
        let greatest = rng.gen();

        if let Some(p) = G1Affine::get_point_from_x_unchecked(x, greatest) {
            if !p.mul_bigint(Fr::characteristic()).is_zero() {
                assert!(!p.is_in_correct_subgroup_assuming_on_curve());
                return;
            }
        }
    }
}

#[test]
fn test_g2_subgroup_membership_via_endomorphism() {
    let mut rng = test_rng();
    let generator = G2Projective::rand(&mut rng).into_affine();
    assert!(generator.is_in_correct_subgroup_assuming_on_curve());
}

#[test]
fn test_g2_subgroup_non_membership_via_endomorphism() {
    let mut rng = test_rng();
    loop {
        let x = Fq2::rand(&mut rng);
        let greatest = rng.gen();

        if let Some(p) = G2Affine::get_point_from_x_unchecked(x, greatest) {
            if !p.mul_bigint(Fr::characteristic()).is_zero() {
                assert!(!p.is_in_correct_subgroup_assuming_on_curve());
                return;
            }
        }
    }
}

#[test]
fn test_g1_uncompressed_rejects_off_curve_point() {
    // (u^2 x, u^3 y) lies on y^2 = x^3 + u^6 b, not on the BLS12-381 curve,
    // but it still passes the endomorphism-based subgroup check.
    let g = G1Affine::generator();
    let u = Fq::from(2u64);
    let p = G1Affine::new_unchecked(g.x * u.square(), g.y * u.square() * u);
    assert!(!p.is_on_curve());
    assert!(p.is_in_correct_subgroup_assuming_on_curve());

    let mut bytes = vec![];
    p.serialize_uncompressed(&mut bytes).unwrap();
    assert!(G1Affine::deserialize_uncompressed(&bytes[..]).is_err());
}

#[test]
fn test_g2_uncompressed_rejects_off_curve_point() {
    let g = G2Affine::generator();
    let u = Fq2::from(2u64);
    let p = G2Affine::new_unchecked(g.x * u.square(), g.y * u.square() * u);
    assert!(!p.is_on_curve());
    assert!(p.is_in_correct_subgroup_assuming_on_curve());

    let mut bytes = vec![];
    p.serialize_uncompressed(&mut bytes).unwrap();
    assert!(G2Affine::deserialize_uncompressed(&bytes[..]).is_err());
}

#[test]
fn test_g2_gls4_digits_and_mul() {
    use ark_ec::{
        bls12::{gls4_digits, Bls12Config},
        scalar_mul::double_and_add,
    };
    use ark_ff::PrimeField;
    let mut rng = test_rng();
    // The digits satisfy `k = \sum_i k_i x^i` with `x = -|X|`.
    let abs_x = crate::Config::X[0];
    let x = -Fr::from(abs_x);
    let abs_x2 = u128::from(abs_x) * u128::from(abs_x);
    let mut scalars = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [abs_x - 1, 0, 0, 0],
        [abs_x, 0, 0, 0],
        [abs_x + 1, 0, 0, 0],
        [abs_x2 as u64, (abs_x2 >> 64) as u64, 0, 0],
        [(abs_x2 - 1) as u64, ((abs_x2 - 1) >> 64) as u64, 0, 0],
        (Fr::from(abs_x2) * Fr::from(abs_x)).into_bigint().0,
        [u64::MAX >> 1, 0, 0, 0],
        [0, 1, 0, 0],
        [u64::MAX, u64::MAX, 0, 0],
        [0, 0, 1, 0],
        (-Fr::one()).into_bigint().0,
    ];
    scalars.extend((0..8).map(|_| Fr::rand(&mut rng).into_bigint().0));
    for k in &scalars {
        let digits = gls4_digits::<crate::Config>(k).unwrap();
        let sum = digits.iter().rev().fold(Fr::zero(), |acc, &(neg, d)| {
            let d = Fr::from(d);
            acc * x + if neg { -d } else { d }
        });
        assert_eq!(
            sum,
            Fr::from_bigint(ark_ff::BigInt(*k)).unwrap(),
            "k = {k:?}"
        );
        assert!(digits.iter().all(|&(_, d)| d < abs_x), "k = {k:?}");
    }
    // `k < 2^63` is below `|x|`, so it is the single digit `k_0` and never uses `psi`.
    for k in [0u64, 1, u64::MAX >> 1, rng.gen::<u64>() >> 1] {
        let digits = gls4_digits::<crate::Config>(&[k]).unwrap();
        assert_eq!(digits.map(|(_, d)| d), [k, 0, 0, 0], "k = {k}");
        assert!(!digits[0].0);
    }
    for _ in 0..4 {
        let p = G2Projective::rand(&mut rng);
        for k in &scalars {
            let expected = double_and_add(&p, k);
            let s = Fr::from_bigint(ark_ff::BigInt(*k)).unwrap();
            assert_eq!(p * s, expected, "projective, k = {k:?}");
            assert_eq!(p.into_affine() * s, expected, "affine, k = {k:?}");
        }
    }
    // Off the subgroup `psi` is not `[x]`. `mul_bigint` stays exact at every width, and the GLS
    // behind `*` is exact only while `k < 2^63` is a single digit.
    let off = loop {
        if let Some(p) = G2Affine::get_point_from_x_unchecked(Fq2::rand(&mut rng), rng.gen()) {
            if !p.is_in_correct_subgroup_assuming_on_curve() {
                break p;
            }
        }
    };
    let exact = [u64::MAX >> 1];
    let wide = [0, 1 << 36];
    for k in [&exact[..], &wide[..], crate::Config::X] {
        assert_eq!(
            off.mul_bigint(k),
            double_and_add(&off.into_group(), k),
            "k = {k:?}"
        );
    }
    assert_eq!(
        off * Fr::from(exact[0]),
        double_and_add(&off.into_group(), exact)
    );
    assert_ne!(
        off * Fr::from_bigint(ark_ff::BigInt([wide[0], wide[1], 0, 0])).unwrap(),
        double_and_add(&off.into_group(), wide)
    );
}

#[test]
fn test_scalar_mul_matches_double_and_add() {
    use ark_ff::PrimeField;
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g1::Config>(8, 128);
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g2::Config>(
        8,
        Fr::MODULUS_BIT_SIZE,
    );
}

// Test vectors and macro adapted from https://github.com/zkcrypto/bls12_381/blob/e224ad4ea1babfc582ccd751c2bf128611d10936/src/tests/mod.rs
macro_rules! test_vectors {
    ($projective:ident, $affine:ident, $compress:expr, $expected:ident) => {
        let mut e = $projective::zero();

        let mut v = vec![];
        {
            let mut expected = $expected;
            for _ in 0..1000 {
                let e_affine = $affine::from(e);
                let mut serialized = vec![0u8; e.serialized_size($compress)];
                e_affine
                    .serialize_with_mode(serialized.as_mut_slice(), $compress)
                    .unwrap();
                v.extend_from_slice(&serialized[..]);

                let mut decoded = serialized;
                let len_of_encoding = decoded.len();
                (&mut decoded[..]).copy_from_slice(&expected[0..len_of_encoding]);
                expected = &expected[len_of_encoding..];
                let decoded =
                    $affine::deserialize_with_mode(&decoded[..], $compress, Validate::Yes).unwrap();
                assert_eq!(e_affine, decoded);

                e += &$projective::generator();
            }
        }

        assert_eq!(&v[..], $expected);
    };
}

#[test]
fn g1_compressed_valid_test_vectors() {
    let bytes: &'static [u8] = include_bytes!("g1_compressed_valid_test_vectors.dat");
    test_vectors!(G1Projective, G1Affine, Compress::Yes, bytes);
}

#[test]
fn g1_uncompressed_valid_test_vectors() {
    let bytes: &'static [u8] = include_bytes!("g1_uncompressed_valid_test_vectors.dat");
    test_vectors!(G1Projective, G1Affine, Compress::No, bytes);
}

#[test]
fn g2_compressed_valid_test_vectors() {
    let bytes: &'static [u8] = include_bytes!("g2_compressed_valid_test_vectors.dat");
    test_vectors!(G2Projective, G2Affine, Compress::Yes, bytes);
}

#[test]
fn g2_uncompressed_valid_test_vectors() {
    let bytes: &'static [u8] = include_bytes!("g2_uncompressed_valid_test_vectors.dat");
    test_vectors!(G2Projective, G2Affine, Compress::No, bytes);
}

#[test]
fn test_g2_prepared_neg() {
    use ark_ec::pairing::Pairing;
    let mut rng = test_rng();
    for _ in 0..10 {
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G2Projective::rand(&mut rng).into_affine();
        let q_prep: <Bls12_381 as Pairing>::G2Prepared = q.into();
        let neg_prep = -q_prep.clone();
        // e(P, -Q) == e(P, neg-of-prepared-Q)
        assert_eq!(
            Bls12_381::pairing(p, -q),
            Bls12_381::pairing(p, neg_prep.clone())
        );
        // e(P, Q) * e(P, -Q) == 1
        let sum = Bls12_381::pairing(p, q_prep) + Bls12_381::pairing(p, neg_prep);
        assert!(sum.is_zero());
    }
}

#[test]
fn test_exp_by_x_chain_matches_generic() {
    use ark_ec::bls12::Bls12Config;
    use ark_ec::pairing::{MillerLoopOutput, Pairing};
    use ark_ff::CyclotomicMultSubgroup;
    let mut rng = test_rng();
    for _ in 0..10 {
        let f = crate::Fq12::rand(&mut rng);
        // Map into the cyclotomic subgroup.
        if let Some(out) = Bls12_381::final_exponentiation(MillerLoopOutput(f)) {
            let cyc = out.0;
            let chain = <crate::Config as Bls12Config>::exp_by_x(cyc);
            let mut generic = cyc.cyclotomic_exp(crate::Config::X);
            generic.cyclotomic_inverse_in_place(); // X is negative
            assert_eq!(chain, generic);
        }
    }
}

#[test]
fn test_gt_membership_fast_matches_naive() {
    use ark_ec::pairing::{Pairing, PairingOutput};
    let mut rng = test_rng();
    for _ in 0..20 {
        // (1) A genuine GT element: both tests accept.
        let p = G1Projective::rand(&mut rng).into_affine();
        let q = G2Projective::rand(&mut rng).into_affine();
        let gt = Bls12_381::pairing(p, q);
        assert!(gt.is_in_group());
        assert!(gt.is_in_group_naive());

        // (2) A random Fq12 is (almost surely) not even cyclotomic: both reject.
        let f = crate::Fq12::rand(&mut rng);
        let rand_out = PairingOutput::<Bls12_381>(f);
        assert_eq!(rand_out.is_in_group(), rand_out.is_in_group_naive());
        assert!(!rand_out.is_in_group());

        // (3) A cyclotomic element that is (almost surely) not in GT: the easy
        // part of the final exponentiation lands in the cyclotomic subgroup, so
        // this exercises the order-r stage. A wrong exponent there would make the
        // fast test accept while the naive test rejects.
        let mut r = f;
        r.frobenius_map_in_place(6); // f^(p^6)
        r *= f.inverse().unwrap(); // f^(p^6 - 1)
        let mut r_p2 = r;
        r_p2.frobenius_map_in_place(2);
        r *= r_p2; // f^((p^6 - 1)(p^2 + 1)) in the cyclotomic subgroup
        let cyc = PairingOutput::<Bls12_381>(r);
        assert_eq!(cyc.is_in_group(), cyc.is_in_group_naive());
        assert!(!cyc.is_in_group());
    }
}

#[test]
fn test_multi_pairing_various_n() {
    use ark_ec::pairing::{Pairing, PairingOutput};
    use ark_std::vec::Vec;
    let mut rng = test_rng();
    for n in 1..=6usize {
        let ps: Vec<_> = (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<_> = (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let multi = Bls12_381::multi_pairing(ps.iter().copied(), qs.iter().copied());
        let prod = ps
            .iter()
            .zip(&qs)
            .map(|(p, q)| Bls12_381::pairing(*p, *q))
            .fold(PairingOutput::zero(), |acc, x| acc + x);
        assert_eq!(multi, prod, "n = {n}");
    }
}

#[test]
fn test_gt_exp_matches_generic() {
    use ark_ec::pairing::Pairing;
    use ark_ff::{CyclotomicMultSubgroup, PrimeField};
    let mut rng = test_rng();
    let p = G1Projective::rand(&mut rng).into_affine();
    let q = G2Projective::rand(&mut rng).into_affine();
    let gt = Bls12_381::pairing(p, q).0;
    for _ in 0..10 {
        let s = Fr::rand(&mut rng);
        let via_gls = <Bls12_381 as Pairing>::gt_exp(&gt, s.into_bigint().as_ref());
        let via_generic = gt.cyclotomic_exp(s.into_bigint().as_ref());
        assert_eq!(via_gls, via_generic);
    }
    for s in [Fr::from(0u64), Fr::from(1u64), Fr::from(2u64), -Fr::from(1u64)] {
        let via_gls = <Bls12_381 as Pairing>::gt_exp(&gt, s.into_bigint().as_ref());
        let via_generic = gt.cyclotomic_exp(s.into_bigint().as_ref());
        assert_eq!(via_gls, via_generic, "s = {s}");
    }
}

#[test]
fn test_fixed_q_miller_loop() {
    use ark_ec::bls12::{Bls12, G2PreparedFixed};
    use ark_ec::pairing::Pairing;
    use ark_std::vec::Vec;
    let mut rng = test_rng();
    for n in 1..=4usize {
        let ps: Vec<G1Affine> =
            (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<G2Affine> =
            (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let fixed: Vec<G2PreparedFixed<crate::Config>> =
            qs.iter().map(|q| (*q).into()).collect();
        let ml_fixed = Bls12::<crate::Config>::multi_miller_loop_fixed(ps.iter().copied(), &fixed);
        let ml_std = Bls12_381::multi_miller_loop(ps.iter().copied(), qs.iter().copied());
        assert_eq!(
            Bls12_381::final_exponentiation(ml_fixed).unwrap(),
            Bls12_381::final_exponentiation(ml_std).unwrap(),
            "n = {n}"
        );
    }
}

#[test]
#[should_panic]
fn test_fixed_q_miller_loop_rejects_length_mismatch() {
    use ark_ec::bls12::{Bls12, G2PreparedFixed};
    use ark_std::vec::Vec;
    let mut rng = test_rng();
    let ps: Vec<G1Affine> = (0..3).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
    let fixed: Vec<G2PreparedFixed<crate::Config>> =
        (0..2).map(|_| G2Projective::rand(&mut rng).into_affine().into()).collect();
    let _ = Bls12::<crate::Config>::multi_miller_loop_fixed(ps.iter().copied(), &fixed);
}

#[test]
fn test_normalized_lines_mixed_miller_loop() {
    use ark_ec::{bls12::G2Prepared, pairing::Pairing};
    use ark_std::vec::Vec;
    type Prep = G2Prepared<crate::Config>;
    let mut rng = test_rng();
    // Up to 9 pairs, so parallel builds split the loop into several chunks.
    for n in [1usize, 2, 3, 5, 9] {
        let ps: Vec<G1Affine> =
            (0..n).map(|_| G1Projective::rand(&mut rng).into_affine()).collect();
        let qs: Vec<G2Affine> =
            (0..n).map(|_| G2Projective::rand(&mut rng).into_affine()).collect();
        let expected = Bls12_381::multi_pairing(ps.iter().copied(), qs.iter().copied());
        for mask in [0usize, 1, 0b01010, 0b10101, usize::MAX] {
            let preps: Vec<Prep> = qs
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let mut prep = Prep::from(*q);
                    if (mask >> i) & 1 == 1 {
                        prep.normalize_lines();
                        assert!(prep.ell_coeffs.iter().all(|c| c.2.is_one()));
                    }
                    prep
                })
                .collect();
            let ml = Bls12_381::multi_miller_loop(ps.iter().copied(), preps);
            assert_eq!(
                Bls12_381::final_exponentiation(ml).unwrap(),
                expected,
                "n = {n}, mask = {mask:b}"
            );
        }
    }
}

#[test]
fn test_normalized_lines_negation_and_raw_fallbacks() {
    use ark_ec::{bls12::G2Prepared, pairing::Pairing};
    type Prep = G2Prepared<crate::Config>;
    let mut rng = test_rng();
    let fe = |p: G1Affine, q: Prep| {
        Bls12_381::final_exponentiation(Bls12_381::multi_miller_loop([p], [q])).unwrap()
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
    assert!(neg.ell_coeffs.iter().all(|c| c.2.is_one()));
    assert_eq!(fe(p, neg), Bls12_381::pairing(-p, q));

    // A line with a zero `P.y` coefficient stays raw.
    let mut raw = Prep::from(q);
    raw.ell_coeffs[17].2 = Fq2::zero();
    let mut partly = raw.clone();
    partly.normalize_lines();
    assert_eq!(partly.ell_coeffs[17], raw.ell_coeffs[17]);
    assert!(partly.ell_coeffs[16].2.is_one());
    assert_eq!(fe(p, partly), fe(p, raw));

    // `P.y = 0` has no `1/P.y`, so its pair runs on raw lines.
    let p0 = G1Affine::new_unchecked(Fq::rand(&mut rng), Fq::zero());
    assert_eq!(fe(p0, normalized), fe(p0, Prep::from(q)));
}

#[test]
fn test_karabina_compressed_squaring() {
    use ark_ec::{bls12::Bls12Config, pairing::Pairing};
    use ark_ff::{fields::fp12_2over3over2::CompressedCyclotomic, CyclotomicMultSubgroup, One};
    let mut rng = test_rng();
    let f = crate::Fq12::rand(&mut rng);
    // f^((p^6 - 1)(p^2 + 1)) is cyclotomic but (almost surely) not in GT.
    let mut cyc = f;
    cyc.frobenius_map_in_place(6);
    cyc *= f.inverse().unwrap();
    let mut cyc_p2 = cyc;
    cyc_p2.frobenius_map_in_place(2);
    cyc *= cyc_p2;
    let gt = Bls12_381::pairing(G1Projective::rand(&mut rng), G2Projective::rand(&mut rng)).0;
    for g in [gt, gt.square(), cyc] {
        let mut full = g;
        let mut c = g.compress_cyclotomic();
        let mut checkpoint = None;
        for i in 1..=47 {
            full.cyclotomic_square_in_place();
            c.square_in_place();
            assert_eq!(c, full.compress_cyclotomic());
            if i == 15 {
                checkpoint = Some((c, full));
            }
        }
        let (c15, full15) = checkpoint.unwrap();
        assert_eq!(CompressedCyclotomic::decompress_pair(&c15, &c), Some((full15, full)));
        let mut expected = g.cyclotomic_exp(crate::Config::X);
        expected.cyclotomic_inverse_in_place();
        assert_eq!(<crate::Config as Bls12Config>::exp_by_x(g), expected);
    }
    // g3 = 0 takes the uncompressed fallback.
    let one = crate::Fq12::one();
    assert_eq!(CompressedCyclotomic::decompress_pair(&one.compress_cyclotomic(), &one.compress_cyclotomic()), None);
    assert_eq!(<crate::Config as Bls12Config>::exp_by_x(one), one);
}

#[test]
fn test_g1_subgroup_check() {
    // h1 = (x - 1)^2 / 3 = 3 * 11^2 * 10177^2 * 859267^2 * 52437899^2.
    ark_algebra_test_templates::subgroup::test_subgroup_check::<crate::g1::Config>(
        &[3, 11, 10177, 859267, 52437899],
        4,
    );
}

#[test]
fn test_g2_subgroup_check() {
    // Every prime of h2 below 2^64; the remaining factor is a 448-bit prime.
    ark_algebra_test_templates::subgroup::test_subgroup_check::<crate::g2::Config>(
        &[13, 23, 2713, 11953, 262069],
        4,
    );
}

/// `(0, \pm 2)` are the order-3 points of G1's curve, fixed by the GLV endomorphism. The
/// subgroup check, `S + (0, 2)`, and checked compressed decoding all reject them.
#[test]
fn test_g1_x_zero_torsion_rejected() {
    let mut rng = test_rng();
    let two = Fq::from(2u64);
    for t in [G1Affine::new_unchecked(Fq::zero(), two), G1Affine::new_unchecked(Fq::zero(), -two)] {
        assert!(t.is_on_curve());
        assert!(!t.is_in_correct_subgroup_assuming_on_curve());
        assert!(t.into_group().mul_bigint(Fr::characteristic()) != G1Projective::zero());
        let s = G1Projective::rand(&mut rng);
        assert!(!(s + t).into_affine().is_in_correct_subgroup_assuming_on_curve());

        let mut bytes = vec![];
        t.serialize_with_mode(&mut bytes, Compress::Yes).unwrap();
        assert!(G1Affine::deserialize_with_mode(&bytes[..], Compress::Yes, Validate::Yes).is_err());
    }
}

/// Checked decoding of prepared points rejects a line vector of the wrong length and a point at
/// infinity carrying lines, which the Miller loop would otherwise run past or misread.
#[test]
fn test_prepared_g2_line_count_is_validated() {
    use ark_ec::bls12::{G2Prepared, G2PreparedFixed};
    let q = G2Affine::rand(&mut test_rng());
    let prepared = G2Prepared::<crate::Config>::from(q);
    let fixed = G2PreparedFixed::<crate::Config>::from(q);
    let decode = |bytes: &[u8]| G2Prepared::<crate::Config>::deserialize_with_mode(bytes, Compress::No, Validate::Yes);
    let decode_fixed = |bytes: &[u8]| G2PreparedFixed::<crate::Config>::deserialize_with_mode(bytes, Compress::No, Validate::Yes);
    fn encode<T: ark_serialize::CanonicalSerialize>(p: &T) -> ark_std::vec::Vec<u8> {
        let mut bytes = vec![];
        p.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        bytes
    }
    assert_eq!(decode(&encode(&prepared)).unwrap(), prepared);
    assert_eq!(decode_fixed(&encode(&fixed)).unwrap(), fixed);

    let mut short = prepared.clone();
    short.ell_coeffs.pop();
    assert!(decode(&encode(&short)).is_err());
    assert!(G2Prepared::<crate::Config>::deserialize_with_mode(&encode(&short)[..], Compress::No, Validate::No).is_ok());
    let mut infinite = prepared.clone();
    infinite.infinity = true;
    assert!(decode(&encode(&infinite)).is_err());

    let mut short_fixed = fixed.clone();
    short_fixed.lines.pop();
    assert!(decode_fixed(&encode(&short_fixed)).is_err());
}
