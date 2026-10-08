use ark_algebra_test_templates::*;
use ark_ff::fields::Field;

use crate::{Bn254, G1Projective, G2Projective};

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<Bn254>; msm);
test_pairing!(pairing; crate::Bn254);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);
// Vectors from gnark-crypto's `hashToG1Vector` and `hashToG2Vector`, hex zero-padded. Each
// file's `source` links the lines.
test_h2c!(g1_h2c; "./src/curves/tests"; "BN254G1"; crate::g1::Config; crate::Fq; crate::Fq; 1; ark_ec::hashing::curve_maps::svdw::SVDWMap<crate::g1::Config>; "SVDW");
test_h2c!(g2_h2c; "./src/curves/tests"; "BN254G2"; crate::g2::Config; crate::Fq2; crate::Fq; 2; ark_ec::hashing::curve_maps::svdw::SVDWMap<crate::g2::Config>; "SVDW");

/// Checks the SVDW constants, then maps `u = 0` and the `u` with `1 - C1 * u^2 = 0` or
/// `1 + C1 * u^2 = 0`, where `inv0` receives 0.
#[test]
fn test_svdw_parameters_and_inv0_inputs() {
    use ark_ec::hashing::{
        curve_maps::svdw::{SVDWConfig, SVDWMap},
        map_to_curve_hasher::MapToCurve,
    };
    use ark_ff::Zero;

    fn check<P: SVDWConfig>() {
        SVDWMap::<P>::check_parameters().unwrap();
        let c1_inv = P::C1.inverse().unwrap();
        let inv0_inputs: ark_std::vec::Vec<_> = [c1_inv.sqrt(), (-c1_inv).sqrt()]
            .into_iter()
            .flatten()
            .flat_map(|u| [u, -u])
            .collect();
        assert!(!inv0_inputs.is_empty());
        for u in inv0_inputs.into_iter().chain([P::BaseField::zero()]) {
            assert!(SVDWMap::<P>::map_to_curve(u).unwrap().is_on_curve());
        }
    }
    check::<crate::g1::Config>();
    check::<crate::g2::Config>();
}

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
        let ps: Vec<crate::G1Affine> = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let qs: Vec<crate::G2Affine> = (0..n)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect();
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
fn test_g2_gls4_digits_and_mul() {
    use ark_ec::{
        bn::{gls4_digits, BnConfig},
        scalar_mul::double_and_add,
        AffineRepr, CurveGroup,
    };
    use ark_ff::{BigInteger, PrimeField, UniformRand, Zero};
    use ark_std::{rand::Rng, test_rng, vec};
    let mut rng = test_rng();
    let params = crate::Config::GT_GLS.unwrap();
    // The digits satisfy `k == \sum_i k_i lambda^i (mod r)` with `lambda = p mod r`.
    let lambda = crate::Fr::from_le_bytes_mod_order(&crate::Fq::MODULUS.to_bytes_le());
    let mut scalars = vec![
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [u64::MAX >> 1, 0, 0, 0],
        [0, 1, 0, 0],
        [u64::MAX, u64::MAX, 0, 0],
        [0, 0, 1, 0],
        (-crate::Fr::from(1u64)).into_bigint().0,
    ];
    scalars.extend((0..8).map(|_| crate::Fr::rand(&mut rng).into_bigint().0));
    // `k < 2^63` rounds every `beta_j` to 0, since `k max_j |adj0[j]| / r < 1/2`, so it is the
    // single digit `k_0` and never uses `psi`.
    for k in [0u64, 1, u64::MAX >> 1, rng.gen::<u64>() >> 1] {
        let digits = gls4_digits::<crate::Config>(&[k], &params);
        assert_eq!(digits.map(|(_, d)| d), [k, 0, 0, 0], "k = {k}");
        assert!(!digits[0].0);
    }
    // Babai rounding keeps `|k_i| <= (1/2) sum_j |basis[j][i]|`, up to the `2^{-130}` error of the
    // shifted rounding.
    for k in scalars
        .iter()
        .copied()
        .chain((0..1000).map(|_| crate::Fr::rand(&mut rng).into_bigint().0))
    {
        for (i, (_, d)) in gls4_digits::<crate::Config>(&k, &params)
            .into_iter()
            .enumerate()
        {
            let bound: u128 = params.basis.iter().map(|row| row[i].unsigned_abs()).sum();
            assert!(u128::from(d) <= bound / 2 + 1, "k = {k:?}, digit {i} = {d}");
        }
    }
    // Scalars at or above `r` and wider than the field reduce mod `r` first.
    let r = crate::Fr::MODULUS.0;
    for k in [
        r.to_vec(),
        vec![r[0] + 5, r[1], r[2], r[3]],
        vec![1, 2, 3, 4, 5, 6],
        vec![7, 0, 0, 0, 0],
    ] {
        let mut limbs = [0u64; 4];
        let reduced = crate::Fr::from_le_bytes_mod_order(
            &k.iter()
                .flat_map(|l| l.to_le_bytes())
                .collect::<ark_std::vec::Vec<u8>>(),
        );
        limbs.copy_from_slice(&reduced.into_bigint().0);
        assert_eq!(
            gls4_digits::<crate::Config>(&k, &params),
            gls4_digits::<crate::Config>(&limbs, &params),
            "k = {k:?}"
        );
    }
    for k in &scalars {
        let digits = gls4_digits::<crate::Config>(k, &params);
        let sum = digits
            .iter()
            .rev()
            .fold(crate::Fr::zero(), |acc, &(neg, d)| {
                let d = crate::Fr::from(d);
                acc * lambda + if neg { -d } else { d }
            });
        assert_eq!(
            sum,
            crate::Fr::from_bigint(ark_ff::BigInt(*k)).unwrap(),
            "k = {k:?}"
        );
    }
    for _ in 0..4 {
        let p = G2Projective::rand(&mut rng);
        for k in &scalars {
            let expected = double_and_add(&p, k);
            let s = crate::Fr::from_bigint(ark_ff::BigInt(*k)).unwrap();
            assert_eq!(p * s, expected, "projective, k = {k:?}");
            assert_eq!(p.into_affine() * s, expected, "affine, k = {k:?}");
        }
    }
    // Off the subgroup `psi` is not `[6x^2]`. `mul_bigint` stays exact at every width, and the
    // GLS behind `*` is exact only while `k < 2^63` is a single digit.
    let off = loop {
        let x = crate::Fq2::rand(&mut rng);
        if let Some(p) = crate::G2Affine::get_point_from_x_unchecked(x, rng.gen()) {
            if !p.is_in_correct_subgroup_assuming_on_curve() {
                break p;
            }
        }
    };
    let exact = [u64::MAX >> 1];
    let wide = [0, 1 << 36];
    assert!(gls4_digits::<crate::Config>(&wide, &params)[1..]
        .iter()
        .any(|&(_, d)| d != 0));
    let x = u128::from(crate::Config::X[0]);
    let six_x_squared = 6 * x * x;
    let six_x_squared = [six_x_squared as u64, (six_x_squared >> 64) as u64];
    for k in [&exact[..], &wide[..], &six_x_squared[..]] {
        assert_eq!(
            off.mul_bigint(k),
            double_and_add(&off.into_group(), k),
            "k = {k:?}"
        );
    }
    assert_eq!(
        off * crate::Fr::from(exact[0]),
        double_and_add(&off.into_group(), exact)
    );
    assert_ne!(
        off * crate::Fr::from_bigint(ark_ff::BigInt([wide[0], wide[1], 0, 0])).unwrap(),
        double_and_add(&off.into_group(), wide)
    );
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
        let ps: Vec<_> = (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect();
        let qs: Vec<_> = (0..n)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect();
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
    for s in [
        crate::Fr::from(0u64),
        crate::Fr::from(1u64),
        crate::Fr::from(2u64),
        -crate::Fr::from(1u64),
    ] {
        let via_gls = <Bn254 as Pairing>::gt_exp(&gt, s.into_bigint().as_ref());
        let via_generic = gt.cyclotomic_exp(s.into_bigint().as_ref());
        assert_eq!(via_gls, via_generic, "s = {s}");
    }
}

/// `mul_bigint` is exact on a cyclotomic element outside GT, so `f^r == 1` and `f^p == f^{6x^2}`
/// reject it.
#[test]
fn test_gt_mul_bigint_exact_outside_gt() {
    use ark_ec::{
        bn::BnConfig,
        pairing::{Pairing, PairingOutput},
        PrimeGroup,
    };
    use ark_ff::{PrimeField, UniformRand};
    use ark_std::{test_rng, One};
    let mut rng = test_rng();
    let f = loop {
        let f = crate::Fq12::rand(&mut rng);
        let g = f.frobenius_map(6) * f.inverse().unwrap();
        let f = g.frobenius_map(2) * g;
        if !Bn254::is_in_gt(&f) {
            break f;
        }
    };
    let r = crate::Fr::MODULUS;
    let out = PairingOutput::<Bn254>(f).mul_bigint(r);
    assert_eq!(out.0, f.pow(r));
    assert!(!out.0.is_one());
    let x = crate::Fr::from(<crate::Config as BnConfig>::X[0]);
    let p_mod_r = (crate::Fr::from(6u64) * x * x).into_bigint();
    let out = PairingOutput::<Bn254>(f).mul_bigint(p_mod_r);
    assert_eq!(out.0, f.pow(p_mod_r));
    assert_ne!(out.0, f.frobenius_map(1));
}

#[test]
fn test_g2_subgroup_check() {
    // Cofactor = 10069 * 5864401 * (218-bit remainder).
    subgroup::test_subgroup_check::<crate::g2::Config>(&[10069, 5864401], 8);
}

#[test]
fn test_scalar_mul_matches_double_and_add() {
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g1::Config>(8);
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g2::Config>(8);
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
            assert_eq!(
                crate::Config::exp_by_x(h),
                h.cyclotomic_exp(crate::Config::X)
            );
        }
    }
}

/// Checked decoding of a prepared point rejects a line vector of the wrong length and a point at
/// infinity carrying lines.
#[test]
fn test_prepared_g2_line_count_is_validated() {
    use ark_ec::bn::G2Prepared;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
    use ark_std::{test_rng, UniformRand};
    let prepared = G2Prepared::<crate::Config>::from(crate::G2Affine::rand(&mut test_rng()));
    let encode = |p: &G2Prepared<crate::Config>| {
        let mut bytes = ark_std::vec::Vec::new();
        p.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        bytes
    };
    let decode = |bytes: &[u8]| {
        G2Prepared::<crate::Config>::deserialize_with_mode(bytes, Compress::No, Validate::Yes)
    };
    assert_eq!(decode(&encode(&prepared)).unwrap(), prepared);
    let mut short = prepared.clone();
    short.ell_coeffs.pop();
    assert!(decode(&encode(&short)).is_err());
    let mut infinite = prepared.clone();
    infinite.infinity = true;
    assert!(decode(&encode(&infinite)).is_err());
}

/// `cyclotomic_exp` against `pow` on GT for exponents of every length up to 130 bits, which
/// crosses each wNAF width, and for full-width and wider ones.
#[test]
fn test_cyclotomic_exp_matches_pow_at_every_width() {
    use ark_ec::{pairing::PairingOutput, PrimeGroup};
    use ark_ff::{CyclotomicMultSubgroup, Field};
    use ark_std::{test_rng, vec, vec::Vec, UniformRand};
    let rng = &mut test_rng();
    let g = PairingOutput::<crate::Bn254>::generator().0;
    let mut exps: Vec<Vec<u64>> = (1..=130usize)
        .map(|bits| {
            let mut e = vec![0u64; bits.div_ceil(64)];
            for (i, limb) in e.iter_mut().enumerate() {
                *limb = u64::rand(rng);
                if i == (bits - 1) / 64 && bits % 64 != 0 {
                    *limb &= (1u64 << (bits % 64)) - 1;
                }
            }
            e[(bits - 1) / 64] |= 1u64 << ((bits - 1) % 64);
            e
        })
        .collect();
    exps.extend([
        vec![u64::MAX; 4],
        vec![u64::MAX; 6],
        vec![2],
        vec![11],
        vec![103],
    ]);
    for e in &exps {
        assert_eq!(g.cyclotomic_exp(e), g.pow(e), "e = {e:?}");
    }
}
