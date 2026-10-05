use crate::*;
use ark_algebra_test_templates::*;
use ark_ff::Field;

test_group!(g1; G1Projective; sw);
test_group!(g2; G2Projective; sw);
test_group!(pairing_output; ark_ec::pairing::PairingOutput<BW6_761>; msm);
test_pairing!(pairing; crate::BW6_761);
test_group!(g1_glv; G1Projective; glv);
test_group!(g2_glv; G2Projective; glv);

#[test]
fn test_g1_subgroup_check() {
    // Cofactor = 2^2 * 127 * 3841927 * (large).
    subgroup::test_subgroup_check::<crate::g1::Config>(&[2, 127, 3841927], 6);
}

#[test]
fn test_g2_subgroup_check() {
    // Cofactor = 3 * 13 * (large); the order-3 points are (0, 2) and (0, -2).
    subgroup::test_subgroup_check::<crate::g2::Config>(&[3, 13], 6);
}

/// Checked decoding of a prepared point rejects either line vector at the wrong length and a point
/// at infinity carrying lines.
#[test]
fn test_prepared_g2_line_count_is_validated() {
    use ark_ec::bw6::G2Prepared;
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
    use ark_std::{test_rng, UniformRand};
    let prepared = G2Prepared::<crate::Config>::from(crate::G2Affine::rand(&mut test_rng()));
    let encode = |p: &G2Prepared<crate::Config>| {
        let mut bytes = ark_std::vec::Vec::new();
        p.serialize_with_mode(&mut bytes, Compress::No).unwrap();
        bytes
    };
    let decode = |bytes: &[u8]| G2Prepared::<crate::Config>::deserialize_with_mode(bytes, Compress::No, Validate::Yes);
    assert_eq!(decode(&encode(&prepared)).unwrap(), prepared);
    for which in 0..2 {
        let mut short = prepared.clone();
        if which == 0 {
            short.ell_coeffs_1.pop();
        } else {
            short.ell_coeffs_2.pop();
        }
        assert!(decode(&encode(&short)).is_err());
    }
    let mut infinite = prepared.clone();
    infinite.infinity = true;
    assert!(decode(&encode(&infinite)).is_err());
}

/// GLV `mul_bigint` against double-and-add on subgroup and random curve points, exact off the
/// subgroup below `2^128`.
#[test]
fn test_scalar_mul_matches_double_and_add() {
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g1::Config>(8);
    subgroup::test_scalar_mul_matches_double_and_add::<crate::g2::Config>(8);
}
