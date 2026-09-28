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
    // Cofactor = 2^2 * 127 * (large).
    subgroup::test_subgroup_check::<crate::g1::Config>(&[2, 127], 6);
}

#[test]
fn test_g2_subgroup_check() {
    // Cofactor = 3 * 13 * (large); the order-3 points are (0, 2) and (0, -2).
    subgroup::test_subgroup_check::<crate::g2::Config>(&[3, 13], 6);
}
