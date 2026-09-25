use ark_std::rand::Rng;
use crate::{Projective, Affine, Wei25519Config};
use ark_algebra_test_templates::*;

test_group!(g1; Projective; sw);

test_compact_serialization!(Affine; 32; 64);

test_h2c_swu!(hash_arbitrary_string_to_curve_swu; Projective; Wei25519Config);

/// `b` is a square, so `(0, \pm sqrt(b))` are curve points outside the prime-order subgroup.
/// They must fail to serialize instead of encoding as infinity, and infinity still round-trips.
#[test]
fn x_zero_point_does_not_serialize_as_infinity() {
    use crate::Fq;
    use ark_ec::{short_weierstrass::SWCurveConfig, AffineRepr};
    use ark_ff::{Field, Zero};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};

    let y = Wei25519Config::COEFF_B.sqrt().expect("b is a square");
    for y in [y, -y] {
        let p = Affine::new_unchecked(Fq::zero(), y);
        assert!(p.is_on_curve());
        assert!(!p.is_in_correct_subgroup_assuming_on_curve());
        for compress in [Compress::Yes, Compress::No] {
            let mut bytes = ark_std::vec::Vec::new();
            assert!(p.serialize_with_mode(&mut bytes, compress).is_err());
        }
    }
    for compress in [Compress::Yes, Compress::No] {
        let mut bytes = ark_std::vec::Vec::new();
        Affine::zero().serialize_with_mode(&mut bytes, compress).unwrap();
        let q = Affine::deserialize_with_mode(&bytes[..], compress, Validate::Yes).unwrap();
        assert!(q.is_zero());
    }
}
