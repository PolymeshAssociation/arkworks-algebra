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
        assert!(bytes.iter().all(|b| *b == 0));
        let q = Affine::deserialize_with_mode(&bytes[..], compress, Validate::Yes).unwrap();
        assert!(q.is_zero());

        // Infinity with the sign bit set, or uncompressed with a nonzero `y`, is not canonical.
        let mut signed = bytes.clone();
        *signed.last_mut().unwrap() |= 1 << 7;
        let mut nonzero_y = bytes.clone();
        nonzero_y[bytes.len() / 2] = 1;
        let mut bad = ark_std::vec![signed];
        if compress == Compress::No {
            bad.push(nonzero_y);
        }
        for b in bad {
            for validate in [Validate::Yes, Validate::No] {
                assert!(Affine::deserialize_with_mode(&b[..], compress, validate).is_err());
            }
        }
    }
}
