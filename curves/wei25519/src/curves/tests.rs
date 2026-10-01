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

/// The Curve25519 2-torsion point `(A/3, 0)` has `2^{cj} T = O` for every window `j > 0`, so the
/// fixed-base table holds identity multiples. They must contribute nothing, alone, negated and
/// added to subgroup points, serial and segmented.
#[test]
fn fixed_base_msm_with_two_torsion_base() {
    use crate::{Fq, Fr};
    use ark_ec::{
        scalar_mul::{double_and_add_affine, fixed_base::FixedBaseMSM},
        AffineRepr, CurveGroup,
    };
    use ark_ff::{Field, PrimeField, UniformRand, Zero};
    use ark_std::test_rng;

    let rng = &mut test_rng();
    let t = Affine::new_unchecked(
        Fq::from(486662u64) * Fq::from(3u64).inverse().unwrap(),
        Fq::zero(),
    );
    assert!(t.is_on_curve());
    assert!(t.mul_bigint([2u64]).is_zero());
    for n in [3usize, 300] {
        let mut bases: ark_std::vec::Vec<Affine> = (0..n)
            .map(|_| Projective::rand(rng).into_affine())
            .collect();
        bases[0] = t;
        bases[1] = -t;
        bases[2] = (bases[2] + t).into_affine();
        let scalars: ark_std::vec::Vec<_> = (0..n)
            .map(|i| if i < 2 { Fr::from(17u64) } else { Fr::rand(rng) }.into_bigint())
            .collect();
        let expected: Projective = bases
            .iter()
            .zip(&scalars)
            .map(|(b, s)| double_and_add_affine(b, s))
            .sum();
        for c in [2usize, 4, 8, 13] {
            let pc = FixedBaseMSM::<Wei25519Config>::new_given_window_size(&bases, c);
            assert_eq!(pc.msm_bigint(&scalars), expected, "n={n} c={c}");
        }
    }
}
