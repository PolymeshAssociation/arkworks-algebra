use ark_std::rand::Rng;
use crate::{Projective, Affine, SeleneConfig};
use ark_algebra_test_templates::*;

test_group!(g1; Projective; sw);

test_compact_serialization!(Affine; 32; 64);

test_h2c_swu!(hash_arbitrary_string_to_curve_swu; Projective; SeleneConfig);

/// `sqrt_or_zeta_sqrt(gx1)` returns `y` with `y^2 = gx1` when `gx1` is a square, else
/// `y^2 = ZETA * gx1`. Covers `0`, `±1`, `±4`, `±ZETA`, `±1/ZETA` and random `gx1`.
#[test]
fn test_sqrt_or_zeta_sqrt() {
    use crate::Fq;
    use ark_ec::hashing::curve_maps::swu::SWUConfig;
    use ark_ff::Field;
    use ark_std::UniformRand;

    fn check(gx1: Fq) {
        let (is_square, y) = SeleneConfig::sqrt_or_zeta_sqrt(gx1);
        assert_eq!(is_square, gx1.sqrt().is_some(), "gx1 = {gx1}");
        let target = if is_square {
            gx1
        } else {
            SeleneConfig::ZETA * gx1
        };
        assert_eq!(y.square(), target, "gx1 = {gx1}");
    }

    let zeta = SeleneConfig::ZETA;
    for a in [
        Fq::from(0u64),
        Fq::from(1u64),
        Fq::from(4u64),
        zeta,
        zeta.inverse().unwrap(),
    ] {
        check(a);
        check(-a);
    }
    let mut rng = ark_std::test_rng();
    for _ in 0..1000 {
        check(Fq::rand(&mut rng));
    }
}
