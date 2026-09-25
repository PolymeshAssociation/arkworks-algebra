use crate::{Projective, Affine};
use ark_algebra_test_templates::*;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

test_group!(g1; Projective; sw);
test_group!(g1_glv; Projective; glv);
test_compact_serialization!(Affine; 32; 64);
test_h2c!(pallas_h2c; "./src/curves/test_vectors"; "Pallas"; crate::PallasConfig; crate::Fq; crate::Fq; 1);

/// `ZETA_TRACE_POWER` is `ZETA^((T - 1) / 2)`, and the one-exponentiation `sqrt_or_zeta_sqrt`
/// returns exactly what the default two-root computation returns.
#[test]
fn swu_sqrt_or_zeta_sqrt_matches_two_roots() {
    use crate::{curves::swu_iso::{SwuIsoConfig, ZETA_TRACE_POWER}, Fq};
    use ark_ec::hashing::curve_maps::swu::SWUConfig;
    use ark_ff::{Field, PrimeField};
    use ark_std::{test_rng, UniformRand};

    let zeta = <SwuIsoConfig as SWUConfig>::ZETA;
    assert_eq!(ZETA_TRACE_POWER, zeta.pow(Fq::TRACE_MINUS_ONE_DIV_TWO));
    let mut rng = test_rng();
    let inputs = [Fq::from(0u64), Fq::from(1u64), -Fq::from(1u64), zeta]
        .into_iter()
        .chain((0..2000).map(|_| Fq::rand(&mut rng)));
    for u in inputs {
        let expected = match u.sqrt() {
            Some(y) => (true, y),
            None => (false, (zeta * u).sqrt().unwrap()),
        };
        assert_eq!(SwuIsoConfig::sqrt_or_zeta_sqrt(u), expected, "u = {u}");
    }
}
