use crate::{Projective, Affine};
use ark_algebra_test_templates::*;
use ark_ec::AffineRepr;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{test_rng, UniformRand, vec};

test_group!(g1; Projective; sw);

#[test]
fn compact_serialization() {
    let mut rng = test_rng();

    fn check(g: Affine) {
        assert_eq!(g.compressed_size(), 32);

        let mut compressed_bytes = vec![];
        g.serialize_compressed(&mut compressed_bytes).unwrap();
        assert_eq!(compressed_bytes.len(), 32);
        let g1_compressed: Affine = CanonicalDeserialize::deserialize_compressed(compressed_bytes.as_slice()).unwrap();
        assert_eq!(g, g1_compressed);

        let mut uncompressed_bytes = vec![];
        g.serialize_uncompressed(&mut uncompressed_bytes).unwrap();
        assert_eq!(uncompressed_bytes.len(), 64);
        let g1_uncompressed: Affine = CanonicalDeserialize::deserialize_uncompressed(uncompressed_bytes.as_slice()).unwrap();
        assert_eq!(g, g1_uncompressed);

        let g_wrong: Result<Affine, _> = CanonicalDeserialize::deserialize_uncompressed(compressed_bytes.as_slice());
        assert!(g_wrong.is_err());
    }

    let iterations = 100;
    for _ in 0..iterations {
        let g = Affine::rand(&mut rng);
        assert!(!g.is_zero());
        check(g);
    }

    let g_zero = Affine::zero();
    check(g_zero);
}