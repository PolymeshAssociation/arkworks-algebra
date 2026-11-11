//! Batch-affine, GLV and fixed-base MSMs, which weigh their buckets in affine, against a naive sum.

use ark_ec::{
    CurveGroup,
    scalar_mul::{
        fixed_base::FixedBaseMSM,
        sw_pippenger::{msm_batch_affine_bigint, msm_batch_affine_glv_bigint},
    },
};
use ark_ff::{PrimeField, UniformRand};
use ark_pallas::{Affine as GAffine, Fr, PallasConfig, Projective as G};
use ark_std::{test_rng, vec::Vec};

type BigInt = <Fr as PrimeField>::BigInt;

#[test]
fn msm_with_affine_weighting_matches() {
    let rng = &mut test_rng();
    for n in [256usize, 1000, 4096] {
        let bases: Vec<GAffine> =
            G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let bigints: Vec<BigInt> = scalars.iter().map(|s| s.into_bigint()).collect();
        let expected: G = bases.iter().zip(&scalars).map(|(b, s)| *b * s).sum();
        assert_eq!(
            msm_batch_affine_bigint::<PallasConfig>(&bases, &bigints),
            expected
        );
        assert_eq!(
            msm_batch_affine_glv_bigint::<PallasConfig>(&bases, &bigints).unwrap(),
            expected
        );
    }
}

#[test]
fn fixed_base_with_affine_weighting_matches() {
    let rng = &mut test_rng();
    for (n, cs) in [
        (37usize, &[4usize, 8][..]),
        (600, &[6, 10, 13][..]),
        (4100, &[8, 12, 16][..]),
    ] {
        let mut bases: Vec<GAffine> =
            G::normalize_batch(&(0..n).map(|_| G::rand(rng)).collect::<Vec<_>>());
        // Repeated and opposite bases, so bucket points collide and cancel.
        bases[1] = bases[0];
        bases[2] = -bases[0];
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
        let bigints: Vec<BigInt> = scalars.iter().map(|s| s.into_bigint()).collect();
        let expected: G = bases.iter().zip(&scalars).map(|(b, s)| *b * s).sum();
        for &c in cs {
            let table = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, c);
            assert_eq!(table.msm_bigint(&bigints), expected, "n {n} c {c}");
        }
    }
}
