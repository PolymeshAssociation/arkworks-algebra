//! Correctness of the BGMW fixed-base MSM against the variable-base paths on Pallas. Covers small
//! and mid sizes, several windows, identity bases, short scalar slices, and the auto window. With
//! `--features parallel`, sizes from `n = 1024` up cross `MIN_PARALLEL_ENTRIES` at every window
//! here, so both the serial and the segmented evaluation are exercised.

use ark_ec::{
    scalar_mul::fixed_base::FixedBaseMSM, short_weierstrass::SWCurveConfig, AffineRepr, CurveGroup,
    VariableBaseMSM,
};
use ark_ec::short_weierstrass::{Affine, Projective};
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;

fn check<P: SWCurveConfig>(n: usize, cs: &[Option<usize>]) {
    let rng = &mut test_rng();
    let bases = Projective::<P>::normalize_batch(
        &(0..n).map(|_| Projective::<P>::rand(rng)).collect::<Vec<_>>(),
    );
    let scalars: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();
    let expected = Projective::<P>::msm_unchecked(&bases, &scalars);
    for &copt in cs {
        let pc = match copt {
            Some(c) => FixedBaseMSM::<P>::new_given_window_size(&bases, c),
            None => FixedBaseMSM::<P>::new(&bases),
        };
        assert_eq!(pc.msm(&scalars), expected, "mismatch n={n} c={}", pc.window());
    }
}

#[test]
fn pallas_matches_variable_base() {
    use ark_pallas::PallasConfig;
    for &n in &[1usize, 2, 5, 33, 100, 257, 1024, 2050] {
        check::<PallasConfig>(n, &[None, Some(3), Some(4), Some(8), Some(13), Some(15), Some(16)]);
    }
}

/// Identity bases scattered through a set large enough for the segmented evaluation, with the
/// scalar slice shorter than the base set.
#[test]
fn parallel_size_identity_bases_and_truncation() {
    use ark_pallas::{Fr, PallasConfig};
    let rng = &mut test_rng();
    let n = 1500;
    let mut bases = Projective::<PallasConfig>::normalize_batch(
        &(0..n).map(|_| Projective::<PallasConfig>::rand(rng)).collect::<Vec<_>>(),
    );
    for i in (0..n).step_by(7) {
        bases[i] = Affine::<PallasConfig>::zero();
    }
    let scalars: Vec<Fr> = (0..n - 3).map(|_| Fr::rand(rng)).collect();
    let expected = Projective::<PallasConfig>::msm_unchecked(&bases[..n - 3], &scalars);
    for c in [3usize, 8, 13, 16] {
        let pc = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, c);
        assert_eq!(pc.msm(&scalars), expected, "mismatch c={c}");
    }
}

#[test]
fn empty_and_identity_bases() {
    use ark_pallas::{Fr, PallasConfig};
    let rng = &mut test_rng();

    let pc = FixedBaseMSM::<PallasConfig>::new(&[]);
    assert!(pc.msm(&[]).is_zero());

    // Identity bases must contribute nothing regardless of their scalar.
    let g = Affine::<PallasConfig>::rand(rng);
    let h = Affine::<PallasConfig>::rand(rng);
    let bases = [g, Affine::<PallasConfig>::zero(), h];
    let scalars = [Fr::from(3u64), Fr::rand(rng), Fr::from(5u64)];
    let pc = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, 8);
    let expected = g.into_group() * Fr::from(3u64) + h.into_group() * Fr::from(5u64);
    assert_eq!(pc.msm(&scalars), expected);
}

#[test]
fn scalars_shorter_than_bases_are_truncated() {
    use ark_pallas::{Fr, PallasConfig};
    let rng = &mut test_rng();
    let bases = Projective::<PallasConfig>::normalize_batch(
        &(0..10).map(|_| Projective::<PallasConfig>::rand(rng)).collect::<Vec<_>>(),
    );
    let pc = FixedBaseMSM::<PallasConfig>::new_given_window_size(&bases, 6);
    let scalars: Vec<Fr> = (0..4).map(|_| Fr::rand(rng)).collect();
    let expected = Projective::<PallasConfig>::msm_unchecked(&bases[..4], &scalars);
    assert_eq!(pc.msm(&scalars), expected);
}
