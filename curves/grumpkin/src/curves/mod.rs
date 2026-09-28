// The parameters for the curve have been taken from
// https://github.com/AztecProtocol/barretenberg/blob/97ccf76c42db581a8b8f8bfbcffe8ca015a3dd22/cpp/src/barretenberg/ecc/curves/grumpkin/grumpkin.hpp

use crate::{fq::Fq, fr::Fr};
use ark_ec::{
    models::CurveConfig,
    scalar_mul::glv::{try_glv_msm_small, GLVConfig, GLVFastDecomp},
    short_weierstrass::{self as sw, SWCurveConfig},
};
use ark_ff::{AdditiveGroup, BigInt, Field, MontFp, PrimeField, Zero};

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct GrumpkinConfig;

impl CurveConfig for GrumpkinConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

pub type Affine = sw::Affine<GrumpkinConfig>;
pub type Projective = sw::Projective<GrumpkinConfig>;

impl SWCurveConfig for GrumpkinConfig {
    /// COEFF_A = 0
    const COEFF_A: Fq = Fq::ZERO;

    /// COEFF_B = -17
    const COEFF_B: Fq = MontFp!("-17");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: Affine = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);

    /// Correctness:
    /// Substituting (0, 0) into the curve equation gives 0^2 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();

    #[inline(always)]
    fn mul_by_a(_: Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }

    /// GLV scalar multiplication. The cofactor is 1, so every curve point is in the
    /// order-`r` group.
    #[inline]
    fn mul_projective(p: &Projective, scalar: &[u64]) -> Projective {
        <Self as GLVConfig>::glv_mul_projective_bigint(p, scalar)
    }

    #[inline]
    fn mul_affine(p: &Affine, scalar: &[u64]) -> Projective {
        <Self as GLVConfig>::glv_mul_affine_projective_bigint(p, scalar)
    }

    #[inline]
    fn try_msm_small(bases: &[Affine], scalars: &[Self::ScalarField]) -> Option<Projective> {
        try_glv_msm_small::<Self>(bases, scalars)
    }
}

/// `phi(x, y) = (beta x, y) = [lambda](x, y)`. Grumpkin swaps BN254's fields, so
/// `beta` is BN254 G1's `LAMBDA` and `lambda` is BN254 G1's `beta`; the basis is the
/// Gauss-reduced lattice `{(a, b) : a + b lambda = 0 mod r}` (`scripts/find_glv_parameters.py`).
impl GLVConfig for GrumpkinConfig {
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "21888242871839275217838484774961031246154997185409878258781734729429964517155"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("21888242871839275220042445260109153167277707414472061641714758635765020556616");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, BigInt!("147946756881789319000765030803803410729")),
        (true, BigInt!("9931322734385697762")),
        (false, BigInt!("9931322734385697762")),
        (false, BigInt!("147946756881789319010696353538189108491")),
    ];

    const FAST_DECOMP: Option<GLVFastDecomp<Self::ScalarField>> = Some(GLVFastDecomp {
        g1: &[
            0xf2c2f0c531c40a89,
            0xca560de9b2444b57,
            0x5398fd0300ff655f,
            0x4ccef014a773d2d2,
            0x0000000000000002,
        ],
        g2: &[
            0x5236df9ec85147d0,
            0x247280ee539a2471,
            0xd91d232ec7e0b3d2,
            0x0000000000000002,
            0x0000000000000000,
        ],
        a12: MontFp!("9931322734385697762"),
        a22: MontFp!("147946756881789319010696353538189108491"),
        negate_k2: true,
    });

    fn endomorphism(p: &Projective) -> Projective {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }

    fn endomorphism_affine(p: &Affine) -> Affine {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

/// G_GENERATOR_X = 1
pub const G_GENERATOR_X: Fq = MontFp!("1");

/// G_GENERATOR_Y = sqrt(-16)
pub const G_GENERATOR_Y: Fq =
    MontFp!("17631683881184975370165255887551781615748388533673675138860");
