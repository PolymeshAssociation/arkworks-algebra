use ark_ec::{
    models::CurveConfig,
    scalar_mul::glv::GLVConfig,
    short_weierstrass::{self as sw, SWCurveConfig},
    AffineRepr,
};
use ark_ff::{AdditiveGroup, Field, MontFp, Zero};

use crate::{fq::Fq, fr::Fr};

#[cfg(test)]
mod tests;

pub type Affine = sw::Affine<Config>;
pub type Projective = sw::Projective<Config>;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct Config;

impl CurveConfig for Config {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = COFACTOR^{-1} mod r = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

impl SWCurveConfig for Config {
    /// COEFF_A = 0
    const COEFF_A: Fq = Fq::ZERO;

    /// COEFF_B = 7
    const COEFF_B: Fq = MontFp!("7");

    /// GENERATOR = (G_GENERATOR_X, G_GENERATOR_Y)
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

    #[inline]
    fn mul_projective(base: &Projective, scalar: &[u64]) -> Projective {
        let s = Self::ScalarField::from_sign_and_limbs(true, scalar);
        GLVConfig::glv_mul_projective(*base, s)
    }

    #[inline]
    fn mul_affine(base: &Affine, scalar: &[u64]) -> Projective {
        let s = Self::ScalarField::from_sign_and_limbs(true, scalar);
        <Self as GLVConfig>::glv_mul_projective(base.into_group(), s)
    }
}

impl GLVConfig for Config {
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "78074008874160198520644763525212887401909906723592317393988542598630163514318"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("60197513588986302554485582024885075108884032450952339817679072026166228089408");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as ark_ff::PrimeField>::BigInt); 4] = [
        (false, ark_ff::BigInt!("303414439467246543595250775667605759171")),
        (true, ark_ff::BigInt!("64502973549206556628585045361533709078")),
        (false, ark_ff::BigInt!("64502973549206556628585045361533709078")),
        (false, ark_ff::BigInt!("367917413016453100223835821029139468249")),
    ];

    const FAST_DECOMP: Option<ark_ec::scalar_mul::glv::GLVFastDecomp<Self::ScalarField>> = Some(ark_ec::scalar_mul::glv::GLVFastDecomp {
        g1: &[
            0x0000000114ca5518,
            0x0000000000000000,
            0x57c1108d9d44cfd9,
            0x14ca50f7a8e2f3f6,
            0x0000000000000001,
        ],
        g2: &[
            0x000000003086d2db,
            0x0000000000000000,
            0xe86c90e49284eb16,
            0x3086d221a7d46bcd,
            0x0000000000000000,
        ],
        a12: MontFp!("64502973549206556628585045361533709078"),
        a22: MontFp!("367917413016453100223835821029139468249"),
        negate_k2: true,
    });

    fn endomorphism(p: &Projective) -> Projective {
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }

    fn endomorphism_affine(p: &Affine) -> Affine {
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

/// G_GENERATOR_X =
/// 53718550993811904772965658690407829053653678808745171666022356150019200052646
pub const G_GENERATOR_X: Fq =
    MontFp!("53718550993811904772965658690407829053653678808745171666022356150019200052646");

/// G_GENERATOR_Y =
/// 28941648020349172432234515805717979317553499307621291159490218670604692907903
pub const G_GENERATOR_Y: Fq =
    MontFp!("28941648020349172432234515805717979317553499307621291159490218670604692907903");
