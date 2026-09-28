use ark_ec::{
    bn,
    hashing::curve_maps::svdw::SVDWConfig,
    models::{short_weierstrass::SWCurveConfig, CurveConfig},
    scalar_mul::glv::{try_glv_msm_bigint_full_width, GLVConfig, GLVFastDecomp},
    short_weierstrass::{Affine, Projective},
};
use ark_ff::{AdditiveGroup, BigInt, Field, MontFp, PrimeField, Zero};

use crate::{Fq, Fr};

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Config;

pub type G1Affine = Affine<Config>;

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

    /// COEFF_B = 3
    const COEFF_B: Fq = MontFp!("3");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: G1Affine = G1Affine::new_unchecked(G1_GENERATOR_X, G1_GENERATOR_Y);

    /// Correctness:
    /// The curve equation is y^2 = x^3  + b
    /// Substituting (0, 0) gives 0^2 = 0^3 + b which simplifies to 0 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();

    /// Host MSM name, distinct from the other group of the curve.
    fn curve_name() -> Option<&'static str> {
        Some("bn254_g1")
    }

    #[inline(always)]
    fn mul_by_a(_: Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }

    #[inline]
    fn mul_projective(
        p: &bn::G1Projective<crate::Config>,
        scalar: &[u64],
    ) -> bn::G1Projective<crate::Config> {
        <Self as GLVConfig>::glv_mul_projective_bigint(p, scalar)
    }

    #[inline]
    fn try_msm_bigint_full_width(
        bases: &[G1Affine],
        bigints: &[<Fr as PrimeField>::BigInt],
    ) -> Option<Projective<Self>> {
        try_glv_msm_bigint_full_width::<Self>(bases, bigints)
    }

    #[inline]
    fn is_in_correct_subgroup_assuming_on_curve(_p: &G1Affine) -> bool {
        // G1 = E(Fq) so if the point is on the curve, it is also in the subgroup.
        true
    }
}

impl GLVConfig for Config {
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "21888242871839275220042445260109153167277707414472061641714758635765020556616"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("21888242871839275217838484774961031246154997185409878258781734729429964517155");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, BigInt!("147946756881789319000765030803803410728")),
        (true, BigInt!("9931322734385697763")),
        (false, BigInt!("9931322734385697763")),
        (false, BigInt!("147946756881789319010696353538189108491")),
    ];

    // Derived from `SCALAR_DECOMP_COEFFS` by `scripts/glv_fast_decomp.py`.
    const FAST_DECOMP: Option<GLVFastDecomp<Self::ScalarField>> = Some(GLVFastDecomp {
        g1: &[
            0x163b4843cb4b9a5f,
            0x149d540fd5e495cc,
            0x5398fd0300ff6565,
            0x4ccef014a773d2d2,
            0x0000000000000002,
        ],
        g2: &[
            0x8fa7d32d2fafba64,
            0x6eb9c714773a6ef2,
            0xd91d232ec7e0b3d7,
            0x0000000000000002,
            0x0000000000000000,
        ],
        a12: MontFp!("9931322734385697763"),
        a22: MontFp!("147946756881789319010696353538189108491"),
        negate_k2: true,
    });

    fn endomorphism(p: &Projective<Self>) -> Projective<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
    fn endomorphism_affine(p: &Affine<Self>) -> Affine<Self> {
        let mut res = *p;
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

/// Shallue-van de Woestijne map of [RFC 9380, section 6.6.1](https://www.rfc-editor.org/rfc/rfc9380.html#section-6.6.1),
/// constants from `svdw_constants(GF(p), 0, 3)` of `ark-ec`'s `curve_map_parameter_helper.sage`.
/// They equal `Z`, `c1..c4` of gnark-crypto's `BN254G1_XMD:SHA-256_SVDW_RO_`
/// [`MapToCurve1`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/hash_to_g1.go#L75-L79),
/// which stores them in Montgomery form.
impl SVDWConfig for Config {
    const Z: Fq = Fq::ONE;

    const C1: Fq = MontFp!("4");

    const C2: Fq =
        MontFp!("10944121435919637611123202872628637544348155578648911831344518947322613104291");

    const C3: Fq = MontFp!("8815841940592487685674414971303048083897117035520822607866");

    const C4: Fq =
        MontFp!("7296080957279758407415468581752425029565437052432607887563012631548408736189");
}

/// G1_GENERATOR_X = 1
pub const G1_GENERATOR_X: Fq = Fq::ONE;

/// G1_GENERATOR_Y = 2
pub const G1_GENERATOR_Y: Fq = MontFp!("2");
