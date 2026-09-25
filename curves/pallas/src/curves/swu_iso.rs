use crate::{Fq, Fr, PallasConfig};
use ark_ec::{
    hashing::curve_maps::{swu::SWUConfig, wb::IsogenyMap},
    models::{
        short_weierstrass::{Affine, SWCurveConfig},
        CurveConfig,
    },
};
use ark_ff::{Field, MontFp};

type IsoAffine = Affine<SwuIsoConfig>;

/// The curve `y^2 = x^3 + A' * x + B'` that is 3-isogenous to Pallas, used by the simplified SWU
/// map since Pallas has `A = 0`. Constants are from
/// pasta_curves
/// [`IsoEp`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1052-L1069) and
/// [`Ep::Z`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1173-L1178).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct SwuIsoConfig;

impl CurveConfig for SwuIsoConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1. The curve is isogenous to Pallas so has the same prime order.
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

impl SWCurveConfig for SwuIsoConfig {
    const COEFF_A: Fq =
        MontFp!("10949663248450308183708987909873589833737836120165333298109615750520499732811");

    /// COEFF_B = 1265
    const COEFF_B: Fq = MontFp!("1265");

    const GENERATOR: IsoAffine = IsoAffine::new_unchecked(ISO_GENERATOR_X, ISO_GENERATOR_Y);

    /// Correctness:
    /// Substituting (0, 0) into the curve equation gives 0^2 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();
}

/// Smallest valid x-coordinate with its even y.
const ISO_GENERATOR_X: Fq = MontFp!("0");
const ISO_GENERATOR_Y: Fq =
    MontFp!("18757143033426632116356352624818188154880656819263833216198390681205662132804");

/// `ZETA^((T - 1) / 2)` with `p - 1 = T * 2^32`, `T` odd.
pub(crate) const ZETA_TRACE_POWER: Fq =
    MontFp!("24572433101797051318859476537138851131189150516971437659543983992465105113839");

impl SWUConfig for SwuIsoConfig {
    /// ZETA = -13
    const ZETA: Fq = MontFp!("-13");

    /// Both roots from one exponentiation, through
    /// [`SqrtPrecomputation::sqrt_or_scaled_sqrt`](ark_ff::SqrtPrecomputation::sqrt_or_scaled_sqrt).
    fn sqrt_or_zeta_sqrt(gx1: Fq) -> (bool, Fq) {
        match <Fq as Field>::SQRT_PRECOMP {
            Some(precomp) => precomp.sqrt_or_scaled_sqrt(&gx1, &Self::ZETA, &ZETA_TRACE_POWER),
            None => unreachable!("the base field has a Sarkar square-root table"),
        }
    }
}

/// The 3-isogeny from `SwuIsoConfig` to Pallas. Coefficients are from
/// pasta_curves
/// [`Ep::ISOGENY_CONSTANTS`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1091-L1170)
/// and listed lowest degree first. `scripts/swu_iso.sage` checks them against Sage.
pub const ISOGENY_MAP_TO_PALLAS: IsogenyMap<'_, SwuIsoConfig, PallasConfig> = IsogenyMap {
    x_map_numerator: &[
        MontFp!("12865787693035132824841220556520878650383580658640693651535411895266652280192"),
        MontFp!("10492611921771203378452795982353351666191589197598957448093274638589204800759"),
        MontFp!("23989696149150192365340222745168215001509815558210986772351135915822265203574"),
        MontFp!("6432893846517566412420610278260439325191790329320346825767705947633326140075"),
    ],

    x_map_denominator: &[
        MontFp!("22768321103861051515190775253992702316905399997697804654926324362758820947460"),
        MontFp!("13271109177048389296812780941310096270046944650307955939477485891950613419807"),
        MontFp!("1"),
    ],

    y_map_numerator: &[
        MontFp!("1072148974419594402070101713043406554198631721553391137627950991272221023311"),
        MontFp!("28823569610051396102362669851238297121581474897215657071023781420043761726004"),
        MontFp!("11994848074575096182670111372584107500754907779105493386175567957911132601787"),
        MontFp!("11793638718615538422771118843477472096184948937087302513907460903994431256804"),
    ],

    y_map_denominator: &[
        MontFp!("28948022309329048855892746252171976963363056481941560715954676764349967629797"),
        MontFp!("10408918692925056833786833257634153023990087029210292532869619559576527581706"),
        MontFp!("5432652610908059517272798285879155923388888734491153551238890455750936314542"),
        MontFp!("1"),
    ],
};

#[cfg(test)]
mod test {
    use super::*;
    use ark_ec::{
        hashing::{curve_maps::wb::WBMap, map_to_curve_hasher::MapToCurve},
        AffineRepr,
    };
    use ark_ff::Zero;

    #[test]
    fn test_gen() {
        let gen: IsoAffine = SwuIsoConfig::GENERATOR;
        assert!(gen.is_on_curve());
        assert!(gen.mul_bigint(Fr::characteristic()).is_zero());
    }

    #[test]
    fn test_isogeny_map() {
        WBMap::<PallasConfig>::check_parameters().unwrap();
        for i in 0..64u64 {
            let p = WBMap::<PallasConfig>::map_to_curve(Fq::from(i)).unwrap();
            assert!(p.is_on_curve());
        }
    }
}
