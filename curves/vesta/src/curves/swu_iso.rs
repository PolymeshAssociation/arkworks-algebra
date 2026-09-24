use crate::{Fq, Fr, VestaConfig};
use ark_ec::{
    hashing::curve_maps::{swu::SWUConfig, wb::IsogenyMap},
    models::{
        short_weierstrass::{Affine, SWCurveConfig},
        CurveConfig,
    },
};
use ark_ff::{Field, MontFp};

type IsoAffine = Affine<SwuIsoConfig>;

/// The curve `y^2 = x^3 + A' * x + B'` that is 3-isogenous to Vesta, used by the simplified SWU
/// map since Vesta has `A = 0`. Constants are from
/// pasta_curves
/// [`IsoEq`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1070-L1087) and
/// [`Eq::Z`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1273-L1278).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct SwuIsoConfig;

impl CurveConfig for SwuIsoConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1. The curve is isogenous to Vesta so has the same prime order.
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

impl SWCurveConfig for SwuIsoConfig {
    const COEFF_A: Fq =
        MontFp!("17413348858408915339762682399132325137863850198379221683097628341577494210225");

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
    MontFp!("24168672240094656118439194445685110693789334875363060050166186932715309622324");

impl SWUConfig for SwuIsoConfig {
    /// ZETA = -13
    const ZETA: Fq = MontFp!("-13");
}

/// The 3-isogeny from `SwuIsoConfig` to Vesta. Coefficients are from
/// pasta_curves
/// [`Eq::ISOGENY_CONSTANTS`](https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs#L1191-L1270)
/// and listed lowest degree first. `scripts/swu_iso.sage` checks them against Sage.
pub const ISOGENY_MAP_TO_VESTA: IsogenyMap<'_, SwuIsoConfig, VestaConfig> = IsogenyMap {
    x_map_numerator: &[
        MontFp!("22515128462811482443472135973911537638171266152621281295306466582083726737451"),
        MontFp!("11064082577423419940183149293632076317553812518550871517841037420579891210813"),
        MontFp!("13377367003779316331268047403600734872799183885837485433911493934102207511749"),
        MontFp!("25731575386070265649682441113041757300767161317281464337493104665238544842753"),
    ],

    x_map_denominator: &[
        MontFp!("9250006497141849826017568406346290940322373181457057184910582871723433210981"),
        MontFp!("4604213796697651557841441623718706001740429044770779386484474413346415813353"),
        MontFp!("1"),
    ],

    y_map_numerator: &[
        MontFp!("13937936667454727226911322269564285204582212380194126516142098360337545123123"),
        MontFp!("11620280474556824258112134491145636201000922752744881519070727793732904824884"),
        MontFp!("21162694656554182593580396827886355918081120183889566406795618341247785229923"),
        MontFp!("8577191795356755216560813704347252433589053772427154779164368221746181614251"),
    ],

    y_map_denominator: &[
        MontFp!("28948022309329048855892746252171976963363056481941647379679742748393362947557"),
        MontFp!("27750019491425549478052705219038872820967119544371171554731748615170299632943"),
        MontFp!("21380331849711001764708535561664047484292171808126992769566582994216305194078"),
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
        WBMap::<VestaConfig>::check_parameters().unwrap();
        for i in 0..64u64 {
            let p = WBMap::<VestaConfig>::map_to_curve(Fq::from(i)).unwrap();
            assert!(p.is_on_curve());
        }
    }
}
