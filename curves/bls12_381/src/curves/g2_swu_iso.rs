use crate::*;

use ark_ec::{
    hashing::curve_maps::{swu::SWUConfig, wb::IsogenyMap},
    models::{
        short_weierstrass::{Affine, SWCurveConfig},
        CurveConfig,
    },
};
use ark_ff::{AdditiveGroup, Field, MontFp, Zero};

type G2Affine = Affine<SwuIsoConfig>;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct SwuIsoConfig;

impl CurveConfig for SwuIsoConfig {
    type BaseField = Fq2;
    type ScalarField = Fr;

    /// Cofactors of g2_iso and g2 are the same.
    /// COFACTOR = (x^8 - 4 x^7 + 5 x^6) - (4 x^4 + 6 x^3 - 4 x^2 - 4 x + 13) //
    /// 9
    /// = 3055023339312683442009997531931215042144660192541881426676640329822676041829718840265074273592599778478322728390416166612858038233783720963550777062779109
    const COFACTOR: &'static [u64] = &[
        0xcf1c38e31c7238e5,
        0x1616ec6e786f0c70,
        0x21537e293a6691ae,
        0xa628f1cb4d9e82ef,
        0xa68a205b2e5a7ddf,
        0xcd91de4547085aba,
        0x91d50792876a202,
        0x5d543a95414e7f1,
    ];

    /// COFACTOR_INV = COFACTOR^{-1} mod r
    /// 26652489039290660355457965112010883481355318854675681319708643586776743290055
    const COFACTOR_INV: Fr =
        MontFp!("26652489039290660355457965112010883481355318854675681319708643586776743290055");
}

// https://datatracker.ietf.org/doc/draft-irtf-cfrg-hash-to-curve/
// Hashing to Elliptic Curves
// 8.8.2.  BLS12-381 G2
//   * E': y'^2 = x'^3 + A' * x' + B', where
//
//      - A' = 240 * I
//
//      - B' = 1012 * (1 + I)
//
//   * Z: -(2 + I)
impl SWCurveConfig for SwuIsoConfig {
    /// COEFF_A = 240 * I
    const COEFF_A: Fq2 = Fq2::new(MontFp!("0"), MontFp!("240"));

    /// COEFF_B = 1012 + 1012 * I
    const COEFF_B: Fq2 = Fq2::new(MontFp!("1012"), MontFp!("1012"));

    const GENERATOR: G2Affine = G2Affine::new_unchecked(G2_GENERATOR_X, G2_GENERATOR_Y);

    /// Correctness:
    /// Substituting (0, 0) into the curve equation gives 0^2 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();
}

/// Lexicographically smallest, valid x-coordinate of a point P on the curve
/// (with its corresponding y) multiplied by the cofactor. P_x = 1
/// P_y = 1199519624119946820355795551601605892701128025883245860600494152840508171012839086684258857614063467038089173303263 + 2721622435888802346851223931977585460571674503470326381323808470905804676865417627238564067834747838523978879375704 * I
/// P = E(P_x, P_y)
/// G = P * COFACTOR
const G2_GENERATOR_X: Fq2 = Fq2::new(
    MontFp!("2595569946714414516067015540153643524656442638788025933727967960306287756885400469291119095920626560658971252184199"),
    MontFp!("1037079738597573406765355774006601850633656296583542639082316151670128374872040593053087014315526494961765370307992")
);
const G2_GENERATOR_Y: Fq2 = Fq2::new(
    MontFp!("3927929472994661655038722055497331445175131868678630546921475383290711810401295661250673209427965906654429357114487"),
    MontFp!("3300326318345570015758639333209189167876318321385223785506096497597561910823001330832964776707374262378602791224889")
);

impl SWUConfig for SwuIsoConfig {
    // ZETA = -(2 + u) as per IETF draft.
    const ZETA: Fq2 = Fq2::new(MontFp!("-2"), MontFp!("-1"));

    /// Inline `Fq2` sqrt-ratio. `Fq2` has `p^2 = 1 mod 4` elements, so the single-exponent
    /// root of the G1 override does not apply, but `Fq` is `3 mod 4`, so an `Fq` square
    /// root is one exponentiation. The complex method takes the root of `t = t0 + t1 u`
    /// as `(c0, t1 / (2 c0))` with `c0^2 = delta = (t0 + sqrt(norm(t))) / 2` or
    /// `delta - sqrt(norm(t))`, whichever is a square in `Fq`. `gx1` is a QR in `Fq2` iff
    /// its norm is a QR in `Fq`, so a single `norm^((p+1)/4)` decides the branch; when
    /// `gx1` is not a QR,
    /// `sqrt(norm(ZETA*gx1)) = N(ZETA)^((p+1)/4) * norm(gx1)^((p+1)/4)` avoids a
    /// second full `Fq2` root. The inner root takes `d = delta^((p+1)/4)` once. When
    /// `d^2 = -delta`, the other candidate `delta - sqrt(norm)` has root `c1 / (2d)`, so
    /// the result is `(c1 / (2d), d)`. A `target` in `Fq` has root `(d, 0)` or `(0, d)`
    /// for `d = target^((p+1)/4)`. Two `Fq` exponentiations and at most one `Fq` inversion
    /// on every path, against three exponentiations per generic `Fq2` root. Complex-method
    /// `Fp2` square root: Adj, Rodriguez-Henriquez, "Square Root Computation over Even
    /// Extension Fields", <https://eprint.iacr.org/2012/685> (algorithm for `q = 3 mod 4`);
    /// the surrounding `sqrt_ratio` is RFC 9380 appendix F.2 / G.2.3
    /// (<https://www.rfc-editor.org/rfc/rfc9380>).
    fn sqrt_or_zeta_sqrt(gx1: Fq2) -> (bool, Fq2) {
        // (p + 1) / 4, little-endian.
        const EXP: [u64; 6] = [
            0xee7fbfffffffeaab,
            0x07aaffffac54ffff,
            0xd9cc34a83dac3d89,
            0xd91dd2e13ce144af,
            0x92c6e9ed90d2eb35,
            0x0680447a8e5ff9a6,
        ];
        // N(ZETA)^((p+1)/4) where N(ZETA) = (-2)^2 + (-1)^2 = 5.
        const C: Fq = MontFp!("248294325734266649657405162895821171812231848760181225578082735178502750823719347628762635478508544819911854747095");
        // 1/2 = (p + 1) / 2.
        const TWO_INV: Fq = MontFp!("2001204777610833696708894912867952078278441409969503942666029068062015825245418932221343814564507832018947136279894");

        // norm(gx1) = gx1.c0^2 + gx1.c1^2 (nonresidue is -1).
        let n = gx1.c0.square() + gx1.c1.square();
        let alpha = n.pow(EXP);
        let is_qr = alpha.square() == n;
        let (target, sqrt_norm) = if is_qr {
            (gx1, alpha)
        } else {
            (Self::ZETA * gx1, C * alpha)
        };

        let (t0, t1) = (target.c0, target.c1);
        if t1.is_zero() {
            // `d^2 = t0` or `d^2 = -t0 = t0 * u^2`.
            let d = t0.pow(EXP);
            let y = if d.square() == t0 {
                Fq2::new(d, Fq::ZERO)
            } else {
                Fq2::new(Fq::ZERO, d)
            };
            return (is_qr, y);
        }

        // Complete the complex-method sqrt of `target` from sqrt(norm(target)).
        // `delta * (delta - sqrt_norm) = -t1^2 / 4` is non-zero, so `d` is non-zero.
        let delta = (sqrt_norm + t0) * TWO_INV;
        let d = delta.pow(EXP);
        let e = t1 * TWO_INV * d.inverse().unwrap();
        let y = if d.square() == delta {
            Fq2::new(d, e)
        } else {
            Fq2::new(e, d)
        };
        (is_qr, y)
    }
}

pub const ISOGENY_MAP_TO_G2  : IsogenyMap<'_, SwuIsoConfig, g2::Config> = IsogenyMap {
    x_map_numerator: &[
        Fq2::new(
                   MontFp!("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542"),
                   MontFp!("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235542")),
        Fq2::new(
                   MontFp!("0"),
                   MontFp!("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706522")),
        Fq2::new(
                   MontFp!("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706526"),
                   MontFp!("1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853261")),
        Fq2::new(
                   MontFp!("3557697382419259905260257622876359250272784728834673675850718343221361467102966990615722337003569479144794908942033"),
                   MontFp!("0")),
    ],

    x_map_denominator:  &[
        Fq2::new(
                   MontFp!("0"),
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559715")),
        Fq2::new(
                   MontFp!("12"),
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559775")),
        Fq2::new(
                   MontFp!("1"),
                   MontFp!("0")),
    ],

    y_map_numerator: &[
        Fq2::new(
                   MontFp!("3261222600550988246488569487636662646083386001431784202863158481286248011511053074731078808919938689216061999863558"),
                   MontFp!("3261222600550988246488569487636662646083386001431784202863158481286248011511053074731078808919938689216061999863558")),
        Fq2::new(
                   MontFp!("0"),
                   MontFp!("889424345604814976315064405719089812568196182208668418962679585805340366775741747653930584250892369786198727235518")),
        Fq2::new(
                   MontFp!("2668273036814444928945193217157269437704588546626005256888038757416021100327225242961791752752677109358596181706524"),
                   MontFp!("1334136518407222464472596608578634718852294273313002628444019378708010550163612621480895876376338554679298090853263")),
        Fq2::new(
                   MontFp!("2816510427748580758331037284777117739799287910327449993381818688383577828123182200904113516794492504322962636245776"),
                   MontFp!("0")),
    ],

    y_map_denominator: &[
        Fq2::new(
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559355"),
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559355")),
        Fq2::new(
                   MontFp!("0"),
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559571")),
        Fq2::new(
                   MontFp!("18"),
                   MontFp!("4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559769")),
        Fq2::new(
                   MontFp!("1"),
                   MontFp!("0")),
    ],
};

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gen() {
        let gen: G2Affine = curves::g2_swu_iso::SwuIsoConfig::GENERATOR;
        assert!(gen.is_on_curve());
        assert!(gen.is_in_correct_subgroup_assuming_on_curve());
    }

    /// `sqrt_or_zeta_sqrt(gx1)` returns `y` with `y^2 = gx1` when `gx1` is a square, else
    /// `y^2 = ZETA * gx1`. Covers `gx1` in `Fq` and `gx1 = a / ZETA` for `a` in `Fq`, whose
    /// `target` has `c1 = 0`, plus random `gx1`.
    #[test]
    fn test_sqrt_or_zeta_sqrt() {
        use ark_ff::UniformRand;

        fn check(gx1: Fq2) {
            let (is_square, y) = SwuIsoConfig::sqrt_or_zeta_sqrt(gx1);
            assert_eq!(is_square, gx1.sqrt().is_some());
            let target = if is_square {
                gx1
            } else {
                SwuIsoConfig::ZETA * gx1
            };
            assert_eq!(y.square(), target);
        }

        let zeta_inv = SwuIsoConfig::ZETA.inverse().unwrap();
        for a in [0u64, 1, 2, 3] {
            for a in [Fq::from(a), -Fq::from(a)] {
                check(Fq2::new(a, Fq::ZERO));
                check(Fq2::new(a, Fq::ZERO) * zeta_inv);
            }
        }
        let mut rng = ark_std::test_rng();
        for _ in 0..1000 {
            check(Fq2::rand(&mut rng));
        }
    }
}
