use crate::models::short_weierstrass::SWCurveConfig;
use ark_ff::{Field, One, Zero};
use core::marker::PhantomData;

use crate::{
    hashing::{curve_maps::parity, map_to_curve_hasher::MapToCurve, HashToCurveError},
    models::short_weierstrass::{Affine, Projective},
    AffineRepr,
};

/// Trait defining the necessary parameters for the SWU hash-to-curve method
/// for the curves of Weierstrass form of:
/// y^2 = x^3 + a*x + b where ab != 0. From [\[WB2019\]]
///
/// - [\[WB2019\]] <https://eprint.iacr.org/2019/403>
pub trait SWUConfig: SWCurveConfig {
    /// An element of the base field that is not a square root see \[WB2019, Section 4\].
    /// It is also convenient to have $g(b/ZETA * a)$ to be square. In general
    /// we use a `ZETA` with low absolute value coefficients when they are
    /// represented as integers.
    const ZETA: Self::BaseField;

    /// Given `gx1`, returns `(is_square, y)` with `y^2 == gx1` when `is_square`,
    /// else `y^2 == ZETA * gx1`. Exactly one holds since `ZETA` is a non-square.
    /// The default takes up to two square roots; a base field where a single
    /// exponentiation yields a root (e.g. `p = 3 mod 4`) should override this to
    /// avoid the second one.
    ///
    /// This is the `sqrt_ratio` primitive of RFC 9380
    /// (<https://www.rfc-editor.org/rfc/rfc9380>, appendix F.2), specialized to a
    /// unit denominator; the optimization lineage is Wahby, Boneh,
    /// <https://eprint.iacr.org/2019/403> section 4.
    fn sqrt_or_zeta_sqrt(gx1: Self::BaseField) -> (bool, Self::BaseField) {
        match gx1.sqrt() {
            Some(y) => (true, y),
            None => {
                let y = (Self::ZETA * gx1).sqrt().expect(
                    "ZETA * gx1 is a quadratic residue because the Legendre symbol is multiplicative",
                );
                (false, y)
            },
        }
    }
}

/// Helper function for `SWUConfig::sqrt_or_zeta_sqrt` when the base field is `p = 3 mod 4`.
/// Takes `gx1`, `exp` = `(p + 1) / 4`, and `zeta_pow` = `ZETA^((p + 1) / 4)`.
pub fn sqrt_ratio_3mod4<F: Field>(gx1: F, exp: &[u64], zeta_pow: F) -> (bool, F) {
    let candidate = gx1.pow(exp);
    if candidate.square() == gx1 {
        (true, candidate)
    } else {
        (false, zeta_pow * candidate)
    }
}

/// Represents the SWU hash-to-curve map defined by `P`.
pub struct SWUMap<P: SWUConfig>(PhantomData<fn() -> P>);

impl<P: SWUConfig> MapToCurve<Projective<P>> for SWUMap<P> {
    /// Checks if `P` represents a valid map.
    fn check_parameters() -> Result<(), HashToCurveError> {
        // Verifying that ZETA is a non-square
        debug_assert!(
            P::ZETA.legendre().is_qnr(),
            "ZETA should be a quadratic non-residue for the SWU map"
        );

        // Verifying the prerequisite for applicability  of SWU map
        debug_assert!(!P::COEFF_A.is_zero() && !P::COEFF_B.is_zero(),
		      "Simplified SWU requires a * b != 0 in the short Weierstrass form of y^2 = x^3 + a*x + b ");

        Ok(())
    }

    /// Map an arbitrary base field element to a curve point.
    /// Based on
    /// <https://github.com/zcash/pasta_curves/blob/main/src/hashtocurve.rs>.
    fn map_to_curve(element: P::BaseField) -> Result<Affine<P>, HashToCurveError> {
        let parts = Self::map_to_curve_unscaled(element);
        let div_inv = parts.div.inverse().expect("`div` is never zero");
        Ok(parts.finish(div_inv))
    }

    /// Both images through one shared inversion of their denominators, summed projectively.
    fn map_to_curve_sum(
        u0: P::BaseField,
        u1: P::BaseField,
    ) -> Result<Projective<P>, HashToCurveError> {
        let (p0, p1) = Self::map_to_curve_pair(u0, u1);
        Ok(p0.into_group() + p1)
    }
}

/// An SSWU image before its one inversion: the point is `(num_x / div, num_y / div^2)` up to
/// the sign of `y`, which [`Self::finish`] fixes from `u`.
struct UnscaledImage<F> {
    u: F,
    num_x: F,
    num_y: F,
    div: F,
}

impl<F: Field> UnscaledImage<F> {
    /// The affine image from `div_inv = 1 / div`, with `sgn0(y) = sgn0(u)`.
    fn finish<P: SWUConfig<BaseField = F>>(&self, div_inv: F) -> Affine<P> {
        let x = self.num_x * div_inv;
        let y = self.num_y * div_inv.square();
        let y = if parity(&y) == parity(&self.u) { y } else { -y };
        let point = Affine::new_unchecked(x, y);
        debug_assert!(point.is_on_curve(), "swu mapped to a point off the curve");
        point
    }
}

impl<P: SWUConfig> SWUMap<P> {
    /// The images of `u0` and `u1` through one inversion shared by their denominators.
    pub fn map_to_curve_pair(u0: P::BaseField, u1: P::BaseField) -> (Affine<P>, Affine<P>) {
        let (a, b) = (Self::map_to_curve_unscaled(u0), Self::map_to_curve_unscaled(u1));
        let ab_inv = (a.div * b.div).inverse().expect("`div` is never zero");
        (a.finish(ab_inv * b.div), b.finish(ab_inv * a.div))
    }

    /// The SSWU map up to its final division by `div`.
    fn map_to_curve_unscaled(element: P::BaseField) -> UnscaledImage<P::BaseField> {
        // 1. tv1 = inv0(Z^2 * u^4 + Z * u^2)
        // 2. x1 = (-B / A) * (1 + tv1)
        // 3. If tv1 == 0, set x1 = B / (Z * A)
        // 4. gx1 = x1^3 + A * x1 + B
        //
        // We use the "Avoiding inversions" optimization in [WB2019, section 4.2]
        // (not to be confused with section 4.3):
        //
        //   here       [WB2019]
        //   -------    ---------------------------------
        //   Z          ξ
        //   u          t
        //   Z * u^2    ξ * t^2 (called u, confusingly)
        //   x1         X_0(t)
        //   x2         X_1(t)
        //   gx1        g(X_0(t))
        //   gx2        g(X_1(t))
        //
        // Using the "here" names:
        //    x1 = num_x1/div      = [B*(Z^2 * u^4 + Z * u^2 + 1)] / [-A*(Z^2 * u^4 + Z * u^2]
        //   gx1 = num_gx1/div_gx1 = [num_x1^3 + A * num_x1 * div^2 + B * div^3] / div^3
        let a = P::COEFF_A;
        let b = P::COEFF_B;

        let zeta_u2 = P::ZETA * element.square();
        let ta = zeta_u2.square() + zeta_u2;
        let num_x1 = b * (ta + <P::BaseField as One>::one());
        // Never zero: `A` and `ZETA` are nonzero.
        let div = a * if ta.is_zero() { P::ZETA } else { -ta };

        let num2_x1 = num_x1.square();
        let div2 = div.square();
        let div3 = div2 * div;
        let num_gx1 = (num2_x1 + a * div2) * num_x1 + b * div3;

        // 5. x2 = Z * u^2 * x1
        let num_x2 = zeta_u2 * num_x1; // same div

        // 6. gx2 = x2^3 + A * x2 + B  [optimized out; see below]
        // 7. If is_square(gx1), set x = x1 and y = sqrt(gx1)
        // 8. Else set x = x2 and y = sqrt(gx2)
        //
        // `gx1 = num_gx1 * div / div^4` is a square exactly when `num_gx1 * div` is, and
        // `sqrt(gx1) = sqrt(num_gx1 * div) / div^2`, so the root needs no inversion.
        let (gx1_square, y1_num) = P::sqrt_or_zeta_sqrt(num_gx1 * div);

        // This optimization also comes from a generalization of [WB2019, section 4.2].
        //
        // We use the specialization with h = Z = ZETA (a fixed quadratic non-residue).
        // Since gx2 = g(Z * u^2 * x1) = Z^3 * u^6 * gx1, when gx1 is not square we take
        // y1 such that y1^2 = Z * gx1 (i.e., y1 = sqrt(Z * gx1)), and then set
        // y2 = Z * u^3 * y1. This gives y2^2 = (Z * u^3)^2 * (Z * gx1) = Z^3 * u^6 * gx1 = gx2,
        // so we avoid computing gx2 explicitly.
        let y2_num = zeta_u2 * element * y1_num;
        UnscaledImage {
            u: element,
            num_x: if gx1_square { num_x1 } else { num_x2 },
            num_y: if gx1_square { y1_num } else { y2_num },
            div,
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(all(
        target_has_atomic = "8",
        target_has_atomic = "16",
        target_has_atomic = "32",
        target_has_atomic = "64",
        target_has_atomic = "ptr"
    ))]
    type DefaultHasher = ahash::AHasher;

    #[cfg(not(all(
        target_has_atomic = "8",
        target_has_atomic = "16",
        target_has_atomic = "32",
        target_has_atomic = "64",
        target_has_atomic = "ptr"
    )))]
    type DefaultHasher = fnv::FnvHasher;

    use crate::{
        hashing::{map_to_curve_hasher::MapToCurveBasedHasher, HashToCurve},
        CurveConfig,
    };
    use ark_ff::field_hashers::DefaultFieldHasher;
    use ark_std::vec::*;

    use super::*;
    use ark_ff::{fields::Fp64, MontBackend, MontFp};
    use hashbrown::HashMap;
    use sha2::Sha256;

    #[derive(ark_ff::MontConfig)]
    #[modulus = "127"]
    #[generator = "6"]
    pub(crate) struct F127Config;
    pub(crate) type F127 = Fp64<MontBackend<F127Config, 1>>;

    const F127_ONE: F127 = MontFp!("1");

    struct TestSWUMapToCurveConfig;

    impl CurveConfig for TestSWUMapToCurveConfig {
        const COFACTOR: &[u64] = &[1];

        const COFACTOR_INV: F127 = F127_ONE;

        type BaseField = F127;
        type ScalarField = F127;
    }

    /// just because not defining another field
    ///
    /// from itertools import product
    /// p = 127
    /// FF = GF(p)
    /// for a,b in product(range(0,p), range(0,p)):
    ///     try:
    ///         E = EllipticCurve([FF(a),FF(b)])
    ///         if E.order() == p:
    ///             print(E)
    ///     except:
    ///         pass
    ///
    /// y^2 = x^3 + x + 63
    impl SWCurveConfig for TestSWUMapToCurveConfig {
        /// COEFF_A = 1
        const COEFF_A: F127 = F127_ONE;

        /// COEFF_B = 63
        const COEFF_B: F127 = MontFp!("63");

        /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
        const GENERATOR: Affine<Self> = Affine::new_unchecked(MontFp!("62"), MontFp!("70"));

        /// We use `bool` because the point (0, 0) could be on the curve.
        type ZeroFlag = bool;
    }

    impl SWUConfig for TestSWUMapToCurveConfig {
        const ZETA: F127 = MontFp!("-1");
    }

    /// test that MontFp make a none zero element out of 1
    #[test]
    fn test_field_element_construction() {
        let a1 = F127::from(1);
        let a2 = F127::from(2);
        let a3 = F127::from(125);

        assert!(F127::from(0) == a2 + a3);
        assert!(F127::from(0) == a2 * a1 + a3);
    }

    #[test]
    fn test_field_division() {
        let num = F127::from(0x3d);
        let den = F127::from(0x7b);
        let num_on_den = F127::from(0x50);

        assert!(num / den == num_on_den);
    }

    /// The point of the test is to get a simple SWU compatible curve and make
    /// simple hash
    #[test]
    fn hash_arbitrary_string_to_curve_swu() {
        let test_swu_to_curve_hasher = MapToCurveBasedHasher::<
            Projective<TestSWUMapToCurveConfig>,
            DefaultFieldHasher<Sha256, 128>,
            SWUMap<TestSWUMapToCurveConfig>,
        >::new(&[1])
        .unwrap();

        let hash_result = test_swu_to_curve_hasher.hash(b"if you stick a Babel fish in your ear you can instantly understand anything said to you in any form of language.").expect("fail to hash the string to curve");

        assert!(
            hash_result.is_on_curve(),
            "hash results into a point off the curve"
        );
    }

    /// Use a simple SWU compatible curve and map the whole field to it. We observe
    /// the map behaviour. Specifically, the map should be non-constant, all
    /// elements should be mapped to curve successfully. everything can be mapped
    #[test]
    fn map_field_to_curve_swu() {
        SWUMap::<TestSWUMapToCurveConfig>::check_parameters().unwrap();

        let mut map_range: Vec<Affine<TestSWUMapToCurveConfig>> = Vec::with_capacity(128);
        for current_field_element in 0..127 {
            let element = F127::from(current_field_element as u64);
            map_range.push(SWUMap::map_to_curve(element).unwrap());
        }

        let mut counts =
            HashMap::with_hasher(core::hash::BuildHasherDefault::<DefaultHasher>::default());

        let mode = map_range
            .iter()
            .copied()
            .max_by_key(|&n| {
                let count = counts.entry(n).or_insert(0);
                *count += 1;
                *count
            })
            .unwrap();

        assert!(
            *counts.get(&mode).unwrap() != 127,
            "a constant hash function is not good."
        );
    }
}
