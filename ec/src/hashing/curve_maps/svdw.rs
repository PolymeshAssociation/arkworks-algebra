use crate::models::short_weierstrass::SWCurveConfig;
use ark_ff::{AdditiveGroup, Field, One, Zero};
use core::marker::PhantomData;

use crate::{
    hashing::{curve_maps::parity, map_to_curve_hasher::MapToCurve, HashToCurveError},
    models::short_weierstrass::{Affine, Projective},
};

/// Trait defining the parameters of the Shallue-van de Woestijne (SVDW) hash-to-curve
/// method for curves of Weierstrass form `y^2 = x^3 + a*x + b`, including `a * b = 0`,
/// which [`SWUConfig`](super::swu::SWUConfig) excludes. From [\[RFC9380\]] section 6.6.1.
/// `find_z_svdw` in `curve_map_parameter_helper.sage` derives the constants.
///
/// - [\[RFC9380\]] <https://www.rfc-editor.org/rfc/rfc9380.html#section-6.6.1>
pub trait SVDWConfig: SWCurveConfig {
    /// `Z` meeting the four criteria of \[RFC9380, section 6.6.1\].
    const Z: Self::BaseField;

    /// `C1 = g(Z)` where `g(x) = x^3 + a*x + b`.
    const C1: Self::BaseField;

    /// `C2 = -Z / 2`.
    const C2: Self::BaseField;

    /// `C3 = sqrt(-g(Z) * (3 * Z^2 + 4 * a))` with `sgn0(C3) = 0`.
    const C3: Self::BaseField;

    /// `C4 = -4 * g(Z) / (3 * Z^2 + 4 * a)`.
    const C4: Self::BaseField;
}

/// Represents the SVDW hash-to-curve map defined by `P`.
pub struct SVDWMap<P: SVDWConfig>(PhantomData<fn() -> P>);

impl<P: SVDWConfig> MapToCurve<Projective<P>> for SVDWMap<P> {
    /// Checks the criteria of \[RFC9380, section 6.6.1\] on `Z` and that `C1..C4` derive from it.
    fn check_parameters() -> Result<(), HashToCurveError> {
        let err = |msg: &str| Err(HashToCurveError::MapToCurveError(msg.into()));
        let g = |x: P::BaseField| P::add_b(x.square() * x + P::mul_by_a(x));
        let z = P::Z;
        let gz = g(z);
        let four_gz = gz.double().double();
        let three_z2_plus_4a = z.square() * P::BaseField::from(3u8) + P::COEFF_A.double().double();

        if P::C1 != gz
            || P::C2.double() != -z
            || P::C3.square() != -gz * three_z2_plus_4a
            || parity(&P::C3)
            || P::C4 * three_z2_plus_4a != -four_gz
        {
            return err("SVDW constants C1..C4 do not match Z");
        }
        if gz.is_zero() || three_z2_plus_4a.is_zero() {
            return err("SVDW requires g(Z) != 0 and 3 * Z^2 + 4 * a != 0");
        }
        if !(-three_z2_plus_4a / four_gz).legendre().is_qr() {
            return err("SVDW requires -(3 * Z^2 + 4 * a) / (4 * g(Z)) to be a square");
        }
        if !gz.legendre().is_qr() && !g(P::C2).legendre().is_qr() {
            return err("SVDW requires g(Z) or g(-Z / 2) to be a square");
        }
        Ok(())
    }

    /// Straight-line SVDW of [\[RFC9380, appendix F.1\]](https://www.rfc-editor.org/rfc/rfc9380.html#appendix-F.1).
    /// Tries `x1`, `x2`, `x3` in order and stops at the first with `g(x)` square, using the
    /// square root as the square test. Returns the same point as the constant-time form.
    fn map_to_curve(u: P::BaseField) -> Result<Affine<P>, HashToCurveError> {
        let tv1 = u.square() * P::C1;
        let tv2 = P::BaseField::one() + tv1;
        let tv1 = P::BaseField::one() - tv1;
        // `inv0`, which maps 0 to 0.
        let tv3 = (tv1 * tv2).inverse().unwrap_or_else(P::BaseField::zero);
        let tv4 = u * tv1 * tv3 * P::C3;

        let with_ys = |x| Affine::<P>::get_ys_from_x_unchecked(x).map(|ys| (x, ys));
        let (x, (y, neg_y)) = with_ys(P::C2 - tv4)
            .or_else(|| with_ys(P::C2 + tv4))
            .or_else(|| with_ys((tv2.square() * tv3).square() * P::C4 + P::Z))
            .ok_or_else(|| {
                HashToCurveError::MapToCurveError("SVDW: g(x3) is not a square".into())
            })?;
        let y = if parity(&y) == parity(&u) { y } else { neg_y };
        Ok(Affine::new_unchecked(x, y))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hashing::{
        curve_maps::wb::test::{TestWBF127MapToCurveConfig as TestSVDWConfig, F127},
        map_to_curve_hasher::MapToCurveBasedHasher,
        HashToCurve,
    };
    use ark_ff::{field_hashers::DefaultFieldHasher, MontFp};
    use sha2::Sha256;

    /// `y^2 = x^3 + 3` over `F127`. `find_z_svdw(GF(127), 0, 3)` of
    /// `curve_map_parameter_helper.sage` gives `Z = 1`.
    impl SVDWConfig for TestSVDWConfig {
        const Z: F127 = MontFp!("1");
        const C1: F127 = MontFp!("4");
        const C2: F127 = MontFp!("63");
        const C3: F127 = MontFp!("78");
        const C4: F127 = MontFp!("37");
    }

    /// Maps the whole field and checks every image is on the curve with `sgn0(y) = sgn0(u)`,
    /// including `u = +-1/2` where `inv0` receives 0.
    #[test]
    fn map_field_to_curve_svdw() {
        SVDWMap::<TestSVDWConfig>::check_parameters().unwrap();

        let images: ark_std::vec::Vec<_> = (0..127u64)
            .map(|u| {
                let u = F127::from(u);
                let p = SVDWMap::<TestSVDWConfig>::map_to_curve(u).unwrap();
                assert!(p.is_on_curve());
                assert_eq!(parity(&p.y), parity(&u));
                p
            })
            .collect();
        assert!(
            images.iter().any(|p| *p != images[0]),
            "a constant hash function is not good."
        );
    }

    #[test]
    fn hash_arbitrary_string_to_curve_svdw() {
        let hasher = MapToCurveBasedHasher::<
            Projective<TestSVDWConfig>,
            DefaultFieldHasher<Sha256, 128>,
            SVDWMap<TestSVDWConfig>,
        >::new(&[1])
        .unwrap();

        let hash_result = hasher.hash(b"if you stick a Babel fish in your ear you can instantly understand anything said to you in any form of language.").expect("fail to hash the string to curve");
        assert!(hash_result.is_on_curve());
    }
}
