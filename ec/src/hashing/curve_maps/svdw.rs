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
    /// Nonzero `Z` meeting the four criteria of \[RFC9380, section 6.6.1\].
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
    /// Checks that `Z` is nonzero and meets the criteria of \[RFC9380, section 6.6.1\], and that
    /// `C1..C4` derive from it.
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
        if z.is_zero() {
            return err("SVDW requires Z != 0");
        }
        if gz.is_zero() || three_z2_plus_4a.is_zero() {
            return err("SVDW requires g(Z) != 0 and 3 * Z^2 + 4 * a != 0");
        }
        if !(-three_z2_plus_4a / four_gz).legendre().is_qr() {
            return err("SVDW requires -(3 * Z^2 + 4 * a) / (4 * g(Z)) to be a square");
        }
        if gz.legendre().is_qnr() && g(P::C2).legendre().is_qnr() {
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
    use crate::{
        hashing::{
            curve_maps::wb::test::{TestWBF127MapToCurveConfig as TestSVDWConfig, F127},
            map_to_curve_hasher::MapToCurveBasedHasher,
            HashToCurve,
        },
        CurveConfig,
    };
    use ark_ff::{field_hashers::DefaultFieldHasher, fields::Fp64, MontBackend, MontFp};
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

    #[derive(ark_ff::MontConfig)]
    #[modulus = "11"]
    #[generator = "2"]
    struct F11Config;
    type F11 = Fp64<MontBackend<F11Config, 1>>;

    /// `y^2 = x^3 + 7` over `F11` with `Z = 1`. `g(Z) = 8` is a nonsquare and `g(-Z / 2) = 0`.
    struct TestSVDWZeroGC2Config;

    impl CurveConfig for TestSVDWZeroGC2Config {
        const COFACTOR: &[u64] = &[1];
        const COFACTOR_INV: F11 = MontFp!("1");
        type BaseField = F11;
        type ScalarField = F11;
    }

    impl SWCurveConfig for TestSVDWZeroGC2Config {
        const COEFF_A: F11 = MontFp!("0");
        const COEFF_B: F11 = MontFp!("7");
        const GENERATOR: Affine<Self> = Affine::new_unchecked(MontFp!("2"), MontFp!("2"));
        type ZeroFlag = ();
    }

    impl SVDWConfig for TestSVDWZeroGC2Config {
        const Z: F11 = MontFp!("1");
        const C1: F11 = MontFp!("8");
        const C2: F11 = MontFp!("5");
        const C3: F11 = MontFp!("8");
        const C4: F11 = MontFp!("4");
    }

    /// `y^2 = x^3 + 2 * x + 1` over `F11` with `Z = 0`, which meets the four criteria.
    struct TestSVDWZeroZConfig;

    impl CurveConfig for TestSVDWZeroZConfig {
        const COFACTOR: &[u64] = &[1];
        const COFACTOR_INV: F11 = MontFp!("1");
        type BaseField = F11;
        type ScalarField = F11;
    }

    impl SWCurveConfig for TestSVDWZeroZConfig {
        const COEFF_A: F11 = MontFp!("2");
        const COEFF_B: F11 = MontFp!("1");
        const GENERATOR: Affine<Self> = Affine::new_unchecked(MontFp!("1"), MontFp!("2"));
        type ZeroFlag = ();
    }

    impl SVDWConfig for TestSVDWZeroZConfig {
        const Z: F11 = MontFp!("0");
        const C1: F11 = MontFp!("1");
        const C2: F11 = MontFp!("0");
        const C3: F11 = MontFp!("6");
        const C4: F11 = MontFp!("5");
    }

    #[test]
    fn check_parameters_rejects_zero_z() {
        assert!(matches!(
            SVDWMap::<TestSVDWZeroZConfig>::check_parameters(),
            Err(HashToCurveError::MapToCurveError(msg)) if msg == "SVDW requires Z != 0"
        ));
    }

    /// Criterion 4 counts 0 as a square, so `g(-Z / 2) = 0` passes and every input maps. Inputs
    /// mapping to `x = -Z / 2` get `y = 0`, whose sign cannot follow `u`.
    #[test]
    fn check_parameters_accepts_zero_g_c2() {
        SVDWMap::<TestSVDWZeroGC2Config>::check_parameters().unwrap();
        for u in 0..11u64 {
            let u = F11::from(u);
            let p = SVDWMap::<TestSVDWZeroGC2Config>::map_to_curve(u).unwrap();
            assert!(p.is_on_curve());
            assert!(p.y.is_zero() || parity(&p.y) == parity(&u));
        }
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
