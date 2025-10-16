use ark_ec::{models::CurveConfig, scalar_mul::glv::GLVConfig, short_weierstrass::{self as sw, SWCurveConfig}, AffineRepr};
use ark_ec::short_weierstrass::SingleBitSWFlags;
use ark_ff::{AdditiveGroup, BigInt, Field, MontFp, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize, CanonicalSerializeWithFlags, Compress, SerializationError, Valid, Validate};
use ark_std::io::{Write, Read};
use crate::{fq::Fq, fr::Fr};

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct PallasConfig;

impl CurveConfig for PallasConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

pub type Affine = sw::Affine<PallasConfig>;
pub type Projective = sw::Projective<PallasConfig>;

impl SWCurveConfig for PallasConfig {
    /// COEFF_A = 0
    const COEFF_A: Fq = Fq::ZERO;

    /// COEFF_B = 5
    const COEFF_B: Fq = MontFp!("5");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: Affine = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);

    /// Correctness:
    /// The curve equation is y^2 = x^3 + b
    /// Substituting (0, 0) gives 0^2 = 0^3 + b which simplifies to 0 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();

    #[inline(always)]
    fn mul_by_a(_: Self::BaseField) -> Self::BaseField {
        Self::BaseField::zero()
    }

    // Following serialization methods could be moved to SWCurveConfig or a trait extending it since
    // these can be used by multiple curves where x=0 is not a valid point

    /// If uncompressed, serializes both x and y coordinates as well as a bit for whether it is
    /// infinity. If compressed, serializes x coordinate with 1 bit to encode whether y is
    /// positive, negative. x=0 means infinity as x=0 is not a valid point
    #[inline]
    fn serialize_with_mode<W: Write>(
        item: &Affine,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (x, y, flags) = match item.is_zero() {
            true => (
                Self::BaseField::zero(),
                Self::BaseField::zero(),
                SingleBitSWFlags::infinity(),
            ),
            false => (item.x, item.y, item.to_single_bit_flags()),
        };

        match compress {
            Compress::Yes => x.serialize_with_flags(writer, flags),
            Compress::No => {
                x.serialize_with_mode(&mut writer, compress)?;
                y.serialize_with_flags(&mut writer, flags)
            },
        }
    }

    /// If `validate` is `Yes`, calls `check()` to make sure the element is valid.
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Affine, SerializationError> {
        let mut is_infinity = false;
        let (x, y, ) = match compress {
            Compress::Yes => {
                let (x, flags): (Self::BaseField, SingleBitSWFlags) =
                    CanonicalDeserializeWithFlags::deserialize_with_flags(reader)?;
                if x.is_zero() {
                    is_infinity = true;
                    let identity = Affine::identity();
                    (identity.x, identity.y)
                } else {
                    let (y, neg_y) = Affine::get_ys_from_x_unchecked(x).ok_or(SerializationError::InvalidData)?;
                    let is_positive = flags.is_positive();
                    if is_positive {
                        (x, y)
                    } else {
                        (x, neg_y)
                    }
                }
            },
            Compress::No => {
                let x: Self::BaseField =
                    CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
                let (y, _): (_, SingleBitSWFlags) =
                    CanonicalDeserializeWithFlags::deserialize_with_flags(&mut reader)?;
                if x.is_zero() {
                    is_infinity = true;
                }
                (x, y)
            },
        };
        if is_infinity {
            Ok(Affine::identity())
        } else {
            let point = Affine::new_unchecked(x, y);
            if validate == Validate::Yes {
                point.check()?;
            }
            Ok(point)
        }
    }

    #[inline]
    fn serialized_size(compress: Compress) -> usize {
        let zero = Self::BaseField::zero();
        match compress {
            Compress::Yes => zero.serialized_size_with_flags::<SingleBitSWFlags>(),
            Compress::No => zero.compressed_size() + zero.serialized_size_with_flags::<SingleBitSWFlags>(),
        }
    }
}

impl GLVConfig for PallasConfig {
    const ENDO_COEFFS: &'static [Self::BaseField] = &[MontFp!(
        "20444556541222657078399132219657928148671392403212669005631716460534733845831"
    )];

    const LAMBDA: Self::ScalarField =
        MontFp!("26005156700822196841419187675678338661165322343552424574062261873906994770353");

    const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as PrimeField>::BigInt); 4] = [
        (false, BigInt!("98231058071100081932162823354453065728")),
        (true, BigInt!("98231058071186745657228807397848383489")),
        (false, BigInt!("196462116142286827589391630752301449217")),
        (false, BigInt!("98231058071100081932162823354453065728")),
    ];

    fn endomorphism(p: &Projective) -> Projective {
        // Endomorphism of the points on the curve.
        // endomorphism_p(x,y) = (BETA * x, y)
        // where BETA is a non-trivial cubic root of unity in Fq.
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }

    fn endomorphism_affine(p: &Affine) -> Affine {
        // Endomorphism of the points on the curve.
        // endomorphism_p(x,y) = (BETA * x, y)
        // where BETA is a non-trivial cubic root of unity in Fq.
        let mut res = (*p).clone();
        res.x *= Self::ENDO_COEFFS[0];
        res
    }
}

/// G_GENERATOR_X = -1
pub const G_GENERATOR_X: Fq = MontFp!("-1");

/// G_GENERATOR_Y = 2
pub const G_GENERATOR_Y: Fq = MontFp!("2");
