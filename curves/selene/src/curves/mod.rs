use ark_ec::{models::CurveConfig, short_weierstrass::{self as sw, SWCurveConfig, SWSerializationXNonZero}};
use ark_ff::{AdditiveGroup, Field, MontFp};
use ark_serialize::{Compress, SerializationError, Validate};
use ark_std::io::{Read, Write};
use ark_ec::hashing::curve_maps::swu::SWUConfig;
use crate::{fq::Fq, fr::Fr};

#[cfg(test)]
mod tests;


/// Selene equation: y^2 = x^3 - 3*x + 25675911719867737339625140396204798989996478324626569376465022644547366285284
/// Info about these curves https://gist.github.com/tevador/4524c2092178df08996487d4e272b096
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct SeleneConfig;

impl CurveConfig for SeleneConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

pub type Affine = sw::Affine<SeleneConfig>;
pub type Projective = sw::Projective<SeleneConfig>;

impl SWCurveConfig for SeleneConfig {
    /// COEFF_A = -3
    /// Like Helios, Selene uses a = -3 for optimized arithmetic
    const COEFF_A: Fq = MontFp!("-3");

    /// COEFF_B for Selene
    const COEFF_B: Fq = MontFp!("25675911719867737339625140396204798989996478324626569376465022644547366285284");

    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: Affine = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);

    /// Correctness:
    /// The curve equation is y^2 = x^3 -3*x + b
    /// Substituting (0, 0) gives 0^2 = 0^3 + 0 + b which simplifies to 0 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();

    #[inline(always)]
    fn mul_by_a(elem: Self::BaseField) -> Self::BaseField {
        // Multiplication by -3
        let double = elem.double();
        -(double + elem)
    }

    #[inline]
    fn serialize_with_mode<W: Write>(
        item: &Affine,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        sw::serialize_with_single_bit_flags(item, writer, compress)
    }

    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Affine, SerializationError> {
        sw::deserialize_with_single_bit_flags(reader, compress, validate)
    }

    #[inline]
    fn serialized_size(compress: Compress) -> usize {
        sw::serialized_size_with_single_bit_flags::<Self>(compress)
    }
}

impl SWSerializationXNonZero for SeleneConfig {}

impl SWUConfig for SeleneConfig {
    const ZETA: Self::BaseField = MontFp!("6");
}

/// G_GENERATOR_X = 1
/// Both Helios and Selene use x=1 as the generator's x-coordinate
/// This is a deliberate design choice that simplifies implementation
pub const G_GENERATOR_X: Fq = MontFp!("1");

/// G_GENERATOR_Y = 25798700515841442074436724357845010259191889815205036617843407906692357567936
pub const G_GENERATOR_Y: Fq = MontFp!("25798700515841442074436724357845010259191889815205036617843407906692357567936");