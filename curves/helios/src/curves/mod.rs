use crate::{fq::Fq, fr::Fr};
use ark_ec::{models::CurveConfig, short_weierstrass::{self as sw, SWCurveConfig, SWSerializationXNonZero}};
use ark_ff::{AdditiveGroup, Field, MontFp};
use ark_serialize::{Compress, SerializationError, Validate};
use ark_std::io::{Read, Write};
use ark_ec::hashing::curve_maps::swu::SWUConfig;

#[cfg(test)]
mod tests;

/// Helios equation: y^2 = x^3 - 3*x + 17523451383230374900436292617863907649717438939964238673872692863501483215968
/// Info about these curves https://gist.github.com/tevador/4524c2092178df08996487d4e272b096
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct HeliosConfig;

impl CurveConfig for HeliosConfig {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 1
    const COFACTOR: &'static [u64] = &[0x1];

    /// COFACTOR_INV = 1
    const COFACTOR_INV: Fr = Fr::ONE;
}

pub type Affine = sw::Affine<HeliosConfig>;
pub type Projective = sw::Projective<HeliosConfig>;

impl SWCurveConfig for HeliosConfig {
    /// COEFF_A = -3
    const COEFF_A: Fq = MontFp!("-3");

    /// COEFF_B for Helios
    const COEFF_B: Fq = MontFp!("17523451383230374900436292617863907649717438939964238673872692863501483215968");


    /// AFFINE_GENERATOR_COEFFS = (G1_GENERATOR_X, G1_GENERATOR_Y)
    const GENERATOR: Affine = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);

    /// Correctness:
    /// Substituting (0, 0) into the curve equation gives 0^2 = b.
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

impl SWSerializationXNonZero for HeliosConfig {}

impl SWUConfig for HeliosConfig {
    const ZETA: Self::BaseField = MontFp!("8");
}

/// G_GENERATOR_X = 1
pub const G_GENERATOR_X: Fq = MontFp!("1");

/// G_GENERATOR_Y = 43927350165885181914572701368652294970994947138804342515295004363921039321018
pub const G_GENERATOR_Y: Fq = MontFp!("43927350165885181914572701368652294970994947138804342515295004363921039321018");
