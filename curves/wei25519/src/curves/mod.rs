use ark_ec::{models::CurveConfig, short_weierstrass::{self as sw, SWCurveConfig, SWSerializationXNonZero}};
use ark_ff::MontFp;
use ark_serialize::{Compress, SerializationError, Validate};
use ark_std::io::{Read, Write};
use ark_ec::hashing::curve_maps::swu::SWUConfig;
use crate::{Fq, Fr};

#[cfg(test)]
mod tests;

/// Wei25519 equation: y^2 = x^3 + A*x + B
/// Reference: https://datatracker.ietf.org/doc/html/draft-ietf-lwig-curve-representations-19
/// This is a short Weierstrass form of curve25519 with the same base and scalar fields.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub struct Wei25519Config;

impl CurveConfig for Wei25519Config {
    type BaseField = Fq;
    type ScalarField = Fr;

    /// COFACTOR = 8
    const COFACTOR: &'static [u64] = &[8];

    /// COFACTOR_INV (mod r) =
    /// 2713877091499598330239944961141122840321418634767465352250731601857045344121
    const COFACTOR_INV: Fr =
        MontFp!("2713877091499598330239944961141122840321418634767465352250731601857045344121");
}

pub type Affine = sw::Affine<Wei25519Config>;
pub type Projective = sw::Projective<Wei25519Config>;

impl SWCurveConfig for Wei25519Config {
    /// COEFF_A = 19298681539552699237261830834781317975544997444273427339909597334573241639236
    const COEFF_A: Fq = MontFp!("19298681539552699237261830834781317975544997444273427339909597334573241639236");

    /// COEFF_B = 55751746669818908907645289078257140818241103727901012315294400837956729358436
    const COEFF_B: Fq = MontFp!("55751746669818908907645289078257140818241103727901012315294400837956729358436");

    /// AFFINE_GENERATOR_COEFFS = (G_GENERATOR_X, G_GENERATOR_Y)
    const GENERATOR: Affine = Affine::new_unchecked(G_GENERATOR_X, G_GENERATOR_Y);

    /// Correctness:
    /// The curve equation is y^2 = x^3 -3*x + b
    /// Substituting (0, 0) gives 0^2 = 0^3 + 0 + b which simplifies to 0 = b.
    /// Since b is not zero, the point (0, 0) is not on the curve.
    /// Therefore, we can safely use (0, 0) as a flag for the zero point.
    type ZeroFlag = ();

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

impl SWSerializationXNonZero for Wei25519Config {}

impl SWUConfig for Wei25519Config {
    const ZETA: Self::BaseField = MontFp!("8");
}

/// G_GENERATOR_X = 19298681539552699237261830834781317975544997444273427339909597334652188435546
pub const G_GENERATOR_X: Fq = MontFp!("19298681539552699237261830834781317975544997444273427339909597334652188435546");

/// G_GENERATOR_Y = 14781619447589544791020593568409986887264606134616475288964881837755586237401
pub const G_GENERATOR_Y: Fq = MontFp!("14781619447589544791020593568409986887264606134616475288964881837755586237401");
