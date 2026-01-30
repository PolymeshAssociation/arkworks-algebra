use ark_serialize::{
    CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, SerializationError, Valid, Validate,
};
use ark_std::io::{Read, Write};
use ark_ff::Zero;

use crate::AffineRepr;
use super::{Affine, SWCurveConfig, SingleBitSWFlags};

/// Marker trait for Short Weierstrass curves where x=0 is not on the curve.
/// Only curves that implement SWSerializationXNonZero can use this serialization
/// 
/// This enables more efficient serialization using `SingleBitSWFlags` instead of the
/// default `SWFlags`. When a curve satisfies the property that (0, 0) is not on the curve,
/// we can use (0, 0) as an encoding for the point at infinity, requiring only 1 bit
/// instead of 2 bits for the flags.
///
/// Ensure that x=0 is not a valid x-coordinate for any point on the curve.
pub trait SWSerializationXNonZero: SWCurveConfig {}

/// Serializes an affine point using `SingleBitSWFlags`.
/// 
/// If uncompressed, serializes both x and y coordinates as well as a bit for whether it is
/// infinity. If compressed, serializes x coordinate with 1 bit to encode whether y is
/// positive or negative. x=0 means infinity as x=0 is not a valid point.
#[inline]
pub fn serialize_with_single_bit_flags<C: SWSerializationXNonZero, W: Write>(
    item: &Affine<C>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError> {
    let (x, y, flags) = match item.is_zero() {
        true => (
            C::BaseField::zero(),
            C::BaseField::zero(),
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

/// Deserializes an affine point using `SingleBitSWFlags`.
/// 
/// If `validate` is `Yes`, calls `check()` to make sure the element is valid.
pub fn deserialize_with_single_bit_flags<C: SWSerializationXNonZero, R: Read>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
) -> Result<Affine<C>, SerializationError> {
    let mut is_infinity = false;
    let (x, y) = match compress {
        Compress::Yes => {
            let (x, flags): (C::BaseField, SingleBitSWFlags) =
                CanonicalDeserializeWithFlags::deserialize_with_flags(reader)?;
            if x.is_zero() {
                is_infinity = true;
                let identity = Affine::<C>::identity();
                (identity.x, identity.y)
            } else {
                let (y, neg_y) = Affine::<C>::get_ys_from_x_unchecked(x)
                    .ok_or(SerializationError::InvalidData)?;
                let is_positive = flags.is_positive();
                if is_positive {
                    (x, y)
                } else {
                    (x, neg_y)
                }
            }
        },
        Compress::No => {
            let x: C::BaseField =
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

/// Returns the serialized size using `SingleBitSWFlags`.
#[inline]
pub fn serialized_size_with_single_bit_flags<C: SWSerializationXNonZero>(
    compress: Compress,
) -> usize {
    let zero = C::BaseField::zero();
    match compress {
        Compress::Yes => zero.serialized_size_with_flags::<SingleBitSWFlags>(),
        Compress::No => zero.compressed_size() + zero.serialized_size_with_flags::<SingleBitSWFlags>(),
    }
}
