use ark_serialize::{
    CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, Compress, SerializationError, Valid, Validate,
};
use ark_std::io::{Read, Write};
use ark_ff::Zero;

use crate::AffineRepr;
use super::{Affine, SWCurveConfig, SingleBitSWFlags};

/// Marker trait for short Weierstrass curves whose prime-order subgroup has no point with
/// `x = 0`. Such curves serialize with `SingleBitSWFlags` instead of `SWFlags`, encoding the
/// point at infinity as `x = 0`, so the flags need 1 bit instead of 2. A curve may still have
/// points with `x = 0` outside the subgroup when `b` is a square (Wei25519); those fail to
/// serialize rather than encode as infinity.
pub trait SWSerializationXNonZero: SWCurveConfig {}

/// Serializes an affine point using `SingleBitSWFlags`.
/// 
/// If uncompressed, serializes both x and y coordinates as well as a bit for whether it is
/// infinity. If compressed, serializes x coordinate with 1 bit to encode whether y is
/// positive or negative. `x = 0` encodes infinity; a finite point with `x = 0` returns
/// `SerializationError::InvalidData`.
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
        false if item.x.is_zero() => return Err(SerializationError::InvalidData),
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
