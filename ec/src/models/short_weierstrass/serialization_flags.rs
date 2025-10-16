use ark_ff::Field;
use ark_serialize::Flags;

/// Flags to be encoded into the serialization.
/// The default flags (empty) should not change the binary representation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SWFlags {
    /// Represents a point with positive y-coordinate by setting all bits to 0.
    YIsPositive = 0,
    /// Represents the point at infinity by setting the setting the last-but-one bit to 1.
    PointAtInfinity = 1 << 6,
    /// Represents a point with negative y-coordinate by setting the MSB to 1.
    YIsNegative = 1 << 7,
}

impl SWFlags {
    #[inline]
    pub const fn infinity() -> Self {
        Self::PointAtInfinity
    }

    #[inline]
    pub fn from_y_coordinate(y: impl Field) -> Self {
        if y <= -y {
            Self::YIsPositive
        } else {
            Self::YIsNegative
        }
    }

    #[inline]
    pub const fn is_infinity(&self) -> bool {
        matches!(self, Self::PointAtInfinity)
    }

    #[inline]
    pub const fn is_positive(&self) -> Option<bool> {
        match self {
            Self::PointAtInfinity => None,
            Self::YIsPositive => Some(true),
            Self::YIsNegative => Some(false),
        }
    }
}

impl Default for SWFlags {
    #[inline]
    fn default() -> Self {
        // YIsNegative doesn't change the serialization
        // Question: How? YIsNegative is 10000000.
        Self::YIsNegative
    }
}

impl Flags for SWFlags {
    const BIT_SIZE: usize = 2;

    #[inline]
    fn u8_bitmask(&self) -> u8 {
        let mut mask = 0;
        match self {
            Self::PointAtInfinity => mask |= 1 << 6,
            Self::YIsNegative => mask |= 1 << 7,
            _ => (),
        }
        mask
    }

    #[inline]
    fn from_u8(value: u8) -> Option<Self> {
        let is_negative = (value >> 7) & 1 == 1;
        let is_infinity = (value >> 6) & 1 == 1;
        match (is_negative, is_infinity) {
            // This is invalid because we only want *one* way to serialize
            // the point at infinity.
            (true, true) => None,
            (false, true) => Some(Self::PointAtInfinity),
            (true, false) => Some(Self::YIsNegative),
            (false, false) => Some(Self::YIsPositive),
        }
    }
}

/// Flags to be encoded into the serialization.
/// The default flags (empty) should not change the binary representation.
/// This is a single bit flag and used when x = 0 is not valid point which makes all 0s to be a representation
/// of the point at infinity and the x coordinates's byte representation has 1 bit unused, eg. a 255 bit
/// field used for Pallas, Vesta curves. That 1 unused bit can then hold this flag
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SingleBitSWFlags {
    /// Represents a point with positive y-coordinate by setting the MSB to 0.
    YIsPositive = 0,
    /// Represents a point with negative y-coordinate by setting the MSB to 1.
    YIsNegative = 1 << 7,
}

impl SingleBitSWFlags {
    #[inline]
    pub const fn infinity() -> Self {
        // Since this doesnt change the serialization
        Self::YIsPositive
    }
    
    #[inline]
    pub fn from_y_coordinate(y: impl Field) -> Self {
        if y <= -y {
            Self::YIsPositive
        } else {
            Self::YIsNegative
        }
    }

    #[inline]
    pub fn is_positive(&self) -> bool {
        match self {
            SingleBitSWFlags::YIsPositive => true,
            SingleBitSWFlags::YIsNegative => false,
        }
    }
}

impl Default for SingleBitSWFlags {
    #[inline]
    fn default() -> Self {
        // YIsPositive doesn't change the serialization
        SingleBitSWFlags::YIsPositive
    }
}

impl Flags for SingleBitSWFlags {
    const BIT_SIZE: usize = 1;

    #[inline]
    fn u8_bitmask(&self) -> u8 {
        let mut mask = 0;
        match self {
            // Set MSB as 1
            SingleBitSWFlags::YIsNegative => mask |= 1 << 7,
            _ => (),
        }
        mask
    }

    #[inline]
    fn from_u8(value: u8) -> Option<Self> {
        // true if MSB is 1
        let is_msb_1 = (value >> 7) & 1 == 1;
        match is_msb_1 {
            true => Some(SingleBitSWFlags::YIsNegative),
            false => Some(SingleBitSWFlags::YIsPositive),
        }
    }
}
