use ark_std::io::{Read, Write};
use parity_scale_codec::{Compact, Decode, Encode, Error as ScaleError, Input, Output};
use crate::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompactU64(pub u64);

impl ark_std::ops::Deref for CompactU64 {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for CompactU64 {
    fn from(value: usize) -> Self {
        Self(value as u64)
    }
}

impl From<u64> for CompactU64 {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

struct WriteOutput<T: Write>(T);

impl<T: Write> Output for WriteOutput<T> {
    fn write(&mut self, bytes: &[u8]) {
        let res = self.0.write_all(bytes);
        debug_assert!(res.is_ok());
    }
}

struct ReadInput<T: Read>(T);

impl<T: Read> Input for ReadInput<T> {
    fn remaining_len(&mut self) -> Result<Option<usize>, ScaleError> {
        Ok(None)
    }

    fn read(&mut self, into: &mut [u8]) -> Result<(), ScaleError> {
        self.0
            .read_exact(into)
            .map_err(|_| ScaleError::from("Read error"))
    }
}

impl Valid for CompactU64 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl CanonicalSerialize for CompactU64 {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        let mut out = WriteOutput(writer);

        Compact(self.0).encode_to(&mut out);
        Ok(())
    }

    #[inline]
    fn serialized_size(&self, _compress: Compress) -> usize {
        Compact(self.0).size_hint()
    }
}

impl CanonicalDeserialize for CompactU64 {
    #[inline]
    fn deserialize_with_mode<R: Read>(
        reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut reader = ReadInput(reader);
        let len = Compact::<u64>::decode(&mut reader)
            .map_err(|_| SerializationError::InvalidData)?
            .0;
        Ok(Self(len))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::vec::Vec;

    /// SCALE compact vectors either side of each of the four mode boundaries. These are the fork's
    /// wire format, so they are spelled out rather than derived from the encoder under test, and
    /// are checkable against the mode table in
    /// [`parity_scale_codec::Compact`](https://docs.rs/parity-scale-codec/3/parity_scale_codec/struct.Compact.html).
    const VECTORS: &[(u64, &[u8])] = &[
        (0, &[0x00]),
        (1, &[0x04]),
        (42, &[0xa8]),
        (63, &[0xfc]),
        (64, &[0x01, 0x01]),
        (69, &[0x15, 0x01]),
        (16383, &[0xfd, 0xff]),
        (16384, &[0x02, 0x00, 0x01, 0x00]),
        (1073741823, &[0xfe, 0xff, 0xff, 0xff]),
        (1073741824, &[0x03, 0x00, 0x00, 0x00, 0x40]),
        (u64::MAX, &[0x13, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
    ];

    #[test]
    fn compact_encoding_works() {
        for (value, expected) in VECTORS {
            for compress in [Compress::Yes, Compress::No] {
                let mut bytes = Vec::new();
                CompactU64(*value)
                    .serialize_with_mode(&mut bytes, compress)
                    .unwrap();
                assert_eq!(bytes, *expected, "value = {value}");
            }
        }
    }

    #[test]
    fn serialized_size_is_exact() {
        for (value, expected) in VECTORS {
            assert_eq!(
                CompactU64(*value).serialized_size(Compress::Yes),
                expected.len(),
                "value = {value}"
            );
        }
    }

    #[test]
    fn round_trips() {
        for (value, _) in VECTORS {
            let mut bytes = Vec::new();
            CompactU64(*value)
                .serialize_with_mode(&mut bytes, Compress::Yes)
                .unwrap();
            let read =
                CompactU64::deserialize_with_mode(&bytes[..], Compress::Yes, Validate::Yes).unwrap();
            assert_eq!(read.0, *value);
        }
    }

    #[test]
    fn slice_works() {
        let payload = [7u8; 69];
        let mut bytes = Vec::new();
        payload
            .as_slice()
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .unwrap();
        // Check in `VECTORS`
        assert_eq!(&bytes[..2], &[0x15, 0x01]);
        assert_eq!(&bytes[2..], &payload);
    }
}
