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