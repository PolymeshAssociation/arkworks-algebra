use crate::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::io::{Read, Write};
use parity_scale_codec::{Compact, Decode, Encode, Error as ScaleError, Input, Output};

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

/// Adapter from `ark_std::io::Write` to SCALE's infallible `Output`. Any write error is
/// recorded and surfaced by `serialize_with_mode` (SCALE's `Output::write` cannot fail, so the
/// error must be carried out of band instead of being asserted away).
struct WriteOutput<T: Write> {
    writer: T,
    error: Option<ark_std::io::Error>,
}

impl<T: Write> Output for WriteOutput<T> {
    fn write(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        if let Err(e) = self.writer.write_all(bytes) {
            self.error = Some(e);
        }
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
        let mut out = WriteOutput {
            writer,
            error: None,
        };

        Compact(self.0).encode_to(&mut out);
        match out.error {
            None => Ok(()),
            Some(e) => Err(SerializationError::IoError(e)),
        }
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
