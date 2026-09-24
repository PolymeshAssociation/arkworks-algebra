#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::short_weierstrass::Affine;
pub use ark_host_hash_to_curve::{
    batch_serialized_size, curve_id, BatchHashToCurveRequest, HashToCurveConfig,
};
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_pallas::PallasConfig;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::boxed::Box;
use ark_std::collections::BTreeMap;
use ark_std::vec::Vec;
use ark_vesta::VestaConfig;

type HashToCurveFn = Box<dyn Fn(&mut [u8], usize) -> u32 + Send + Sync>;

pub struct RegisteredCurves {
    curves: BTreeMap<CurveMSMId, HashToCurveFn>,
}

impl RegisteredCurves {
    pub fn new() -> Self {
        let mut curves = RegisteredCurves {
            curves: BTreeMap::new(),
        };
        curves.register_curve::<PallasConfig>();
        curves.register_curve::<VestaConfig>();
        curves
    }

    pub fn register_curve<C: HashToCurveConfig + 'static>(&mut self) -> bool {
        self.curves
            .insert(curve_id::<C>(), Box::new(batch_hash_to_curve_impl::<C>));
        true
    }

    /// Returns 1 for a buffer holding only the ID of a supported curve, the length of the result
    /// written to the start of `buffer` for a valid request, and 0 otherwise.
    pub fn batch_hash_to_curve(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
        let buf_len = buf_len as usize;
        if buf_len < CURVE_ID_LEN || buf_len > buffer.len() {
            return 0;
        }
        let Ok(curve_id) = CurveMSMId::deserialize_uncompressed_unchecked(&buffer[..CURVE_ID_LEN])
        else {
            return 0;
        };
        match self.curves.get(&curve_id) {
            Some(_) if buf_len == CURVE_ID_LEN => 1,
            Some(hash_fn) => hash_fn(buffer, buf_len),
            None => 0,
        }
    }
}

#[cfg(feature = "std")]
lazy_static::lazy_static! {
    pub static ref SUPPORTED_CURVES: RegisteredCurves = {
        RegisteredCurves::new()
    };
}

#[cfg(not(feature = "std"))]
static mut SUPPORTED_CURVES: Option<RegisteredCurves> = None;

#[cfg(not(feature = "std"))]
#[allow(static_mut_refs)]
fn get_supported_curves() -> &'static RegisteredCurves {
    unsafe {
        if SUPPORTED_CURVES.is_none() {
            SUPPORTED_CURVES = Some(RegisteredCurves::new());
        }
        SUPPORTED_CURVES.as_ref().unwrap()
    }
}

/// The host function registered under the extern name `host_batch_hash_to_curve`.
#[cfg(feature = "std")]
pub fn host_batch_hash_to_curve(buffer: &mut [u8], buf_len: u32) -> u32 {
    SUPPORTED_CURVES.batch_hash_to_curve(buffer, buf_len)
}

/// The host function registered under the extern name `host_batch_hash_to_curve`.
#[cfg(not(feature = "std"))]
pub fn host_batch_hash_to_curve(buffer: &mut [u8], buf_len: u32) -> u32 {
    get_supported_curves().batch_hash_to_curve(buffer, buf_len)
}

/// Decodes the `BatchHashToCurveRequest` in `buffer[CURVE_ID_LEN..buf_len]`, hashes, and writes the
/// uncompressed points to the start of `buffer`. Returns the result length, or 0 when the request
/// does not decode, `gens_offset + gens_count` overflows, or the result does not fit in `buf_len`.
fn batch_hash_to_curve_impl<C: HashToCurveConfig>(buffer: &mut [u8], buf_len: usize) -> u32 {
    let Ok(req) =
        BatchHashToCurveRequest::deserialize_uncompressed(&buffer[CURVE_ID_LEN..buf_len])
    else {
        return 0;
    };
    if req.gens_offset.checked_add(req.gens_count).is_none()
        || batch_serialized_size::<C>(req.gens_count) > buf_len
    {
        return 0;
    }
    let res: Vec<Affine<C>> =
        C::batch_hash_to_curve(&req.dst, &req.msg_prefix, req.gens_offset, req.gens_count);
    let res_len = res.serialized_size(Compress::No);
    if res_len > buf_len {
        return 0;
    }
    match res.serialize_uncompressed(&mut buffer[..res_len]) {
        Ok(()) => res_len as u32,
        Err(_) => 0,
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use ark_std::vec;

    fn request_buffer<C: HashToCurveConfig>(req: &BatchHashToCurveRequest) -> Vec<u8> {
        let mut buffer = Vec::new();
        curve_id::<C>().serialize_uncompressed(&mut buffer).unwrap();
        req.serialize_uncompressed(&mut buffer).unwrap();
        let len = buffer.len().max(batch_serialized_size::<C>(req.gens_count));
        buffer.resize(len, 0);
        buffer
    }

    fn request(gens_offset: u32, gens_count: u32) -> BatchHashToCurveRequest {
        BatchHashToCurveRequest {
            dst: b"BulletproofGens-G\0\0\0\0".to_vec(),
            msg_prefix: b"test".to_vec(),
            gens_offset,
            gens_count,
        }
    }

    fn round_trip<C: HashToCurveConfig>() {
        for (offset, count) in [(0u32, 1u32), (0, 16), (100, 33)] {
            let req = request(offset, count);
            let mut buffer = request_buffer::<C>(&req);
            let len = buffer.len() as u32;
            let res_len = host_batch_hash_to_curve(&mut buffer, len) as usize;
            assert!(res_len > 0);
            let got = Vec::<Affine<C>>::deserialize_uncompressed(&buffer[..res_len]).unwrap();
            assert_eq!(got, C::batch_hash_to_curve(&req.dst, &req.msg_prefix, offset, count));
        }
    }

    #[test]
    fn round_trip_matches_local() {
        round_trip::<PallasConfig>();
        round_trip::<VestaConfig>();
    }

    #[test]
    fn probe() {
        for id in [curve_id::<PallasConfig>(), curve_id::<VestaConfig>()] {
            let mut buffer = Vec::new();
            id.serialize_uncompressed(&mut buffer).unwrap();
            assert_eq!(host_batch_hash_to_curve(&mut buffer, CURVE_ID_LEN as u32), 1);
        }

        let mut buffer = Vec::new();
        CurveMSMId::from_curve_name("unregistered_curve")
            .serialize_uncompressed(&mut buffer)
            .unwrap();
        assert_eq!(host_batch_hash_to_curve(&mut buffer, CURVE_ID_LEN as u32), 0);

        let mut short = vec![0u8; CURVE_ID_LEN - 1];
        assert_eq!(host_batch_hash_to_curve(&mut short, (CURVE_ID_LEN - 1) as u32), 0);
    }

    #[test]
    fn malformed_requests_return_zero() {
        // `buf_len` larger than the buffer.
        let mut buffer = request_buffer::<PallasConfig>(&request(0, 1));
        let len = buffer.len() as u32 + 1;
        assert_eq!(host_batch_hash_to_curve(&mut buffer, len), 0);

        // Truncated request.
        let mut buffer = Vec::new();
        curve_id::<PallasConfig>().serialize_uncompressed(&mut buffer).unwrap();
        buffer.extend_from_slice(&[5, 1, 2]);
        let len = buffer.len() as u32;
        assert_eq!(host_batch_hash_to_curve(&mut buffer, len), 0);

        // `gens_offset + gens_count` overflows.
        let mut buffer = request_buffer::<PallasConfig>(&request(u32::MAX, 2));
        let len = buffer.len() as u32;
        assert_eq!(host_batch_hash_to_curve(&mut buffer, len), 0);

        // Buffer too small for the result.
        let mut buffer = Vec::new();
        curve_id::<PallasConfig>().serialize_uncompressed(&mut buffer).unwrap();
        request(0, 16).serialize_uncompressed(&mut buffer).unwrap();
        let len = buffer.len() as u32;
        assert_eq!(host_batch_hash_to_curve(&mut buffer, len), 0);
    }
}
