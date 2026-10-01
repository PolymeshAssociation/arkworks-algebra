#![cfg_attr(not(feature = "std"), no_std)]

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::vec::Vec;

pub const CURVE_ID_LEN: usize = 32;

/// A wrapper around a fixed-size byte array to represent the curve ID for host MSM.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    CanonicalSerialize,
    CanonicalDeserialize,
)]
pub struct CurveMSMId([u8; CURVE_ID_LEN]);

impl CurveMSMId {
    /// Creates a `CurveMSMId` from a curve name string. The curve name is truncated to fit into `CURVE_ID_LEN` bytes if necessary.
    pub fn from_curve_name(curve_name: &str) -> Self {
        let curve_name = curve_name.trim_start_matches("ark_");
        let curve_name = curve_name.split("::").next().unwrap_or(curve_name);
        let mut id = [0u8; CURVE_ID_LEN];
        let bytes = curve_name.as_bytes();
        let len = bytes.len().min(CURVE_ID_LEN);
        id[..len].copy_from_slice(&bytes[..len]);
        Self(id)
    }

    /// Returns the curve name string from the `CurveMSMId`. The returned string is trimmed to remove trailing zeros.
    #[cfg(feature = "std")]
    pub fn name(&self) -> String {
        let first_zero = self.0.iter().position(|&b| b == 0).unwrap_or(CURVE_ID_LEN);
        let name_bytes = &self.0[..first_zero];
        String::from_utf8_lossy(name_bytes).to_string()
    }
}

/// Pack a fat pointer (ptr and length) into a u64.
pub const fn pack_fat_pointer(ptr: u32, len: u32) -> u64 {
    let ptr_val = ptr as u64;
    let len_val = len as u64;
    (len_val << 32) | ptr_val
}

/// Unpack a fat pointer (ptr and length) from a u64.
pub const fn unpack_fat_pointer(fat_ptr: u64) -> (u32, u32) {
    let ptr = (fat_ptr & 0xFFFFFFFF) as u32;
    let len = (fat_ptr >> 32) as u32;
    (ptr, len)
}

#[cfg(not(feature = "std"))]
#[cfg_attr(feature = "polkavm", polkavm_derive::polkavm_import)]
#[cfg_attr(target_arch = "wasm32", link(wasm_import_module = "env"))]
extern "C" {
    /// let (buf_ptr, buf_len) = unpack_fat_pointer(fat_ptr);
    ///
    /// If `buf_len` is 32, then the call is to check if the host supports MSM for the specified curve, and the `buffer` contains only `CurveMSMId`.
    /// If `buf_len` is greater than 32, then the call is to perform MSM, and the `buffer` contains the serialized bases and scalars, with the first 32 bytes being the `CurveMSMId`.
    ///
    /// A return value of 0 indicates that the host does not support the curve or that an error occurred during MSM, while a non-zero return value indicates the length of the serialized result of the MSM operation.
    fn host_msm_unchecked(fat_ptr: u64) -> u32;
}

/// Curve names whose support probe the host answered with `1`, keyed by the name's address.
/// The answer is fixed for a node binary, so later calls skip the probe. A full cache leaves the
/// remaining curves probing on every call.
#[cfg(not(feature = "std"))]
mod supported_cache {
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY: AtomicUsize = AtomicUsize::new(0);
    static NAMES: [AtomicUsize; 8] = [EMPTY; 8];

    pub(super) fn contains(name: &'static str) -> bool {
        let key = name.as_ptr() as usize;
        NAMES.iter().any(|slot| slot.load(Ordering::Relaxed) == key)
    }

    pub(super) fn insert(name: &'static str) {
        let key = name.as_ptr() as usize;
        for slot in &NAMES {
            match slot.compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(current) if current == key => return,
                Err(_) => {},
            }
        }
    }
}

#[cfg(not(feature = "std"))]
pub fn use_host_msm_unchecked<
    B: CanonicalSerialize,
    S: CanonicalSerialize,
    R: CanonicalDeserialize,
>(
    curve_name: &'static str,
    bases: &[B],
    scalars: &[S],
) -> Option<R> {
    let mut buffer = Vec::new();
    let curve_id = CurveMSMId::from_curve_name(curve_name);
    curve_id.serialize_uncompressed(&mut buffer).ok()?;

    // Call the host function with only the curve ID to check if the host supports MSM for this
    // curve, once per curve that it does.
    if !supported_cache::contains(curve_name) {
        let fat_ptr = pack_fat_pointer(buffer.as_ptr() as u32, buffer.len() as u32);
        let res_len = unsafe { host_msm_unchecked(fat_ptr) as usize };
        if res_len == 0 {
            // Host does not support MSM for this curve or an error occurred.
            return None;
        }
        supported_cache::insert(curve_name);
    }

    bases.serialize_uncompressed(&mut buffer).ok()?;
    scalars.serialize_uncompressed(&mut buffer).ok()?;
    let fat_ptr = pack_fat_pointer(buffer.as_ptr() as u32, buffer.len() as u32);
    let res_len = unsafe { host_msm_unchecked(fat_ptr) as usize };
    if res_len > 0 && res_len <= buffer.len() {
        R::deserialize_uncompressed_unchecked(&buffer[..res_len]).ok()
    } else {
        // An error occurred during MSM, or the host reported a length past the buffer.
        None
    }
}
