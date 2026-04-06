#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::VariableBaseMSM;
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_pallas::Projective as Pallas;
use ark_serialize::{CanonicalDeserialize, Compress};
use ark_std::boxed::Box;
use ark_std::collections::BTreeMap;
use ark_std::vec::Vec;
use ark_vesta::Projective as Vesta;

type CurveMSMFn = Box<dyn Fn(&mut [u8], u32) -> u32 + Send + Sync>;

pub struct RegisteredCurves {
    curves: BTreeMap<CurveMSMId, CurveMSMFn>,
}

impl RegisteredCurves {
    pub fn new() -> Self {
        let mut curves = RegisteredCurves {
            curves: BTreeMap::new(),
        };
        curves.register_curve::<Pallas>();
        curves.register_curve::<Vesta>();
        curves
    }

    pub fn register_curve<V: VariableBaseMSM + 'static>(&mut self) -> bool {
        if let Some(name) = V::curve_name() {
            let curve_id = CurveMSMId::from_curve_name(name);
            self.curves
                .insert(curve_id, Box::new(host_msm_unchecked_impl::<V>));
            true
        } else {
            false
        }
    }

    pub fn msm_unchecked(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
        if (buf_len as usize) < CURVE_ID_LEN {
            return 0; // Buffer too small to contain curve ID
        }
        if let Some(curve_id) = CurveMSMId::deserialize_uncompressed_unchecked(&buffer[..]).ok() {
            if let Some(msm_fn) = self.curves.get(&curve_id) {
                if buf_len as usize > CURVE_ID_LEN {
                    return msm_fn(buffer, buf_len);
                } else {
                    return 1; // Curve is supported, but no MSM data provided
                }
            }
        }
        0
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

#[cfg(feature = "std")]
pub fn host_msm_unchecked(buffer: &mut [u8], buf_len: u32) -> u32 {
    SUPPORTED_CURVES.msm_unchecked(buffer, buf_len)
}

#[cfg(not(feature = "std"))]
pub fn host_msm_unchecked(buffer: &mut [u8], buf_len: u32) -> u32 {
    get_supported_curves().msm_unchecked(buffer, buf_len)
}

fn host_msm_unchecked_impl<V: VariableBaseMSM>(buffer: &mut [u8], buf_len: u32) -> u32 {
    if buf_len as usize == CURVE_ID_LEN {
        // The curve is supported.
        return 1;
    }
    let buf_len = buf_len as usize;
    let mut cursor = ark_std::io::Cursor::new(&buffer[CURVE_ID_LEN..buf_len]);
    let bases: Vec<V::MulBase> =
        CanonicalDeserialize::deserialize_uncompressed_unchecked(&mut cursor).unwrap();
    let scalars: Vec<V::ScalarField> =
        CanonicalDeserialize::deserialize_uncompressed_unchecked(&mut cursor).unwrap();
    let res = V::msm_unchecked(&bases, &scalars);
    let res_len = res.serialized_size(Compress::No);
    res.serialize_uncompressed(&mut buffer[0..res_len]).unwrap();
    res_len as u32
}
