#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::VariableBaseMSM;
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_pallas::Projective as Pallas;
use ark_serialize::{CanonicalDeserialize, Compress};
use ark_std::vec::Vec;
use ark_vesta::Projective as Vesta;

pub fn host_msm_unchecked(buffer: &mut [u8], buf_len: u32) -> u32 {
    let curve =
        if let Some(curve) = CurveMSMId::deserialize_uncompressed_unchecked(&buffer[..]).ok() {
            curve
        } else {
            return 0; // Invalid curve ID
        };

    if curve == CurveMSMId::from_curve_name("pallas") {
        host_msm_unchecked_impl::<Pallas>(curve, buffer, buf_len)
    } else if curve == CurveMSMId::from_curve_name("vesta") {
        host_msm_unchecked_impl::<Vesta>(curve, buffer, buf_len)
    } else {
        0
    }
}

fn host_msm_unchecked_impl<V: VariableBaseMSM>(
    _curve: CurveMSMId,
    buffer: &mut [u8],
    buf_len: u32,
) -> u32 {
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
