#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::{
    hashing::{
        curve_maps::{svdw::SVDWMap, swu::SWUMap, wb::WBMap},
        map_to_curve_hasher::{MapToCurve, MapToCurveBasedHasher},
        HashToCurve,
    },
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    VariableBaseMSM,
};
use ark_ff::field_hashers::{DefaultFieldHasher, HashToField, LegacyFieldHasher};
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_serialize::{
    impls::compact::CompactU64, CanonicalDeserialize, CanonicalSerialize, Compress,
};
use ark_std::vec::Vec;
use sha2::Sha256;

use ark_bls12_381::{g1::Config as Bls12_381G1Config, g2::Config as Bls12_381G2Config};
use ark_bn254::{g1::Config as Bn254G1Config, g2::Config as Bn254G2Config};
use ark_helios::HeliosConfig;
use ark_pallas::PallasConfig;
use ark_selene::SeleneConfig;
use ark_vesta::VestaConfig;
use ark_wei25519::Wei25519Config;

/// Most points one host call hashes, about 1.3 s of single-threaded host work on Pallas. Larger
/// batches are hashed in the guest.
pub const MAX_HOST_GENS_PER_CALL: u32 = 1 << 16;

/// Hash-to-curve for a short Weierstrass curve using the curve's `FieldHasher` and `Map`.
/// Pallas, Vesta, Helios, Selene and Wei25519 use `LegacyFieldHasher<Sha256, 128>`, which keeps
/// the points equal to those hashed with ark-ff 0.5, so generators derived before the RFC 9380
/// padding fix in `DefaultFieldHasher` are unchanged. BLS12-381 and BN254 use
/// `DefaultFieldHasher<Sha256, 128>`. The host serves Pallas, Vesta and G1 and G2 of BLS12-381
/// and BN254; Helios, Selene and Wei25519 always hash in the guest.
pub trait HashToCurveConfig: SWCurveConfig {
    /// Hash from a message to base field elements.
    type FieldHasher: HashToField<Self::BaseField>;

    /// Map from a base field element to the curve.
    type Map: MapToCurve<Projective<Self>>;

    /// Hash `message` to a curve point, using `dst` as the domain separation tag.
    fn hash_to_curve(dst: &[u8], message: &[u8]) -> Affine<Self> {
        MapToCurveBasedHasher::<Projective<Self>, Self::FieldHasher, Self::Map>::new(
            dst,
        )
        .expect("hasher construction does not fail")
        .hash(message)
        .expect("SWU, WB and SVDW maps do not fail")
    }

    /// Hash `msg_prefix || j.to_le_bytes()` for each `j` in `gens_offset..gens_offset + gens_count`,
    /// using `dst` as the domain separation tag. Uses the host function in a no_std build with the
    /// `host_hash_to_curve` feature when the host supports the curve and `gens_count` is at most
    /// [`MAX_HOST_GENS_PER_CALL`]. Panics if `gens_offset + gens_count` overflows `u32`. Allocates
    /// `gens_count` points, so the caller bounds it.
    fn batch_hash_to_curve(
        dst: &[u8],
        msg_prefix: &[u8],
        gens_offset: u32,
        gens_count: u32,
    ) -> Vec<Affine<Self>> {
        #[cfg(all(feature = "host_hash_to_curve", not(feature = "std")))]
        if let Some(gens) =
            guest::use_host_batch_hash_to_curve::<Self>(dst, msg_prefix, gens_offset, gens_count)
        {
            return gens;
        }
        batch_hash_to_curve_local::<Self>(dst, msg_prefix, gens_offset, gens_count)
    }
}

impl HashToCurveConfig for PallasConfig {
    type FieldHasher = LegacyFieldHasher<Sha256, 128>;
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for VestaConfig {
    type FieldHasher = LegacyFieldHasher<Sha256, 128>;
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for HeliosConfig {
    type FieldHasher = LegacyFieldHasher<Sha256, 128>;
    type Map = SWUMap<Self>;
}

impl HashToCurveConfig for SeleneConfig {
    type FieldHasher = LegacyFieldHasher<Sha256, 128>;
    type Map = SWUMap<Self>;
}

impl HashToCurveConfig for Wei25519Config {
    type FieldHasher = LegacyFieldHasher<Sha256, 128>;
    type Map = SWUMap<Self>;
}

impl HashToCurveConfig for Bls12_381G1Config {
    type FieldHasher = DefaultFieldHasher<Sha256, 128>;
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for Bls12_381G2Config {
    type FieldHasher = DefaultFieldHasher<Sha256, 128>;
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for Bn254G1Config {
    type FieldHasher = DefaultFieldHasher<Sha256, 128>;
    type Map = SVDWMap<Self>;
}

impl HashToCurveConfig for Bn254G2Config {
    type FieldHasher = DefaultFieldHasher<Sha256, 128>;
    type Map = SVDWMap<Self>;
}

fn batch_hash_to_curve_local<C: HashToCurveConfig>(
    dst: &[u8],
    msg_prefix: &[u8],
    gens_offset: u32,
    gens_count: u32,
) -> Vec<Affine<C>> {
    let gens_end = gens_offset
        .checked_add(gens_count)
        .expect("gens_offset + gens_count overflows u32");
    // One hasher and one message buffer per worker; the last four bytes take `j`.
    let new_state = || {
        let hasher = MapToCurveBasedHasher::<Projective<C>, C::FieldHasher, C::Map>::new(dst)
            .expect("hasher construction does not fail");
        let mut msg = Vec::with_capacity(msg_prefix.len() + 4);
        msg.extend_from_slice(msg_prefix);
        msg.extend_from_slice(&[0; 4]);
        (hasher, msg)
    };
    let hash_one = |(hasher, msg): &mut (_, Vec<u8>), j: u32| {
        msg[msg_prefix.len()..].copy_from_slice(&j.to_le_bytes());
        HashToCurve::hash(hasher, msg).expect("SWU, WB and SVDW maps do not fail")
    };

    #[cfg(feature = "parallel")]
    let gens = {
        use rayon::prelude::*;
        (gens_offset..gens_end)
            .into_par_iter()
            .map_init(new_state, hash_one)
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let gens = {
        let mut state = new_state();
        (gens_offset..gens_end).map(|j| hash_one(&mut state, j)).collect()
    };

    gens
}

/// Host function ID of curve `C`. Same as the one used by the host MSM.
pub fn curve_id<C: SWCurveConfig>() -> CurveMSMId {
    let name = <Projective<C> as VariableBaseMSM>::curve_name()
        .expect("short Weierstrass curves have a name");
    CurveMSMId::from_curve_name(name)
}

/// Host function input following the `CurveMSMId`.
#[derive(Clone, Debug, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchHashToCurveRequest {
    pub dst: Vec<u8>,
    pub msg_prefix: Vec<u8>,
    pub gens_offset: u32,
    pub gens_count: u32,
}

/// Uncompressed serialized size of a `Vec` of `count` points of `C`.
pub fn batch_serialized_size<C: SWCurveConfig>(count: u32) -> usize {
    CompactU64(count as u64)
        .uncompressed_size()
        .saturating_add((count as usize).saturating_mul(C::serialized_size(Compress::No)))
}

#[cfg(all(feature = "host_hash_to_curve", not(feature = "std")))]
mod guest {
    use super::*;

    #[cfg_attr(feature = "polkavm", polkavm_derive::polkavm_import)]
    #[cfg_attr(target_arch = "wasm32", link(wasm_import_module = "env"))]
    extern "C" {
        /// `let (buf_ptr, buf_len) = unpack_fat_pointer(fat_ptr)`. The buffer starts with a
        /// `CurveMSMId`. If `buf_len` is `CURVE_ID_LEN`, the call checks whether the host supports
        /// the curve and returns non-zero if so. Otherwise the rest of the buffer is a
        /// `BatchHashToCurveRequest`, the buffer must have `batch_serialized_size` bytes for the
        /// result, and the host writes the uncompressed `Vec` of points to the start of the buffer
        /// and returns its length. Returns 0 on an unsupported curve or an error.
        fn host_batch_hash_to_curve(fat_ptr: u64) -> u32;
    }

    pub(crate) fn use_host_batch_hash_to_curve<C: HashToCurveConfig>(
        dst: &[u8],
        msg_prefix: &[u8],
        gens_offset: u32,
        gens_count: u32,
    ) -> Option<Vec<Affine<C>>> {
        if gens_count > MAX_HOST_GENS_PER_CALL {
            return None;
        }
        let mut buffer = Vec::new();
        curve_id::<C>().serialize_uncompressed(&mut buffer).ok()?;

        let fat_ptr = pack_fat_pointer(buffer.as_ptr() as u32, buffer.len() as u32);
        if unsafe { host_batch_hash_to_curve(fat_ptr) } == 0 {
            return None;
        }

        BatchHashToCurveRequest {
            dst: dst.to_vec(),
            msg_prefix: msg_prefix.to_vec(),
            gens_offset,
            gens_count,
        }
        .serialize_uncompressed(&mut buffer)
        .ok()?;
        let expected_res_len = batch_serialized_size::<C>(gens_count);
        if expected_res_len > buffer.len() {
            buffer.resize(expected_res_len, 0);
        }

        let fat_ptr = pack_fat_pointer(buffer.as_ptr() as u32, buffer.len() as u32);
        let res_len = unsafe { host_batch_hash_to_curve(fat_ptr) } as usize;
        if res_len == 0 || res_len > buffer.len() {
            return None;
        }
        let gens = Vec::<Affine<C>>::deserialize_uncompressed_unchecked(&buffer[..res_len]).ok()?;
        (gens.len() == gens_count as usize).then_some(gens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::vec;

    fn point<C: SWCurveConfig>(h: &str) -> Affine<C> {
        Affine::<C>::deserialize_compressed(&hex::decode(h).unwrap()[..]).unwrap()
    }

    /// `expected` holds compressed hex of `hash_to_curve(b"test-dst", b"msg-1")`,
    /// `hash_to_curve(b"test-dst", b"msg-2")`, the `PedersenGens` label `b"test"` bases `B` and
    /// `B_blinding`, then the first 4 `BulletproofGens` label `b"test"` party 0 `G` and `H` bases.
    /// Values are from curve-trees `bulletproofs` before the move.
    fn check_known_answers<C: HashToCurveConfig>(expected: [&str; 12]) {
        assert_eq!(C::hash_to_curve(b"test-dst", b"msg-1"), point::<C>(expected[0]));
        assert_eq!(C::hash_to_curve(b"test-dst", b"msg-2"), point::<C>(expected[1]));
        assert_eq!(C::hash_to_curve(b"PedersenGens", b"test-B"), point::<C>(expected[2]));
        assert_eq!(
            C::hash_to_curve(b"PedersenGens", b"test-B_blinding"),
            point::<C>(expected[3])
        );
        let g = C::batch_hash_to_curve(b"BulletproofGens-G\0\0\0\0", b"test", 0, 4);
        let h = C::batch_hash_to_curve(b"BulletproofGens-H\0\0\0\0", b"test", 0, 4);
        let expected_g: Vec<_> = expected[4..8].iter().map(|e| point::<C>(e)).collect();
        let expected_h: Vec<_> = expected[8..12].iter().map(|e| point::<C>(e)).collect();
        assert_eq!(g, expected_g);
        assert_eq!(h, expected_h);
    }

    #[test]
    fn pallas_known_answers() {
        check_known_answers::<PallasConfig>([
            "8a371389e6eb923e7cc6714041bed32926e890bd4d6f4207c591e1efacee081c",
            "cac0539620aa7fd4b6718cc3f4dcc78d14b313277f806e33287738806fe21091",
            "143ac4d7392321a2c43795c07b3ef3bf3c52f72c46a64d3a9373d7b1c2c7f517",
            "b06efcb2aef98c0ca309b5753b6e2140b6908fba51fade1f8e61403a9205d82d",
            "0ff4ced5544ec8c26343cb105f0bd3c1176431ecb0529145bf2d83fb21c860ac",
            "85675f8a1aa34f6c673570177cba374d5f8590e42fefb16be3b62cd62ba966b4",
            "a5c8f074b68da078f1b1164eac8914b4c2e07094a567422965c2e8569dc18409",
            "31a07e4c79bd443e8a52d5beaf81a13ce5f01c77a5ed37155d7f5e32327a57ab",
            "9d76c4946f5adcb67d8ae2979dfadacda2d7a85d74daca28ef5575dc9b41f62f",
            "936b1ec580c020c9f902a720c576f5e23ee05f5990b4bd15fe0ed932bbe3b8af",
            "9909f360a7a87b02d92ae473f11e59ab2d908299f22c7250030a25c60b14e21b",
            "a36d3a1d0f8c64529bf49bc125f0d2811ba9585c667ba4b47250062d406c90be",
        ]);
        assert_eq!(
            PallasConfig::hash_to_curve(b"pallas", b"curve_trees_delta"),
            point::<PallasConfig>(
                "9643f0714aa4792a6ba9702608e79c047941c410903513de313bc4c15cdf520f"
            )
        );
    }

    #[test]
    fn vesta_known_answers() {
        check_known_answers::<VestaConfig>([
            "56b41fa2237a00eacbf1553e9f145841d4362d50c402f736fc94bd38136ced3e",
            "4a2dcb19a289e2ca28eaf56f2f58490054d156edc96bcb38b822e3c8baa69901",
            "036ec31db44279217fa58bca97ed76efdb4a6c4868bddab44700227b395843a7",
            "1c58a61c78bf2b3c8e9c3d4c2cba7ddeea707f8cceae0e96f10bcd19303f5bb1",
            "edb9ea2a58157e292c0d0e526ab9d98bd1e4ed5f46e66b1c6cf382d0ee680815",
            "d5b5f43641e7ca7f14e71cdf8455f30f17ed9df306af26499f7effc8e2c45a0f",
            "ae7798d7658324b98165f210866ee43499c8781a2789cb68455d05acc1ee5999",
            "a32d4da06f4f0041db8893910cb2aa1c8350a003bb2f40d86bdbcf05e0bedbbd",
            "9dbe54ef7e2d0deb29e8ffee7782daaf685f1e42fbee712291b0728f673f34a8",
            "00cb1cb1ff6c3e82779f112a7b907b15a04ddc6b0574698923791d87ba7c3b3d",
            "00b7d94e7a511f9e8a8986dbb5cee737ac9281f1653135cfd0e4f85defe8a413",
            "85489cd7d7ecc7809390e5b2d0eebeb057495dfd1f587fbfcdbfbf7bd0abf384",
        ]);
        assert_eq!(
            VestaConfig::hash_to_curve(b"vesta", b"curve_trees_delta"),
            point::<VestaConfig>(
                "b2ce9cce16caf05b8c742f843032e374e0c835c38bd57027ddda990aee52ec0a"
            )
        );
    }

    #[test]
    fn helios_known_answers() {
        check_known_answers::<HeliosConfig>([
            "96351c60284b20d629eb4dc2c06975bb63b9c629c1e5ed7cd8c6e8594a39f1a4",
            "ae238936a8291709b8da7eb2797da60c7f8524b4eb6d5030c17e662b93c0a77c",
            "21d0d2a2ce8a4dcf60d4c8aeec0b0f13c6acda767cf0d2cab233dc3a18722118",
            "d4f5e55fab8135770386637734cf3313e68a3dc00c35a1c88be9d6aa9a01ced7",
            "efec26ce9eb8049ebb7c2294cd1320261d914dc4b5d785241299eb5d403b29c5",
            "448e6e1050274f2272a7685ad03bd7603e4ccddc211487c11b154f1f9300a0cc",
            "742d60bda29ea09b24cddd9f76c4865a79ed393e92c1082d790cab9fe2a748b3",
            "f6a569780b732a12c9ba74e381ef03ef16c9d7466ade75ed65f26645767aa3c9",
            "aecbaf016cef5fe4ffe73d1141d2b8d0d4d68da6b7258d4c24156022d11d7862",
            "c2161934bb6b3fd1993feb13082c4cce8cb14c54fb8c13e2fbabfccb1c908dd1",
            "e85d387b6aabf3111b0a61224a1aaccd4babd87b435479fdd4e71b8f2697d236",
            "b97e9b4d9eba3b4387ac21a6de3d2eeeeba25d3a4d2269e2b12338c30cddc181",
        ]);
    }

    #[test]
    fn selene_known_answers() {
        check_known_answers::<SeleneConfig>([
            "aa416031ab45d7e7f2df18ce16b2d29edabd2ebbe1c0562d2c47771761977260",
            "f9f25e3f3db49c2a3e93197c4725e6c6b9372ec00173d475871624cce34bf772",
            "8ec8897d935ca7b26ee7143abc7f2eaa8827ad8e0eaa97867e3bf49fcb215f05",
            "d0ea2867a26ae80d8db733c80e10ce619b2e6df7de2367ec225c48e2ad241ac9",
            "25a7787cd23a5db53002bea5a9f5f0803c029abca1b265497a7a33ac2409ca25",
            "7f3b98b3235d212285ec9cd10512d2a725c92c9811ba4982549ef75c58fc8867",
            "ed5ea2088f83125db0e1463db1eedf94a4fe35d13cc78161ce92a58f4b40c869",
            "d30e102b7b97aa224d24ae0a84d8dcadf0b91a3b3932b29a5605c17d2009641e",
            "d0d643450e44b2eca453258efd1c7c00cf07c47780b7b643d5f918c68e2918ae",
            "da90a5eddbdba0172063232586f7d92f16545cc41c3d1fa62731ab0918a664ae",
            "9f7156603036d616c408588a98aa8c93bb2aaa645db4d440ad127703f340e911",
            "caf882441321d3070475f3b34bd79ed01b6bde5f992617a17a0710ac78baf3fe",
        ]);
    }

    #[test]
    fn wei25519_known_answers() {
        check_known_answers::<Wei25519Config>([
            "a2b23a0883b03e04dec705da83a89ad26998589a645bd1c5349fa2853a11a869",
            "4f4b50110778cf169b1dd230c7c5a913298bb36b1ea9d86ab4090c00819d262a",
            "3f9fdc03ca1516269384b0ca0471b290e47f9f6c18e3bad9e0d4c76f88551059",
            "33d11257a2572f4bed99387aef742e8d5088443d5f04fc66f861adbeca5c38ef",
            "650e1e87b154e5f3f768132706e1e5c0f221d46eec9f4fc1c9802b139b6dd0de",
            "c7a626b57a6b03ece759843d4265c8c9f95385936657d49c1eab275b48ab96ed",
            "6c45e89f8ef4b527af042652079cacb8c076d3d4722678353ab82b4a4a333c55",
            "b67994f4d2fecf078ec48931cfbf9696aecd00a74aaff6d4ffb43d4018864afb",
            "4f5adde1f73cca335180d688a2e99aba28f3b557dac7f2241f4d13b800acb873",
            "e06dc88c04e329569e86b1432b6f34552b847b133cd20b644b390a273cf20a27",
            "e30a6489d4e13525400c07c2d0336e86194e4a3cf3ec894c1acc0b811120fe74",
            "afeb6e273b5953d6219dc677b9ca108add9d05eee5737f44626edc278ec3b54d",
        ]);
    }

    #[test]
    fn batch_matches_single() {
        let gens = PallasConfig::batch_hash_to_curve(b"dst", b"prefix", 5, 3);
        for (j, g) in (5u32..8).zip(gens.iter()) {
            let msg = [b"prefix".as_slice(), j.to_le_bytes().as_slice()].concat();
            assert_eq!(*g, PallasConfig::hash_to_curve(b"dst", &msg));
        }
    }

    #[test]
    fn curve_ids_unchanged() {
        fn check<C: SWCurveConfig>(name: &str) {
            assert_eq!(curve_id::<C>(), CurveMSMId::from_curve_name(name));
        }
        check::<PallasConfig>("pallas");
        check::<VestaConfig>("vesta");
        check::<HeliosConfig>("helios");
        check::<SeleneConfig>("selene");
        check::<Wei25519Config>("wei25519");
        check::<Bls12_381G1Config>("bls12_381_g1");
        check::<Bls12_381G2Config>("bls12_381_g2");
        check::<Bn254G1Config>("bn254_g1");
        check::<Bn254G2Config>("bn254_g2");
    }

    #[test]
    fn curve_ids_are_distinct() {
        let ids = [
            curve_id::<PallasConfig>(),
            curve_id::<VestaConfig>(),
            curve_id::<HeliosConfig>(),
            curve_id::<SeleneConfig>(),
            curve_id::<Wei25519Config>(),
            curve_id::<Bls12_381G1Config>(),
            curve_id::<Bls12_381G2Config>(),
            curve_id::<Bn254G1Config>(),
            curve_id::<Bn254G2Config>(),
        ];
        let distinct: ark_std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(distinct.len(), ids.len());
    }

    #[test]
    fn batch_serialized_size_matches() {
        for count in [0u32, 1, 63, 64, 1000] {
            let gens = vec![PallasConfig::GENERATOR; count as usize];
            assert_eq!(
                batch_serialized_size::<PallasConfig>(count),
                gens.uncompressed_size(),
                "count = {count}"
            );
        }
    }

    #[test]
    fn request_serialization_unchanged() {
        let req = BatchHashToCurveRequest {
            dst: b"dst".to_vec(),
            msg_prefix: b"prefix".to_vec(),
            gens_offset: 7,
            gens_count: 9,
        };
        let mut derived = Vec::new();
        req.serialize_uncompressed(&mut derived).unwrap();
        let mut manual = Vec::new();
        b"dst".to_vec().serialize_uncompressed(&mut manual).unwrap();
        b"prefix".to_vec().serialize_uncompressed(&mut manual).unwrap();
        7u32.serialize_uncompressed(&mut manual).unwrap();
        9u32.serialize_uncompressed(&mut manual).unwrap();
        assert_eq!(derived, manual);
    }
}
