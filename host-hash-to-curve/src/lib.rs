#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::{
    hashing::{
        curve_maps::{swu::SWUMap, wb::WBMap},
        map_to_curve_hasher::{MapToCurve, MapToCurveBasedHasher},
        HashToCurve,
    },
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    VariableBaseMSM,
};
use ark_ff::field_hashers::DefaultFieldHasher;
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_serialize::{
    impls::compact::CompactU64, CanonicalDeserialize, CanonicalSerialize, Compress,
};
use ark_std::vec::Vec;
use sha2::Sha256;

use ark_helios::HeliosConfig;
use ark_pallas::PallasConfig;
use ark_selene::SeleneConfig;
use ark_vesta::VestaConfig;
use ark_wei25519::Wei25519Config;

/// Hash-to-curve for a short Weierstrass curve using `DefaultFieldHasher<Sha256, 128>` and the
/// curve's `Map`.
pub trait HashToCurveConfig: SWCurveConfig {
    /// Map from a base field element to the curve.
    type Map: MapToCurve<Projective<Self>>;

    /// Hash `message` to a curve point, using `dst` as the domain separation tag.
    fn hash_to_curve(dst: &[u8], message: &[u8]) -> Affine<Self> {
        MapToCurveBasedHasher::<Projective<Self>, DefaultFieldHasher<Sha256, 128>, Self::Map>::new(
            dst,
        )
        .expect("hasher construction does not fail")
        .hash(message)
        .expect("SWU and WB maps do not fail")
    }

    /// Hash `msg_prefix || j.to_le_bytes()` for each `j` in `gens_offset..gens_offset + gens_count`,
    /// using `dst` as the domain separation tag. Uses the host function in a no_std build with the
    /// `host_hash_to_curve` feature when the host supports the curve. Panics if
    /// `gens_offset + gens_count` overflows `u32`.
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
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for VestaConfig {
    type Map = WBMap<Self>;
}

impl HashToCurveConfig for HeliosConfig {
    type Map = SWUMap<Self>;
}

impl HashToCurveConfig for SeleneConfig {
    type Map = SWUMap<Self>;
}

impl HashToCurveConfig for Wei25519Config {
    type Map = SWUMap<Self>;
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
    let hash_one = |j: u32| {
        let msg = [msg_prefix, j.to_le_bytes().as_slice()].concat();
        C::hash_to_curve(dst, &msg)
    };

    #[cfg(feature = "parallel")]
    let gens = {
        use rayon::prelude::*;
        (gens_offset..gens_end).into_par_iter().map(hash_one).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let gens = (gens_offset..gens_end).map(hash_one).collect();

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
            "188626a57163df7fe13fa8cbef8d9de91fe214882aa21e18233ef636aa94c92c",
            "00ba3d75f21e3a017d548ef7231d903282a3d80c65269cecd266bf4fb0191125",
            "e84b3ec2751a9ea0ca064226ff390af79f7dfc2a3fa2e4f63cc85b4e45763a88",
            "889de1d2955b1b6d1fe195f41e4ab674df5ca8ce2a82bdddcce20f8165e76723",
            "f42e4ad4db99e1e577ee362bfc1f14773d30205518c0ed1ddb8ecb19a6e77597",
            "54bb81ff3918b4057dbcb13815090a58a8c71d4dbfeb815dc959086077ee14a6",
            "ab3d603534567f2dc0aacefaa1c86fb396d75ef9f2366c909537985bc7687986",
            "06a13137807deb67103af3a5a64e14d5dba7150284960ea96b24f763f1b11d30",
            "5bb1a0567966c5af05895bc52172ac0162b9c34002baffc53f5f6b1282008f9c",
            "70ffc94d03d54ebdace891527896c2544098a3205a067bd328d3e16a11ac158e",
            "112a7b2cf750a1af545852c8980ecfd0e6503fd8d4ecbf075e1986ba589eb72d",
            "21e0fdea6c7324eece2bf570b6e1e9ce5bc39b27bf6c301a7ae75f887bc6baa8",
        ]);
        assert_eq!(
            PallasConfig::hash_to_curve(b"pallas", b"curve_trees_delta"),
            point::<PallasConfig>(
                "03b085a32de17d09fd88e060c8f2cc5c4df41767545cc077ca79b8f38d87af82"
            )
        );
    }

    #[test]
    fn vesta_known_answers() {
        check_known_answers::<VestaConfig>([
            "547adfdcb7583ee2380fc42aa1f974c4c8af654fdb590800897e35bd7b2c1525",
            "8f99ed0fa81dee09bf61d749ee1da0ef3b5d7fdcdf752e6cf1939c7563f15d16",
            "7b4787071299f2e0595ed32e914f40b7688c1f86ada7aba32e2b13c3438fc71f",
            "b9fb73da838566d8e64115f96a402fb2e33a283f8043de759f5e4b256884f12f",
            "0d963ac04d3f2da7e33f0719a7608dd2fc635aa167755f8fe4716304ddd01c91",
            "f3d537cea480a7fa97c1b34b13e80328f6c50fc807d4344982668408d3e6a7ab",
            "de205f0ab922a94cb9a42bdf514a2c62bcaa8ebbbbc4b4efca525267ce7f29bd",
            "7d013333fcf82021dd0c508ab68bb009d44e603374c86ff7d787ddb9b037422c",
            "316d3cc333af3853d8b0a2f488516afaac99eeca38c1cac2c54c60641deb08bc",
            "4a25f33220cac2f52966bd784a33a71c6c0081f0878edfd2f2ee0d1b8c01c490",
            "6f53e4ef32bef3507359ee3df3c42b8db588028055bf4caf329498bbd2dc3480",
            "71a2ac5c2f662f9104b85bbbe7e066645597de579d25cc1adabe1e47d4766481",
        ]);
        assert_eq!(
            VestaConfig::hash_to_curve(b"vesta", b"curve_trees_delta"),
            point::<VestaConfig>(
                "f790660d996331a780360119197a3c5238039cdaacbbc7a5c52757a01365639c"
            )
        );
    }

    #[test]
    fn helios_known_answers() {
        check_known_answers::<HeliosConfig>([
            "77c2a4310dc3a732b6de41dac53884135aaac6d3c010505bd0be3bb096f58168",
            "d62471c1aeaba78b9f5e731ae7b72c46477e852721a72ae683a4e65256e6dbfd",
            "69bd9e21a6ee6e21c0e0a7a03f0b89246d26efe1822e7995d0012334a8495bdf",
            "aed1d6c105ab0ebff8e36eb68798d4ab6e29749541efa2b44af1e03e94f094f5",
            "7241ee457e78b97bc768d117ed62a3db820d020a4ef40783f42e39be93884735",
            "d152f6e67fb53683c01d01adf10aeb4c706c4154b8aa032b9e6b3b596b352767",
            "fd57e2c06967e1db3cd1d09723de548138dc347c14e7559c49d6771034123314",
            "3c322934c7c476528eda7ed5ec0857dfb152d7cc5aed0ef8f3363e120b7d9b20",
            "2aeff1610f457c118fe63443e7acffd2d76124deeaa799ce28ece98e9cbff902",
            "e257ea0f826cd144834b0a1124afaff5e9e9e32ec07ffd44114dc5597054f26c",
            "0ebc211f2bfe31758482c876ba4c165cd5c590cec7dbcba5f40c7e1faccf09f9",
            "125e0b938a5f9cb797d5d0ce6a8324aa6fb35a7889573ccefbca3c4c484b2a64",
        ]);
    }

    #[test]
    fn selene_known_answers() {
        check_known_answers::<SeleneConfig>([
            "3ec154d555a41150cb0ba96f27fdb69ce149dfd38c415bb705b6457a70dd944f",
            "9e20344502a8bf60f3fad784d23083f67970c7bbbfe86d66e403923a606d5a87",
            "bf6372b4e169a2f50d10c6df40c07177e962f96de7d0bd62a344ffa64c04ca1b",
            "35288c4c46ee4e1afba5abf659c3d850b36051f0de45272247662796e01b3249",
            "3b92b1812714ff35ed2ab6c82eadbc295ada1b10e999f51c0031f197d809d228",
            "03c7faa329bb7afbddc977f2846a294be749ce1b2b1e98473efb6502c72b41b1",
            "d0eac2b107f98c84d23f51912d344db3d408f38a284c3a96d8221371c3af34ce",
            "6f4786c0450ed2bef3675662b831eb027d0fa183a06dbab4928c462736f38126",
            "29012f131326520b137f2f60a884a4cdd915881d4579480bab37d4a9acdf4139",
            "570538c15ef869e1147f9e36d461b8228e185d5e8edfd60adc8f50aa3c48e582",
            "a90fbfcacaac86a4b1489af092b362bca12c7cb08e99ba65c45e4594a6837cea",
            "f985b63918a49ea8f089e36ac1cdace2c5c3310a784a7a1d950bab52a3f84b2e",
        ]);
    }

    #[test]
    fn wei25519_known_answers() {
        check_known_answers::<Wei25519Config>([
            "ac78470b2b5fe0afc5e6462617e93d04caa6813f6020ffd949ab5c7127b7b600",
            "1a8d48b4b233cd547acc67bc9ca953d05c538ae46c3135051fbcbfb2061371d9",
            "ea464316dd135da33868a7a23a104f9f09622ca42b1033c94feb93fe542e8270",
            "3b115e51bed6f183d93bcf322edb2eb5cb7626f5d4aa69c0aedd6b9624178283",
            "9fe3f6557d95098b5a61e4691313af1506b9bf96dcb22184b6ad6cd91b538354",
            "16005b048a5368a9d7996b33e82530e9763cdc9f1222a36488eb49951a7e3024",
            "157a9de84e3da51e3f8a827e84065ae5b4596ffdb6e70efdaa7d42cf31cc4b01",
            "a38606ab99b6b40e5c44f967cbee29785487baf67173088904bc455c44179147",
            "8421ef0c6de7bd91e441dfb1f019223b7315f2c32811e274f7dfc9ba8eb6d458",
            "f209aa5cbc2f60c2e5542e7b40668f252b38e0b42db0659f92c4ebaffe809d29",
            "d0a9edd9ba899bf981faf65d6bd279c29ee1d8dcbbd6b29a4196da94c4352c06",
            "6f434eb7e84df8e1fd63052bc8393f1510326aec3203c2cf0589798aca3c8a9e",
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
