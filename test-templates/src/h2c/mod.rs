pub mod json;
use hex;
pub use hex::decode;
use serde_json;
pub use serde_json::from_reader;
use sha2;
pub use sha2::Sha256;

#[macro_export]
macro_rules! test_h2c {
    ($mod_name: ident; $test_path: literal; $test_name: literal; $group: ty; $field: ty; $base_prime_field: ty; $m: literal) => {
        $crate::test_h2c!($mod_name; $test_path; $test_name; $group; $field; $base_prime_field; $m; ark_ec::hashing::curve_maps::wb::WBMap<$group>; "SSWU");
    };
    ($mod_name: ident; $test_path: literal; $test_name: literal; $group: ty; $field: ty; $base_prime_field: ty; $m: literal; $mapper: ty; $map_name: literal) => {
        mod $mod_name {
            use ark_ff::PrimeField;

            extern crate std;
            use ark_ec::{
                hashing::{
                    map_to_curve_hasher::{MapToCurve, MapToCurveBasedHasher},
                    HashToCurve,
                },
                short_weierstrass::{Affine, Projective},
            };
            use ark_ff::{
                field_hashers::{DefaultFieldHasher, HashToField},
                fields::Field,
                One, UniformRand, Zero,
            };
            use ark_std::{format, string::String, vec::*};
            use std::{
                fs::{read_dir, File},
                io::BufReader,
            };
            use $crate::{decode, Sha256};

            use $crate::json::SuiteVector;
            #[test]
            fn test_h2c() {
                let filename = format!("{}/{}_XMD-SHA-256_{}_RO_.json", $test_path, $test_name, $map_name);

                let file = File::open(filename).unwrap();
                let data: SuiteVector = $crate::from_reader(BufReader::new(file)).unwrap();

                assert_eq!(data.hash, "sha256");
                assert_eq!(data.map.name, $map_name);
                let dst = data.dst.as_bytes();
                let hasher;
                let g1_mapper = MapToCurveBasedHasher::<
                    Projective<$group>,
                    DefaultFieldHasher<Sha256, 128>,
                    $mapper,
                >::new(dst)
                .unwrap();
                hasher = <DefaultFieldHasher<Sha256, 128> as HashToField<$field>>::new(dst);

                for v in data.vectors.iter() {
                    // first, hash-to-field tests
                    let got: [$base_prime_field; { 2 * $m }] =
                        hasher.hash_to_field(&v.msg.as_bytes());
                    let want: Vec<$base_prime_field> =
                        v.u.iter().map(read_fq_vec).flatten().collect();
                    assert_eq!(got[..], *want);

                    // then, map-to-curve tests, where the vectors give Q0 and Q1
                    for (q, u) in [&v.q0, &v.q1].into_iter().zip(got.chunks($m)) {
                        if let Some(q) = q {
                            let u = <$field>::from_base_prime_field_elems(u.iter().copied()).unwrap();
                            let got = <$mapper as MapToCurve<Projective<$group>>>::map_to_curve(u).unwrap();
                            assert_eq!(got, read_point(q));
                        }
                    }

                    // then, test curve points
                    let got = g1_mapper.hash(&v.msg.as_bytes()).unwrap();
                    let want = read_point(&v.p);
                    assert!(got.is_on_curve());
                    assert!(want.is_on_curve());
                    assert_eq!(got, want);
                }
            }
            pub fn read_fq_vec(input: &String) -> Vec<$base_prime_field> {
                input
                    .split(",")
                    .map(|f| {
                        <$base_prime_field>::from_be_bytes_mod_order(
                            &decode(f.trim_start_matches("0x")).unwrap(),
                        )
                    })
                    .collect()
            }
            pub fn read_point(p: &$crate::json::P) -> Affine<$group> {
                Affine::<$group>::new_unchecked(
                    <$field>::from_base_prime_field_elems(read_fq_vec(&p.x)).unwrap(),
                    <$field>::from_base_prime_field_elems(read_fq_vec(&p.y)).unwrap(),
                )
            }
        }
    };
}

#[macro_export]
macro_rules! test_h2c_swu {
    ($test_name:ident; $group: ty; $config: ty) => {
        #[test]
        fn $test_name() {
            use ark_ec::{
                hashing::{
                    curve_maps::swu::SWUMap, map_to_curve_hasher::MapToCurveBasedHasher,
                    HashToCurve,
                },
            };
            use ark_ff::field_hashers::DefaultFieldHasher;
            use ark_std::{test_rng, vec, UniformRand};
            use $crate::sha2::Sha256;

            let hasher = MapToCurveBasedHasher::<
                $group,
                DefaultFieldHasher<Sha256, 128>,
                SWUMap<$config>,
            >::new(b"test")
            .unwrap();

            let mut rng = test_rng();
            for _ in 0..100 {
                let mut bytes = vec![0u8; rng.gen_range(1..10000)];
                let hash_result = hasher.hash(&mut bytes).unwrap();
                assert!(
                    hash_result.is_on_curve(),
                    "hash results into a point off the curve"
                );
            }
        }
    };
}
