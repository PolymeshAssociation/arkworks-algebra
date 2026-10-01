#[macro_export]
macro_rules! test_pairing {
    ($mod_name: ident; $Pairing: ty) => {
        mod $mod_name {
            pub const ITERATIONS: usize = 100;
            use ark_ec::{pairing::*, CurveGroup, PrimeGroup};
            use ark_ff::{CyclotomicMultSubgroup, Field, PrimeField};
            use ark_std::{test_rng, vec, vec::Vec, One, UniformRand, Zero};
            #[test]
            fn test_bilinearity() {
                for _ in 0..100 {
                    let mut rng = test_rng();
                    let a: <$Pairing as Pairing>::G1 = UniformRand::rand(&mut rng);
                    let b: <$Pairing as Pairing>::G2 = UniformRand::rand(&mut rng);
                    let s: <$Pairing as Pairing>::ScalarField = UniformRand::rand(&mut rng);

                    let sa = a * s;
                    let sb = b * s;

                    let ans1 = <$Pairing>::pairing(sa, b);
                    let ans2 = <$Pairing>::pairing(a, sb);
                    let ans3 = <$Pairing>::pairing(a, b) * s;

                    assert_eq!(ans1, ans2);
                    assert_eq!(ans2, ans3);

                    assert_ne!(ans1, PairingOutput::zero());
                    assert_ne!(ans2, PairingOutput::zero());
                    assert_ne!(ans3, PairingOutput::zero());
                    let group_order = <<$Pairing as Pairing>::ScalarField>::characteristic();

                    assert_eq!(ans1.mul_bigint(group_order), PairingOutput::zero());
                    assert_eq!(ans2.mul_bigint(group_order), PairingOutput::zero());
                    assert_eq!(ans3.mul_bigint(group_order), PairingOutput::zero());
                }
            }

            #[test]
            fn test_multi_pairing() {
                for _ in 0..ITERATIONS {
                    let rng = &mut test_rng();

                    let a = <$Pairing as Pairing>::G1::rand(rng).into_affine();
                    let b = <$Pairing as Pairing>::G2::rand(rng).into_affine();
                    let c = <$Pairing as Pairing>::G1::rand(rng).into_affine();
                    let d = <$Pairing as Pairing>::G2::rand(rng).into_affine();
                    let ans1 = <$Pairing>::pairing(a, b) + &<$Pairing>::pairing(c, d);
                    let ans2 = <$Pairing>::multi_pairing(&[a, c], &[b, d]);
                    assert_eq!(ans1, ans2);
                }
            }

            #[test]
            fn test_multi_pairing_many() {
                let rng = &mut test_rng();
                let a: Vec<_> = (0..9)
                    .map(|_| <$Pairing as Pairing>::G1::rand(rng).into_affine())
                    .collect();
                let b: Vec<_> = (0..9)
                    .map(|_| <$Pairing as Pairing>::G2::rand(rng).into_affine())
                    .collect();
                let singles: Vec<_> = a.iter().zip(&b).map(|(p, q)| <$Pairing>::pairing(p, q)).collect();
                for n in [1, 3, 4, 5, 8, 9] {
                    let expected = singles[..n].iter().sum::<PairingOutput<$Pairing>>();
                    assert_eq!(<$Pairing>::multi_pairing(&a[..n], &b[..n]), expected, "n = {n}");
                }
            }

            /// `mul_bits_be` and `mul_bigint` agree with `pow` on GT, for scalars that are not
            /// bit palindromes, wide ones, and ones whose length is not a multiple of 64.
            #[test]
            fn test_gt_mul_bits_be() {
                use ark_ff::BitIteratorBE;
                let rng = &mut test_rng();
                let g = PairingOutput::<$Pairing>::generator() * <$Pairing as Pairing>::ScalarField::rand(rng);
                let mut scalars: Vec<Vec<u64>> = vec![
                    vec![1],
                    vec![2],
                    vec![0x8000_0000_0000_0001, 5],
                    vec![u64::MAX, 0, 1],
                    vec![7, 0, 0, 0, 0, 0, 3],
                ];
                scalars.push(<$Pairing as Pairing>::ScalarField::rand(rng).into_bigint().as_ref().to_vec());
                for k in &scalars {
                    let expected = PairingOutput::<$Pairing>(g.0.pow(k));
                    assert_eq!(g.mul_bigint(k), expected, "mul_bigint, k = {k:?}");
                    assert_eq!(g.mul_bits_be(BitIteratorBE::new(k)), expected, "mul_bits_be, k = {k:?}");
                    let trimmed: Vec<bool> = BitIteratorBE::without_leading_zeros(k).collect();
                    assert_eq!(g.mul_bits_be(trimmed.into_iter()), expected, "trimmed bits, k = {k:?}");
                }
            }

            #[test]
            fn test_final_exp() {
                for _ in 0..ITERATIONS {
                    let rng = &mut test_rng();
                    let fp_ext = <$Pairing as Pairing>::TargetField::rand(rng);
                    let gt = <$Pairing as Pairing>::final_exponentiation(MillerLoopOutput(fp_ext))
                        .unwrap()
                        .0;
                    let r = <$Pairing as Pairing>::ScalarField::MODULUS;
                    assert!(gt.cyclotomic_exp(r).is_one());
                }
            }
        }
    };
}
