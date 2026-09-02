use ark_ec::{
    scalar_mul::{double_and_add, double_and_add_affine, glv::*},
    short_weierstrass::{Affine, Projective},
    AffineRepr, CurveGroup, PrimeGroup,
};
use ark_ff::{BigInteger, PrimeField, Zero, One, AdditiveGroup};
use ark_std::{ops::Mul, UniformRand, test_rng, vec::Vec};
use std::time::Instant;

pub fn glv_scalar_decomposition<P: GLVConfig>() {
    let mut rng = ark_std::test_rng();
    for _i in 0..100 {
        let k = P::ScalarField::rand(&mut rng);

        let ((is_k1_positive, k1), (is_k2_positive, k2)) =
            <P as GLVConfig>::scalar_decomposition(k);

        if is_k1_positive && is_k2_positive {
            assert_eq!(k1 + k2 * P::LAMBDA, k);
        }
        if is_k1_positive && !is_k2_positive {
            assert_eq!(k1 - k2 * P::LAMBDA, k);
        }
        if !is_k1_positive && is_k2_positive {
            assert_eq!(-k1 + k2 * P::LAMBDA, k);
        }
        if !is_k1_positive && !is_k2_positive {
            assert_eq!(-k1 - k2 * P::LAMBDA, k);
        }

        // check if k1 and k2 are indeed small.
        // We add 2 to the expected max bits to account for the slightly looser bounds 
        // in some curves like secp256k1/secq256k1 where the lattice vectors can result 
        // in a decomposed scalar of up to 130 bits.
        let expected_max_bits = P::ScalarField::MODULUS_BIT_SIZE.div_ceil(2) + 2;
        assert!(
            k1.into_bigint().num_bits() <= expected_max_bits as u32,
            "k1 has {} bits",
            k1.into_bigint().num_bits()
        );
        assert!(
            k2.into_bigint().num_bits() <= expected_max_bits as u32,
            "k2 has {} bits",
            k2.into_bigint().num_bits()
        );
    }
}

pub fn glv_endomorphism_eigenvalue<P: GLVConfig>() {
    let g = Projective::<P>::generator();
    let endo_g = <P as GLVConfig>::endomorphism(&g);
    assert_eq!(endo_g, g.mul(P::LAMBDA));
}

pub fn glv_projective<P: GLVConfig>() {
    // check that glv_mul indeed computes the scalar multiplication
    let mut rng = ark_std::test_rng();

    let g = Projective::<P>::generator();
    for _i in 0..100 {
        let k = P::ScalarField::rand(&mut rng);

        let k_g = <P as GLVConfig>::glv_mul_projective(g, k);
        let k_g_2 = double_and_add(&g, k.into_bigint());
        assert_eq!(k_g, k_g_2);
    }
}

pub fn glv_affine<P: GLVConfig>() {
    // check that glv_mul indeed computes the scalar multiplication
    let mut rng = ark_std::test_rng();

    let g = Affine::<P>::generator();
    for _i in 0..100 {
        let k = P::ScalarField::rand(&mut rng);

        let k_g = <P as GLVConfig>::glv_mul_affine(g, k);
        let k_g_2 = double_and_add_affine(&g, k.into_bigint()).into_affine();
        assert_eq!(k_g, k_g_2);
    }
}

pub fn jsf_reconstructs_the_scalars<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    for _ in 0..2000 {
        let k1 = P::ScalarField::rand(rng);
        let k2 = P::ScalarField::rand(rng);
        let digits = joint_sparse_form(k1.into_bigint().as_ref(), k2.into_bigint().as_ref());

        // digits are most-significant first: acc = 2*acc + digit.
        let mut a1 = P::ScalarField::zero();
        let mut a2 = P::ScalarField::zero();
        for (u1, u2) in digits {
            a1.double_in_place();
            a2.double_in_place();
            match u1 {
                1 => a1 += P::ScalarField::one(),
                -1 => a1 -= P::ScalarField::one(),
                _ => {},
            }
            match u2 {
                1 => a2 += P::ScalarField::one(),
                -1 => a2 -= P::ScalarField::one(),
                _ => {},
            }
        }
        assert_eq!(a1, k1);
        assert_eq!(a2, k2);
    }
}

pub fn jsf_mul_matches_shamir_and_naive<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    for _ in 0..300 {
        let b1 = Projective::<P>::rand(rng);
        let b2 = Projective::<P>::rand(rng);
        let k1 = P::ScalarField::rand(rng);
        let k2 = P::ScalarField::rand(rng);

        let naive = b1.mul(k1) + b2.mul(k2);
        assert_eq!(binary_scalar_mul_jsf(b1, k1, b2, k2), naive);
        assert_eq!(binary_scalar_mul_shamir(b1, k1, b2, k2), naive);
        assert_eq!(
            binary_scalar_mul_jsf_affine(b1.into_affine(), k1, b2.into_affine(), k2),
            naive
        );
    }
}

pub fn jsf_recodes_full_width_limbs<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let cases: [(u128, u128); 7] = [
        (0, 0),
        (1, 0),
        (u128::MAX, 0),
        (u128::MAX, u128::MAX),
        (u128::MAX, 1),
        (0x8000_0000_0000_0000_0000_0000_0000_0000, u128::MAX),
        (0xDEAD_BEEF_DEAD_BEEF_FFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_0000_0000_FFFF_FFFF_FFFF_FFFF),
    ];
    let to_limbs = |x: u128| [x as u64, (x >> 64) as u64];
    for (k1, k2) in cases {
        let digits = joint_sparse_form(&to_limbs(k1), &to_limbs(k2));
        let (mut a1, mut a2) = (0u128, 0u128);
        for (u1, u2) in digits {
            a1 = a1.wrapping_mul(2).wrapping_add(u1 as i128 as u128);
            a2 = a2.wrapping_mul(2).wrapping_add(u2 as i128 as u128);
        }
        assert_eq!((a1, a2), (k1, k2), "JSF reconstruction failed for ({k1:#x}, {k2:#x})");
    }
}

pub fn jsf_mul_handles_edge_scalars<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    let b1 = Projective::<P>::rand(rng);
    let b2 = Projective::<P>::rand(rng);
    let specials = [P::ScalarField::zero(), P::ScalarField::one(), -P::ScalarField::one(), P::LAMBDA];
    for &k1 in &specials {
        for &k2 in &specials {
            let naive = b1.mul(k1) + b2.mul(k2);
            assert_eq!(binary_scalar_mul_jsf(b1, k1, b2, k2), naive);
            assert_eq!(binary_scalar_mul_shamir(b1, k1, b2, k2), naive);
            assert_eq!(
                binary_scalar_mul_jsf_affine(b1.into_affine(), k1, b2.into_affine(), k2),
                naive
            );
        }
    }
}

pub fn glv_mul_identity_point<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let id = Projective::<P>::zero();
    for k in [P::ScalarField::one(), -P::ScalarField::one(), P::LAMBDA, P::ScalarField::from(12345u64)] {
        assert_eq!(P::glv_mul_projective(id, k), Projective::<P>::zero());
        assert!(P::glv_mul_affine(id.into_affine(), k).is_zero());
    }
}

pub fn glv_mul_handles_edge_scalars<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    let p = Projective::<P>::rand(rng);
    let cases = [
        P::ScalarField::zero(),
        P::ScalarField::one(),
        -P::ScalarField::one(),
        P::LAMBDA,
        -P::LAMBDA,
    ];
    for k in cases {
        assert_eq!(P::glv_mul_projective(p, k), p.mul(k));
    }
}

fn compare<T: PartialEq>(
    n: usize,
    tag: &str,
    label_a: &str,
    a: impl FnOnce() -> T,
    label_b: &str,
    b: impl FnOnce() -> T,
) {
    let t = Instant::now();
    let res_a = a();
    let time_a = t.elapsed();

    let t = Instant::now();
    let res_b = b();
    let time_b = t.elapsed();

    assert!(res_a == res_b, "{tag}: implementations disagree");
    println!(
        "{tag} over {n}: {label_a} {time_a:?}, {label_b} {time_b:?}  ({:.2}x)",
        time_a.as_secs_f64() / time_b.as_secs_f64()
    );
}

pub fn jsf_affine_vs_projective<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    let n = 1000;
    let b1: Vec<Affine<P>> = (0..n).map(|_| Projective::<P>::rand(rng).into_affine()).collect();
    let b2: Vec<Affine<P>> = (0..n).map(|_| Projective::<P>::rand(rng).into_affine()).collect();
    let k1: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();
    let k2: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();

    compare(
        n,
        "jsf (affine bases)",
        "projective",
        || {
            let mut acc = Projective::<P>::zero();
            for i in 0..n {
                acc += binary_scalar_mul_jsf(b1[i].into_group(), k1[i], b2[i].into_group(), k2[i]);
            }
            acc
        },
        "mixed-add",
        || {
            let mut acc = Projective::<P>::zero();
            for i in 0..n {
                acc += binary_scalar_mul_jsf_affine(b1[i], k1[i], b2[i], k2[i]);
            }
            acc
        },
    );
}

pub fn jsf_vs_shamir<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    let rng = &mut test_rng();
    let n = 1000;
    let b1: Vec<Projective<P>> = (0..n).map(|_| Projective::<P>::rand(rng)).collect();
    let b2: Vec<Projective<P>> = (0..n).map(|_| Projective::<P>::rand(rng)).collect();
    let k1: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();
    let k2: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();

    compare(
        n,
        "b1*k1+b2*k2",
        "shamir",
        || {
            let mut acc = Projective::<P>::zero();
            for i in 0..n {
                acc += binary_scalar_mul_shamir(b1[i], k1[i], b2[i], k2[i]);
            }
            acc
        },
        "jsf",
        || {
            let mut acc = Projective::<P>::zero();
            for i in 0..n {
                acc += binary_scalar_mul_jsf(b1[i], k1[i], b2[i], k2[i]);
            }
            acc
        },
    );
}

pub fn fast_decomposition_throughput<P: ark_ec::short_weierstrass::SWCurveConfig + GLVConfig>() {
    // Run only when FAST_DECOMP is set.
    let Some(fd) = P::FAST_DECOMP else { return };
    let rng = &mut test_rng();
    let n = 10000;
    let scalars: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(rng)).collect();
    let lambda = P::LAMBDA;
    let signed = |(pos, mag): (bool, P::ScalarField)| if pos { mag } else { -mag };

    compare(
        n,
        "scalar_decomposition",
        "generic",
        || {
            let mut acc = P::ScalarField::zero();
            for s in &scalars {
                let (k1, k2) = generic_scalar_decomposition::<P>(*s);
                acc += signed(k1) + lambda * signed(k2);
            }
            acc
        },
        "fast",
        || {
            let mut acc = P::ScalarField::zero();
            for s in &scalars {
                let (k1, k2) = fast_scalar_decomposition::<P>(*s, &fd);
                acc += signed(k1) + lambda * signed(k2);
            }
            acc
        },
    );
}
