use ark_algebra_test_templates::*;
use ark_ff::{
    biginteger::{BigInt, BigInteger, BigInteger256},
    fields::{FftField, Field, Fp6Config, PrimeField},
    One, UniformRand, Zero,
};
use ark_std::{cmp::Ordering, ops::MulAssign};

use crate::{Fq, Fq12, Fq2, Fq6, Fq6Config, Fr};

test_field!(fr; Fr; mont_prime_field);
test_field!(fq; Fq; mont_prime_field);
test_field!(fq2; Fq2);
test_field!(fq6; Fq6);
test_field!(fq12; Fq12);

#[test]
fn test_fq_repr_from() {
    assert_eq!(BigInteger256::from(100u64), BigInt::new([100, 0, 0, 0]));
}

#[test]
fn test_fq_repr_is_odd() {
    assert!(!BigInteger256::from(0u64).is_odd());
    assert!(BigInteger256::from(0u64).is_even());
    assert!(BigInteger256::from(1u64).is_odd());
    assert!(!BigInteger256::from(1u64).is_even());
    assert!(!BigInteger256::from(324834872u64).is_odd());
    assert!(BigInteger256::from(324834872u64).is_even());
    assert!(BigInteger256::from(324834873u64).is_odd());
    assert!(!BigInteger256::from(324834873u64).is_even());
}

#[test]
fn test_fq_repr_is_zero() {
    assert!(BigInteger256::from(0u64).is_zero());
    assert!(!BigInteger256::from(1u64).is_zero());
    assert!(!BigInt::new([0, 0, 1, 0]).is_zero());
}

#[test]
fn test_fq_repr_num_bits() {
    let mut a = BigInteger256::from(0u64);
    assert_eq!(0, a.num_bits());
    a = BigInteger256::from(1u64);
    for i in 1..257 {
        assert_eq!(i, a.num_bits());
        a.mul2();
    }
    assert_eq!(0, a.num_bits());
}

#[test]
fn test_fq_num_bits() {
    assert_eq!(Fq::MODULUS_BIT_SIZE, 254);
}

#[test]
fn test_fq_root_of_unity() {
    assert_eq!(Fq::TWO_ADICITY, 1);
    assert_eq!(
        Fq::GENERATOR.pow([
            0x9e10460b6c3e7ea3,
            0xcbc0b548b438e546,
            0xdc2822db40c0ac2e,
            0x183227397098d014,
        ]),
        Fq::TWO_ADIC_ROOT_OF_UNITY
    );
    assert_eq!(
        Fq::TWO_ADIC_ROOT_OF_UNITY.pow([1 << Fq::TWO_ADICITY]),
        Fq::one()
    );
    assert!(Fq::GENERATOR.sqrt().is_none());
}

#[test]
fn test_fq_ordering() {
    // BigInteger256's ordering is well-tested, but we still need to make sure the
    // Fq elements aren't being compared in Montgomery form.
    for i in 0..100u64 {
        assert!(Fq::from(BigInteger256::from(i + 1)) > Fq::from(BigInteger256::from(i)));
    }
}

#[test]
fn test_fq_legendre() {
    use ark_ff::fields::LegendreSymbol::*;

    assert_eq!(QuadraticResidue, Fq::one().legendre());
    assert_eq!(Zero, Fq::zero().legendre());
    assert_eq!(
        QuadraticResidue,
        Fq::from(BigInteger256::from(4u64)).legendre()
    );
    assert_eq!(
        QuadraticNonResidue,
        Fq::from(BigInteger256::from(5u64)).legendre()
    );
}

#[test]
fn test_fq2_ordering() {
    let mut a = Fq2::new(Fq::zero(), Fq::zero());
    let mut b = a.clone();

    assert!(a.cmp(&b) == Ordering::Equal);
    b.c0 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c0 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Equal);
    b.c1 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c0 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Less);
    a.c1 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Greater);
    b.c0 += &Fq::one();
    assert!(a.cmp(&b) == Ordering::Equal);
}

#[test]
fn test_fq2_basics() {
    assert_eq!(Fq2::new(Fq::zero(), Fq::zero(),), Fq2::zero());
    assert_eq!(Fq2::new(Fq::one(), Fq::zero(),), Fq2::one());
    assert!(Fq2::zero().is_zero());
    assert!(!Fq2::one().is_zero());
    assert!(!Fq2::new(Fq::zero(), Fq::one(),).is_zero());
}

#[test]
fn test_fq2_legendre() {
    use ark_ff::fields::LegendreSymbol::*;

    assert_eq!(Zero, Fq2::zero().legendre());
    // i^2 = -1
    let mut m1 = -Fq2::one();
    assert_eq!(QuadraticResidue, m1.legendre());
    Fq6Config::mul_fp2_by_nonresidue_in_place(&mut m1);
    assert_eq!(QuadraticNonResidue, m1.legendre());
}

#[test]
fn test_fq6_mul_by_1() {
    let mut rng = ark_std::test_rng();

    for _ in 0..1000 {
        let c1 = Fq2::rand(&mut rng);
        let mut a = Fq6::rand(&mut rng);
        let mut b = a;

        a.mul_by_1(&c1);
        b *= &Fq6::new(Fq2::zero(), c1, Fq2::zero());

        assert_eq!(a, b);
    }
}

#[test]
fn test_fq6_mul_by_01() {
    let mut rng = ark_std::test_rng();

    for _ in 0..1000 {
        let c0 = Fq2::rand(&mut rng);
        let c1 = Fq2::rand(&mut rng);
        let mut a = Fq6::rand(&mut rng);
        let mut b = a;

        a.mul_by_01(&c0, &c1);
        b *= &Fq6::new(c0, c1, Fq2::zero());

        assert_eq!(a, b);
    }
}

#[test]
fn test_fq12_mul_by_014() {
    let mut rng = ark_std::test_rng();

    for _ in 0..1000 {
        let c0 = Fq2::rand(&mut rng);
        let c1 = Fq2::rand(&mut rng);
        let c5 = Fq2::rand(&mut rng);
        let mut a = Fq12::rand(&mut rng);
        let mut b = a;

        a.mul_by_014(&c0, &c1, &c5);
        b.mul_assign(&Fq12::new(
            Fq6::new(c0, c1, Fq2::zero()),
            Fq6::new(Fq2::zero(), c5, Fq2::zero()),
        ));

        assert_eq!(a, b);
    }
}

#[test]
fn test_fq12_mul_by_034() {
    let mut rng = ark_std::test_rng();

    for _ in 0..1000 {
        let c0 = Fq2::rand(&mut rng);
        let c3 = Fq2::rand(&mut rng);
        let c4 = Fq2::rand(&mut rng);
        let mut a = Fq12::rand(&mut rng);
        let mut b = a;

        a.mul_by_034(&c0, &c3, &c4);
        b.mul_assign(&Fq12::new(
            Fq6::new(c0, Fq2::zero(), Fq2::zero()),
            Fq6::new(c3, c4, Fq2::zero()),
        ));

        assert_eq!(a, b);
    }
}

/// BN254's base field has 2 spare bits, so the sparse line products and `Fp6::mul_by_12` take
/// their Karatsuba forms instead of six-term `sum_of_products`. Each matches the full product
/// with the sparse element, on random inputs and with every coordinate `p - 1`.
#[test]
fn sparse_products_karatsuba_forms_match_full_multiplication() {
    use ark_ff::{AdditiveGroup, One, PrimeField};
    use ark_std::{test_rng, vec::Vec, UniformRand};

    assert!(256 - Fq::MODULUS_BIT_SIZE < 3);
    let mut rng = test_rng();
    let max = Fq2::new(-Fq::one(), -Fq::one());
    let z = Fq2::ZERO;
    let one = Fq2::one();
    let mut cases: Vec<(Fq12, [Fq2; 6])> = (0..200)
        .map(|_| {
            (
                Fq12::rand(&mut rng),
                core::array::from_fn(|_| Fq2::rand(&mut rng)),
            )
        })
        .collect();
    cases.push((
        Fq12::new(Fq6::new(max, max, max), Fq6::new(max, max, max)),
        [max; 6],
    ));

    for (f, l) in cases {
        let line_014 = |a: Fq2, b: Fq2, c: Fq2| Fq12::new(Fq6::new(a, b, z), Fq6::new(z, c, z));
        let line_034 = |a: Fq2, b: Fq2, c: Fq2| Fq12::new(Fq6::new(a, z, z), Fq6::new(b, c, z));

        let mut a = f;
        a.mul_by_014_c4_one(&l[0], &l[1]);
        assert_eq!(a, f * line_014(l[0], l[1], one), "mul_by_014_c4_one");

        let mut a = f;
        a.mul_by_034_c0_one(&l[0], &l[1]);
        assert_eq!(a, f * line_034(one, l[0], l[1]), "mul_by_034_c0_one");

        let mut a = f;
        a.mul_by_014_pair(&l[0], &l[1], &l[2], &l[3], &l[4], &l[5]);
        let expected = f * line_014(l[0], l[1], l[2]) * line_014(l[3], l[4], l[5]);
        assert_eq!(a, expected, "mul_by_014_pair");

        let mut a = f;
        a.mul_by_034_pair(&l[0], &l[1], &l[2], &l[3], &l[4], &l[5]);
        let expected = f * line_034(l[0], l[1], l[2]) * line_034(l[3], l[4], l[5]);
        assert_eq!(a, expected, "mul_by_034_pair");

        let mut a = f.c0;
        a.mul_by_12(&l[0], &l[1]);
        assert_eq!(a, f.c0 * Fq6::new(z, l[0], l[1]), "mul_by_12");
    }
}
