//! AArch64 assembly Montgomery kernels for 4-limb moduli of the Pasta shape
//! ([`is_pasta_shape`]). Whenever [`pasta_asm_compiled`] holds, the `asm` feature on AArch64, the
//! `MontConfig` derive routes `add_assign`, `sub_assign`, `double_in_place`, `mul_assign`,
//! `square_in_place` and two-term `sum_of_products` of such a field here, the last to a fused
//! kernel with one Montgomery reduction. Every kernel requires canonical inputs and returns the
//! same canonical element as the portable path. On an Apple M3 Max the kernels make Pallas mixed
//! addition 1.54x, addition 1.22x, doubling 1.52x, a GLV scalar multiplication 1.47x and MSMs of
//! `2^8` to `2^16` points 1.15-1.16x faster than the portable path with loose arithmetic. On
//! x86-64 Pasta fields take the portable path with loose arithmetic. On an i9-12900H, with and
//! without BMI2 and ADX enabled at compile time, Zakura's MULX/ADX Pasta kernels made MSMs
//! 0.78-0.91x and square roots 0.70-0.84x as fast as it.

#[cfg(doc)]
use super::loose::is_pasta_shape;
use super::{
    loose::{applies, params},
    Fp, MontBackend, MontConfig,
};

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
#[allow(unsafe_code)]
pub(super) mod aarch64;
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
pub(super) use aarch64 as kernel;

#[cfg(not(all(feature = "asm", target_arch = "aarch64")))]
mod kernel {
    type Limbs = [u64; 4];
    const ABSENT: &str = "Pasta kernels are not compiled in";

    pub(super) fn add(_: &Limbs, _: &Limbs, _: &[u64; 5]) -> Limbs {
        unreachable!("{ABSENT}")
    }

    pub(super) fn sub(_: &Limbs, _: &Limbs, _: &[u64; 5]) -> Limbs {
        unreachable!("{ABSENT}")
    }

    pub(super) fn mul(_: &Limbs, _: &Limbs, _: &[u64; 5]) -> Limbs {
        unreachable!("{ABSENT}")
    }

    pub(super) fn square(_: &Limbs, _: &[u64; 5]) -> Limbs {
        unreachable!("{ABSENT}")
    }
}

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use kernel::sum_of_two_products as fused_kernel;

#[cfg(not(all(feature = "asm", target_arch = "aarch64")))]
fn fused_kernel(_: &[u64; 4], _: &[u64; 4], _: &[u64; 4], _: &[u64; 4], _: &[u64; 5]) -> [u64; 4] {
    unreachable!("the fused Pasta kernel is not compiled in")
}

/// Whether the kernels are compiled in: the `asm` feature on AArch64.
#[doc(hidden)]
#[inline(always)]
pub const fn pasta_asm_compiled() -> bool {
    cfg!(all(feature = "asm", target_arch = "aarch64"))
}

type F<T> = Fp<MontBackend<T, 4>, 4>;

#[inline(always)]
fn debug_check<T: MontConfig<4>>(values: &[&F<T>]) {
    debug_assert!(applies::<T, 4>(), "modulus is not of the Pasta shape");
    debug_assert!(pasta_asm_compiled(), "Pasta kernels are not compiled in");
    for v in values {
        debug_assert!(v.0 < T::MODULUS, "Pasta kernels require canonical inputs");
    }
}

/// `a += b` by the Pasta-shape kernel. Emitted by the `MontConfig` derive behind
/// [`pasta_asm_compiled`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_add_assign<T: MontConfig<4>>(a: &mut F<T>, b: &F<T>) {
    debug_check::<T>(&[a, b]);
    (a.0).0 = kernel::add(&(a.0).0, &(b.0).0, params::<T>());
}

/// `a -= b` by the Pasta-shape kernel; see [`pasta_add_assign`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_sub_assign<T: MontConfig<4>>(a: &mut F<T>, b: &F<T>) {
    debug_check::<T>(&[a, b]);
    (a.0).0 = kernel::sub(&(a.0).0, &(b.0).0, params::<T>());
}

/// `a = 2a` by the Pasta-shape addition kernel; see [`pasta_add_assign`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_double_in_place<T: MontConfig<4>>(a: &mut F<T>) {
    debug_check::<T>(&[a]);
    (a.0).0 = kernel::add(&(a.0).0, &(a.0).0, params::<T>());
}

/// `a *= b` by the Pasta-shape kernel; see [`pasta_add_assign`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_mul_assign<T: MontConfig<4>>(a: &mut F<T>, b: &F<T>) {
    debug_check::<T>(&[a, b]);
    (a.0).0 = kernel::mul(&(a.0).0, &(b.0).0, params::<T>());
}

/// `a = a^2` by the Pasta-shape kernel; see [`pasta_add_assign`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_square_in_place<T: MontConfig<4>>(a: &mut F<T>) {
    debug_check::<T>(&[a]);
    (a.0).0 = kernel::square(&(a.0).0, params::<T>());
}

/// `a[0] * b[0] + a[1] * b[1]` by the fused Pasta-shape kernel, one Montgomery reduction for
/// both products. Both slices hold exactly two elements. Emitted by the `MontConfig` derive for
/// two-term `sum_of_products` behind [`pasta_asm_compiled`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_sum_of_two_products<T: MontConfig<4>>(a: &[F<T>], b: &[F<T>]) -> F<T> {
    debug_assert!(a.len() == 2 && b.len() == 2);
    debug_check::<T>(&[&a[0], &a[1], &b[0], &b[1]]);
    let mut r = a[0];
    (r.0).0 = fused_kernel(
        &(a[0].0).0,
        &(b[0].0).0,
        &(a[1].0).0,
        &(b[1].0).0,
        params::<T>(),
    );
    r
}

#[cfg(test)]
mod tests {
    use super::{
        super::loose::tests::{edge_values, PallasBase, VestaBase},
        *,
    };
    use crate::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField, UniformRand};
    use ark_std::test_rng;
    use num_bigint::BigUint;

    fn big(x: &F<impl MontConfig<4>>) -> BigUint {
        BigUint::from((x.0).0[0])
            | BigUint::from((x.0).0[1]) << 64
            | BigUint::from((x.0).0[2]) << 128
            | BigUint::from((x.0).0[3]) << 192
    }

    /// Each wrapper against `BigUint` arithmetic on the Montgomery representations.
    fn check<T: MontConfig<4>>() {
        assert!(applies::<T, 4>());
        if !pasta_asm_compiled() {
            return;
        }
        let p = BigUint::from(T::MODULUS);
        let r_inv = (BigUint::from(1u8) << 256u32).modpow(&(&p - 2u8), &p);
        let mut values = edge_values::<T>();
        let mut rng = test_rng();
        values.extend((0..64).map(|_| F::<T>::rand(&mut rng)));
        let canonical = |x: &F<T>| assert!(x.0 < T::MODULUS);
        for a in &values {
            let (ab, mut r) = (big(a), *a);
            pasta_double_in_place(&mut r);
            canonical(&r);
            assert_eq!(big(&r), (&ab << 1) % &p, "double {a}");
            let mut r = *a;
            pasta_square_in_place(&mut r);
            canonical(&r);
            assert_eq!(big(&r), &ab * &ab * &r_inv % &p, "square {a}");
            for b in &values {
                let bb = big(b);
                let mut r = *a;
                pasta_add_assign(&mut r, b);
                canonical(&r);
                assert_eq!(big(&r), (&ab + &bb) % &p, "{a} + {b}");
                let mut r = *a;
                pasta_sub_assign(&mut r, b);
                canonical(&r);
                assert_eq!(big(&r), (&ab + &p - &bb) % &p, "{a} - {b}");
                let mut r = *a;
                pasta_mul_assign(&mut r, b);
                canonical(&r);
                assert_eq!(big(&r), &ab * &bb * &r_inv % &p, "{a} * {b}");
            }
        }
        let fused = |a: &[F<T>; 2], b: &[F<T>; 2]| {
            let mut r = a[0];
            (r.0).0 = fused_kernel(
                &(a[0].0).0,
                &(b[0].0).0,
                &(a[1].0).0,
                &(b[1].0).0,
                params::<T>(),
            );
            r
        };
        let n = values.len();
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                for c in [values[(i + j + 1) % n], values[(i * 7 + j * 3) % n], *a, *b] {
                    let d = values[(i + 3 * j + 5) % n];
                    let r = fused(&[*a, c], &[*b, d]);
                    canonical(&r);
                    let expected = (big(a) * big(b) + big(&c) * big(&d)) * &r_inv % &p;
                    assert_eq!(big(&r), expected, "{a} * {b} + {c} * {d}");
                }
            }
        }
        let top = F::<T>::new_unchecked({
            let mut v = T::MODULUS;
            v.sub_with_borrow(&BigInt::one());
            v
        });
        let r = fused(&[top, top], &[top, top]);
        canonical(&r);
        assert_eq!(big(&r), (big(&top) * big(&top) * 2u8) * &r_inv % &p);
    }

    #[test]
    fn kernels_match_biguint() {
        check::<PallasBase>();
        check::<VestaBase>();
    }

    /// The derived operators take the kernels when available and must agree with the portable
    /// multiply and square.
    #[test]
    fn derived_operators_match_portable() {
        fn run<T: MontConfig<4>>() {
            let mut rng = test_rng();
            for _ in 0..10_000 {
                let a = F::<T>::rand(&mut rng);
                let b = F::<T>::rand(&mut rng);
                assert_eq!(a * b, a.mul_portable(&b));
                assert_eq!(a.square(), a.square_portable());
                assert_eq!((a + b) - b, a);
                assert_eq!(a.double(), a + a);
                assert_eq!(F::<T>::from_bigint(a.into_bigint()), Some(a));
                let (c, d) = (F::<T>::rand(&mut rng), F::<T>::rand(&mut rng));
                assert_eq!(F::<T>::sum_of_products(&[a, c], &[b, d]), a * b + c * d);
            }
        }
        run::<PallasBase>();
        run::<VestaBase>();
    }
}
