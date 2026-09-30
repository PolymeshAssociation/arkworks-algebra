//! Assembly Montgomery kernels for 4-limb moduli of the Pasta shape, `modulus[2] = 0` and
//! `modulus[3] = 2^62`, which both Pasta moduli have. The `MontConfig` derive routes
//! `add_assign`, `sub_assign` and `double_in_place` of such a field here whenever
//! [`pasta_asm_compiled`] holds, the `asm` feature on AArch64 or x86-64, and `mul_assign` and
//! `square_in_place` whenever [`pasta_asm_available`] holds, which on x86-64 also needs a CPU with
//! BMI2 and ADX. Every kernel requires canonical inputs and returns
//! the same canonical element as the portable path. On an Apple M3 Max the AArch64 kernels make
//! Pallas mixed addition 1.3x, a GLV scalar multiplication 1.45x and MSMs of `2^8` to `2^16`
//! points 1.35-1.4x faster than the portable path.

use super::{Fp, MontBackend, MontConfig};

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(all(feature = "asm", target_arch = "aarch64"))]
use aarch64 as kernel;

#[cfg(all(feature = "asm", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86_64;
#[cfg(all(feature = "asm", target_arch = "x86_64"))]
use x86_64 as kernel;

#[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
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

/// Whether a 4-limb modulus has the shape the kernels hardcode.
pub const fn is_pasta_shape(modulus: &[u64; 4]) -> bool {
    modulus[2] == 0 && modulus[3] == 1 << 62
}

/// The kernel operands of a Pasta-shape [`MontConfig`].
trait PastaParams: MontConfig<4> {
    const APPLIES: bool = is_pasta_shape(&Self::MODULUS.0);
    /// `[modulus[0..4], INV]`.
    const PARAMS: [u64; 5] = [
        Self::MODULUS.0[0],
        Self::MODULUS.0[1],
        Self::MODULUS.0[2],
        Self::MODULUS.0[3],
        Self::INV,
    ];
}

impl<T: MontConfig<4>> PastaParams for T {}

/// Whether the kernels are compiled in. Enough for addition, subtraction and doubling, which use
/// only baseline instructions.
#[doc(hidden)]
#[inline(always)]
pub const fn pasta_asm_compiled() -> bool {
    cfg!(all(
        feature = "asm",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))
}

/// Whether the multiply and square kernels are compiled in and usable on this CPU. Cached after
/// the first call on x86-64, a compile-time constant elsewhere.
#[doc(hidden)]
#[inline(always)]
#[allow(clippy::missing_const_for_fn)]
pub fn pasta_asm_available() -> bool {
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    {
        true
    }
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    {
        super::asm_backend_active()
    }
    #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
    {
        false
    }
}

type F<T> = Fp<MontBackend<T, 4>, 4>;

#[inline(always)]
const fn params<T: MontConfig<4>>() -> &'static [u64; 5] {
    const { &<T as PastaParams>::PARAMS }
}

#[inline(always)]
fn debug_check<T: MontConfig<4>>(values: &[&F<T>]) {
    debug_assert!(
        <T as PastaParams>::APPLIES,
        "modulus is not of the Pasta shape"
    );
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

/// `a *= b` by the Pasta-shape kernel. Emitted by the `MontConfig` derive behind
/// [`pasta_asm_available`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_mul_assign<T: MontConfig<4>>(a: &mut F<T>, b: &F<T>) {
    debug_check::<T>(&[a, b]);
    debug_assert!(
        pasta_asm_available(),
        "Pasta multiply kernel is not available"
    );
    (a.0).0 = kernel::mul(&(a.0).0, &(b.0).0, params::<T>());
}

/// `a = a^2` by the Pasta-shape kernel; see [`pasta_mul_assign`].
#[doc(hidden)]
#[inline(always)]
pub fn pasta_square_in_place<T: MontConfig<4>>(a: &mut F<T>) {
    debug_check::<T>(&[a]);
    debug_assert!(
        pasta_asm_available(),
        "Pasta square kernel is not available"
    );
    (a.0).0 = kernel::square(&(a.0).0, params::<T>());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdditiveGroup, BigInt, BigInteger, Field, PrimeField, UniformRand};
    use ark_std::{test_rng, vec::Vec};
    use num_bigint::BigUint;

    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337"]
    #[generator = "5"]
    pub struct PallasBase;

    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941647379679742748393362948097"]
    #[generator = "5"]
    pub struct VestaBase;

    /// Canonical Montgomery representations that stress the carry and borrow chains and the
    /// final conditional subtraction.
    fn edge_values<T: MontConfig<4>>() -> Vec<F<T>> {
        let p = T::MODULUS;
        let mut raw = ark_std::vec![
            BigInt::<4>::zero(),
            BigInt::one(),
            BigInt::from(2u64),
            T::R,
            T::R2,
            BigInt([u64::MAX, 0, 0, 0]),
            BigInt([u64::MAX, u64::MAX, 0, 0]),
            BigInt([u64::MAX, u64::MAX, u64::MAX, 0]),
            BigInt([0, 0, 0, p.0[3] - 1]),
            BigInt([0, 0, u64::MAX, p.0[3] - 1]),
            BigInt([u64::MAX, u64::MAX, u64::MAX, p.0[3] - 1]),
        ];
        for k in 1..=4u64 {
            let mut v = p;
            v.sub_with_borrow(&BigInt::from(k));
            raw.push(v);
        }
        let half = {
            let mut h = p;
            h.div2();
            h
        };
        for d in [0u64, 1] {
            let mut v = half;
            v.add_with_carry(&BigInt::from(d));
            raw.push(v);
        }
        raw.into_iter()
            .inspect(|v| assert!(*v < p))
            .map(F::<T>::new_unchecked)
            .collect()
    }

    fn big(x: &F<impl MontConfig<4>>) -> BigUint {
        BigUint::from((x.0).0[0])
            | BigUint::from((x.0).0[1]) << 64
            | BigUint::from((x.0).0[2]) << 128
            | BigUint::from((x.0).0[3]) << 192
    }

    /// Each wrapper against `BigUint` arithmetic on the Montgomery representations.
    fn check<T: MontConfig<4>>() {
        assert!(<T as PastaParams>::APPLIES);
        if !pasta_asm_compiled() {
            return;
        }
        let multiplicative = pasta_asm_available();
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
            if multiplicative {
                let mut r = *a;
                pasta_square_in_place(&mut r);
                canonical(&r);
                assert_eq!(big(&r), &ab * &ab * &r_inv % &p, "square {a}");
            }
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
                if multiplicative {
                    let mut r = *a;
                    pasta_mul_assign(&mut r, b);
                    canonical(&r);
                    assert_eq!(big(&r), &ab * &bb * &r_inv % &p, "{a} * {b}");
                }
            }
        }
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
            }
        }
        run::<PallasBase>();
        run::<VestaBase>();
    }

    #[test]
    fn shape_predicate() {
        assert!(is_pasta_shape(&PallasBase::MODULUS.0));
        assert!(is_pasta_shape(&VestaBase::MODULUS.0));
        assert!(!is_pasta_shape(&[1, 2, 0, 3]));
        assert!(!is_pasta_shape(&[1, 2, 7, 1 << 62]));
    }
}
