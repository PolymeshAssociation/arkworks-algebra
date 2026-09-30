//! Deferred Montgomery reduction: unreduced `2N`-limb products accumulated in one buffer and
//! reduced once, for sums of many products.
//!
//! Port of Zakura `deferred.rs` (`Product::mul_accumulate`, `accumulate`, `partial_reduce`),
//! generic over `N` instead of specialized to 4 limbs. Per term the naive path pays a
//! schoolbook multiply, a Montgomery reduction and a conditional subtraction; this pays only
//! the schoolbook multiply, so the reduction cost amortizes over the whole inner product.
//!
//! The accumulator holds `sum(a_i * b_i)` unreduced, as `2N` limbs plus a 64-bit carry, so
//! it takes `2^64` terms to overflow. [`MontAccumulator::reduce`] folds the top limb and the
//! carry back under `2^{64(2N-1)}` using their residues, then runs the same REDC loop as
//! `square_in_place`. Under the `asm` feature on AArch64 the 4-limb accumulate and fold run in
//! the inline assembly of `deferred/aarch64.rs`, which on an Apple M3 Max takes a Pallas inner
//! product from 6.3 to 4.5 ns per term.
//!
//! # Sources
//!
//! - Zakura `deferred.rs`, whose `Product::mul_accumulate`, `accumulate` and
//!   `partial_reduce` this generalizes over `N`:
//!   <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/deferred.rs>
//! - Zakura `deferred/aarch64.rs` for the AArch64 kernels, cited in `deferred/aarch64.rs`
//! - Their [blog](https://zakura.com/engineering/deferred-montgomery-products/) on the technique

use super::{mont29::Mont29Params, Fp, MontBackend, MontConfig};
use crate::{biginteger::arithmetic as fa, const_helpers::MulBuffer, BigInt};
use ark_std::marker::PhantomData;

#[cfg(all(feature = "asm", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod aarch64;

/// A running sum of unreduced `2N`-limb Montgomery products.
#[derive(educe::Educe)]
#[educe(Clone, Copy, Debug)]
pub struct MontAccumulator<T: MontConfig<N>, const N: usize> {
    buf: MulBuffer<N>,
    /// Overflow beyond the `2N` limbs, in units of `2^{128N}`.
    carry: u64,
    phantom: PhantomData<T>,
}

impl<T: MontConfig<N>, const N: usize> Default for MontAccumulator<T, N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<T: MontConfig<N>, const N: usize> MontAccumulator<T, N> {
    /// The empty sum.
    pub const ZERO: Self = Self {
        buf: MulBuffer::zeroed(),
        carry: 0,
        phantom: PhantomData,
    };

    /// Adds `a * b` to the running sum, without reducing. Row `i` starts from the stored
    /// limbs, so each accumulator limb is read once; the row's carry and the previous row's
    /// one-bit overflow land in the next untouched limb.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn mul_accumulate(&mut self, a: &Fp<MontBackend<T, N>, N>, b: &Fp<MontBackend<T, N>, N>) {
        let (a, b) = (&(a.0).0, &(b.0).0);
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        if N == 4 {
            let (limbs, overflow) = aarch64::mul_accumulate(
                self.limbs8(),
                a[..4].try_into().unwrap(),
                b[..4].try_into().unwrap(),
            );
            self.set_limbs8(limbs);
            self.add_carry(overflow);
            return;
        }
        let mut overflow = 0;
        for i in 0..N {
            let mut carry = 0;
            for j in 0..N {
                *self.buf.get_mut(i + j) =
                    fa::mac_with_carry(self.buf[i + j], a[i], b[j], &mut carry);
            }
            overflow = fa::adc(self.buf.get_mut(i + N), carry, overflow);
        }
        self.add_carry(overflow);
    }

    #[inline(always)]
    fn add_carry(&mut self, overflow: u64) {
        let (carry, wrapped) = self.carry.overflowing_add(overflow);
        debug_assert!(!wrapped, "accumulator carry overflow");
        self.carry = carry;
    }

    /// The `2N = 8` limbs, for the 4-limb kernels.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn limbs8(&self) -> [u64; 8] {
        core::array::from_fn(|i| self.buf[i])
    }

    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn set_limbs8(&mut self, limbs: [u64; 8]) {
        for (i, limb) in limbs.into_iter().enumerate() {
            *self.buf.get_mut(i) = limb;
        }
    }

    /// Whether the 4-limb fold kernel's bounds hold, `B_HIGH < 2^253` and `R2 < 2^252`.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    const ASM_FOLD: bool = N == 4 && T::B_HIGH.0[N - 1] < 1 << 61 && T::R2.0[N - 1] < 1 << 60;

    /// Reduces the running sum to a canonical field element.
    pub fn reduce(mut self) -> Fp<MontBackend<T, N>, N> {
        // Fold the top limb and the carry back in through their residues, leaving a value
        // below `2^{64(2N-1)} + 2^65 m`, which `T::CAN_DEFER` keeps under `R*m`.
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        let folded = Self::ASM_FOLD && {
            let limbs = aarch64::partial_reduce(
                self.limbs8(),
                self.carry,
                T::B_HIGH.0[..4].try_into().unwrap(),
                T::R2.0[..4].try_into().unwrap(),
            );
            self.set_limbs8(limbs);
            true
        };
        #[cfg(not(all(feature = "asm", target_arch = "aarch64")))]
        let folded = false;
        if !folded {
            let hi = self.buf[2 * N - 1];
            let carry = self.carry;
            *self.buf.get_mut(2 * N - 1) = 0;
            self.add_scaled(hi, &T::B_HIGH);
            self.add_scaled(carry, &T::R2);
        }

        // Montgomery reduction over `2N` limbs, as in `square_in_place`.
        let mut carry2 = 0;
        for i in 0..N {
            let k = self.buf[i].wrapping_mul(T::INV);
            let mut carry = 0;
            fa::mac_discard(self.buf[i], k, T::MODULUS.0[0], &mut carry);
            for j in 1..N {
                *self.buf.get_mut(i + j) =
                    fa::mac_with_carry(self.buf[i + j], k, T::MODULUS.0[j], &mut carry);
            }
            carry2 = fa::adc(self.buf.get_mut(i + N), carry, carry2);
        }
        let mut r = Fp::new_unchecked(BigInt(self.buf.b1));
        if T::MODULUS_HAS_SPARE_BIT {
            r.subtract_modulus();
        } else {
            r.subtract_modulus_with_carry(carry2 != 0);
        }
        r
    }

    /// Adds `scalar * value` into the buffer, propagating the carry upward.
    #[inline]
    fn add_scaled(&mut self, scalar: u64, value: &BigInt<N>) {
        let mut carry = 0;
        for j in 0..N {
            *self.buf.get_mut(j) = fa::mac_with_carry(self.buf[j], scalar, value.0[j], &mut carry);
        }
        for j in N..(2 * N) {
            carry = fa::adc(self.buf.get_mut(j), carry, 0);
            if carry == 0 {
                return;
            }
        }
        debug_assert_eq!(carry, 0, "accumulator overflow while folding");
    }
}

/// Shortest inner product that takes the deferred path. Below it `reduce` costs more than the
/// per-term reductions it saves: on an Apple M3 Max a Pallas naive sum is 1.7x faster at 1 term
/// and 1.02x at 2. On wasm32, for a field whose naive multiply takes the radix-`2^29` path
/// ([`Mont29Params::APPLIES`]), the accumulator emulates its 64-bit products, so in wasmtime the
/// naive sum is 1.5x faster at 2 terms, 1.2x at 4, level at 8 and 1.1x slower at 16.
const fn min_deferred_len<T: MontConfig<N>, const N: usize>() -> usize {
    if cfg!(target_family = "wasm") && <T as Mont29Params<N>>::APPLIES {
        8
    } else {
        3
    }
}

/// `sum(a_i * b_i)` with one Montgomery reduction for the whole sum. Falls back to the naive
/// sum when the accumulator's bound does not hold ([`MontConfig::CAN_DEFER`]) or the sum is
/// shorter than [`min_deferred_len`].
#[inline]
pub(super) fn inner_product<T: MontConfig<N>, const N: usize>(
    a: &[Fp<MontBackend<T, N>, N>],
    b: &[Fp<MontBackend<T, N>, N>],
) -> Fp<MontBackend<T, N>, N> {
    assert_eq!(a.len(), b.len());
    if !T::CAN_DEFER || a.len() < min_deferred_len::<T, N>() {
        return a.iter().zip(b).map(|(a, b)| *a * b).sum();
    }
    deferred_sum::<T, N>(a, b)
}

/// The deferred path of [`inner_product`], kept out of line so the short-sum fallback inlines.
fn deferred_sum<T: MontConfig<N>, const N: usize>(
    a: &[Fp<MontBackend<T, N>, N>],
    b: &[Fp<MontBackend<T, N>, N>],
) -> Fp<MontBackend<T, N>, N> {
    let mut acc = MontAccumulator::<T, N>::ZERO;
    for (a, b) in a.iter().zip(b) {
        acc.mul_accumulate(a, b);
    }
    acc.reduce()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::models::fp::{Fp, MontBackend};
    use ark_std::{test_rng, vec::Vec, UniformRand};

    // Goldilocks: one limb, so no accumulator bound holds and `inner_product` must fall back.
    #[derive(MontConfig)]
    #[modulus = "18446744069414584321"]
    #[generator = "7"]
    pub struct Goldilocks;

    // 2^64 + 13: two limbs but only 65 bits, so `CAN_DEFER`'s bit condition fails as well.
    #[derive(MontConfig)]
    #[modulus = "18446744073709551629"]
    #[generator = "2"]
    pub struct Narrow;

    fn check<T: MontConfig<N>, const N: usize>() {
        assert!(!T::CAN_DEFER);
        let mut rng = test_rng();
        for len in [0usize, 1, 2, 3, 100] {
            let a: Vec<_> = (0..len)
                .map(|_| Fp::<MontBackend<T, N>, N>::rand(&mut rng))
                .collect();
            let b: Vec<_> = (0..len)
                .map(|_| Fp::<MontBackend<T, N>, N>::rand(&mut rng))
                .collect();
            let naive = a.iter().zip(&b).map(|(a, b)| *a * b).sum();
            assert_eq!(inner_product::<T, N>(&a, &b), naive, "len {len}");
        }
    }

    #[test]
    fn falls_back_when_the_bound_fails() {
        // The fields no curve crate provides: the ones the accumulator's bound rejects.
        check::<Goldilocks, 1>();
        check::<Narrow, 2>();
    }

    // The BW6-761 base field: twelve limbs, so a sum needs about `2^14` maximal products before
    // the accumulator carries past `2^{128N}`.
    #[derive(MontConfig)]
    #[modulus = "6891450384315732539396789682275657542479668912536150109513790160209623422243491736087683183289411687640864567753786613451161759120554247759349511699125301598951605099378508850372543631423596795951899700429969112842764913119068299"]
    #[generator = "2"]
    pub struct Twelve;

    /// A seeded carry `c` stands for `c 2^{128N} = c R^2`, which the reduction maps to `c`.
    #[test]
    fn carry_folds_on_twelve_limbs() {
        type F = Fp<MontBackend<Twelve, 12>, 12>;
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(Twelve::CAN_DEFER);
        }
        let mut rng = test_rng();
        let mut max = Twelve::MODULUS;
        max.0[0] -= 1;
        let extremes = [F::new_unchecked(max), F::rand(&mut rng)];
        for carry in [1u64, 7, 1 << 20, u64::MAX >> 8] {
            for a in extremes {
                let mut acc = MontAccumulator::<Twelve, 12>::ZERO;
                for _ in 0..50 {
                    acc.mul_accumulate(&a, &a);
                }
                acc.carry += carry;
                let expected = a * a * F::from(50u64) + F::from(carry);
                assert_eq!(acc.reduce(), expected, "carry {carry}");
            }
        }
    }

    // Pallas base field: the Pasta shape, whose `B_HIGH` and `R2` meet the 4-limb fold kernel's
    // bounds.
    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337"]
    #[generator = "5"]
    pub struct PallasBase;

    // secp256k1 base field: dense, but `2^256 mod p` is small, so its residues meet the bounds.
    #[derive(MontConfig)]
    #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
    #[generator = "3"]
    pub struct Secp256k1Base;

    // secp256k1 scalar field: `B_HIGH` and `R2` exceed the bounds, so the portable fold runs.
    #[derive(MontConfig)]
    #[modulus = "115792089237316195423570985008687907852837564279074904382605163141518161494337"]
    #[generator = "7"]
    pub struct Secp256k1Scalar;

    fn big<const N: usize>(limbs: &[u64]) -> num_bigint::BigUint {
        limbs
            .iter()
            .rev()
            .fold(num_bigint::BigUint::default(), |acc, l| (acc << 64u32) + *l)
    }

    /// The accumulator state against the exact sum, and `reduce` against `sum * R^{-1} mod p`,
    /// on limbs chosen to saturate every carry chain. `mul_accumulate` takes any limbs, so the
    /// inputs need not be canonical.
    fn check_exact<T: MontConfig<4>>() {
        type F<T> = Fp<MontBackend<T, 4>, 4>;
        let p = num_bigint::BigUint::from(T::MODULUS);
        let r_inv = (num_bigint::BigUint::from(1u8) << 256u32).modpow(&(&p - 2u8), &p);
        let mut rng = test_rng();
        fn pattern<R: ark_std::rand::Rng>(i: usize, rng: &mut R) -> [u64; 4] {
            match i % 4 {
                0 => [u64::MAX; 4],
                1 => [u64::MAX, 0, u64::MAX, 0],
                2 => [0, 0, 0, u64::MAX],
                _ => rng.gen(),
            }
        }
        for len in [0usize, 1, 2, 3, 7, 64, 1000] {
            let mut acc = MontAccumulator::<T, 4>::ZERO;
            let mut sum = num_bigint::BigUint::default();
            for i in 0..len {
                let (a, b) = (pattern(i, &mut rng), pattern(i / 4 + i, &mut rng));
                acc.mul_accumulate(
                    &F::<T>::new_unchecked(BigInt(a)),
                    &F::<T>::new_unchecked(BigInt(b)),
                );
                sum += big::<4>(&a) * big::<4>(&b);
            }
            let mut state = big::<8>(&acc.buf.b1);
            state = (state << 256u32)
                + big::<4>(&acc.buf.b0)
                + (num_bigint::BigUint::from(acc.carry) << 512u32);
            assert_eq!(state, sum, "accumulator state, len {len}");
            let r = acc.reduce();
            assert!(r.0 < T::MODULUS, "reduce is not canonical, len {len}");
            assert_eq!(
                num_bigint::BigUint::from(r.0),
                sum * &r_inv % &p,
                "reduce, len {len}"
            );
        }
    }

    #[test]
    fn accumulator_is_exact() {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            const { assert!(MontAccumulator::<PallasBase, 4>::ASM_FOLD) };
            const { assert!(MontAccumulator::<Secp256k1Base, 4>::ASM_FOLD) };
            const { assert!(!MontAccumulator::<Secp256k1Scalar, 4>::ASM_FOLD) };
        }
        check_exact::<PallasBase>();
        check_exact::<Secp256k1Base>();
        check_exact::<Secp256k1Scalar>();
    }

    #[test]
    fn default_is_the_empty_sum() {
        use crate::AdditiveGroup;
        let zero = MontAccumulator::<Goldilocks, 1>::default();
        assert_eq!(zero.reduce(), Fp::<MontBackend<Goldilocks, 1>, 1>::ZERO);
    }
}
