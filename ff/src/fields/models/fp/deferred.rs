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
//! `square_in_place`.
//!
//! # Sources
//!
//! - Zakura `deferred.rs`, whose `Product::mul_accumulate`, `accumulate` and
//!   `partial_reduce` this generalizes over `N`:
//!   <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/deferred.rs>
//! - Their [blog](https://zakura.com/engineering/deferred-montgomery-products/) on the technique

use super::{Fp, MontBackend, MontConfig};
use crate::{biginteger::arithmetic as fa, const_helpers::MulBuffer, BigInt};
use ark_std::marker::PhantomData;

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
        let mut overflow = 0;
        for i in 0..N {
            let mut carry = 0;
            for j in 0..N {
                *self.buf.get_mut(i + j) =
                    fa::mac_with_carry(self.buf[i + j], a[i], b[j], &mut carry);
            }
            overflow = fa::adc(self.buf.get_mut(i + N), carry, overflow);
        }
        let (carry, wrapped) = self.carry.overflowing_add(overflow);
        debug_assert!(!wrapped, "accumulator carry overflow");
        self.carry = carry;
    }

    /// Reduces the running sum to a canonical field element.
    pub fn reduce(mut self) -> Fp<MontBackend<T, N>, N> {
        // Fold the top limb and the carry back in through their residues, leaving a value
        // below `2^{64(2N-1)} + 2^65 m`, which `T::CAN_DEFER` keeps under `R*m`.
        let hi = self.buf[2 * N - 1];
        let carry = self.carry;
        *self.buf.get_mut(2 * N - 1) = 0;
        self.add_scaled(hi, &T::B_HIGH);
        self.add_scaled(carry, &T::R2);

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
/// and 1.02x at 2, and in wasmtime, where the naive multiply also emulates its 64-bit products,
/// 1.35x faster at 1 term and 1.08x slower at 2.
const MIN_DEFERRED_LEN: usize = if cfg!(target_family = "wasm") { 2 } else { 3 };

/// `sum(a_i * b_i)` with one Montgomery reduction for the whole sum. Falls back to the naive
/// sum when the accumulator's bound does not hold ([`MontConfig::CAN_DEFER`]) or the sum is
/// shorter than [`MIN_DEFERRED_LEN`].
#[inline]
pub(super) fn inner_product<T: MontConfig<N>, const N: usize>(
    a: &[Fp<MontBackend<T, N>, N>],
    b: &[Fp<MontBackend<T, N>, N>],
) -> Fp<MontBackend<T, N>, N> {
    assert_eq!(a.len(), b.len());
    if !T::CAN_DEFER || a.len() < MIN_DEFERRED_LEN {
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
        assert!(Twelve::CAN_DEFER);
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

    #[test]
    fn default_is_the_empty_sum() {
        use crate::AdditiveGroup;
        let zero = MontAccumulator::<Goldilocks, 1>::default();
        assert_eq!(zero.reduce(), Fp::<MontBackend<Goldilocks, 1>, 1>::ZERO);
    }
}
