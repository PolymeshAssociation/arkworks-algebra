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
//! `square_in_place`. Under the `asm` feature the 4-limb accumulate and fold run in inline
//! assembly, `deferred/aarch64.rs` on AArch64 and `deferred/x86_64.rs` on x86-64 CPUs with BMI2
//! and ADX. On an Apple M3 Max the AArch64 kernels take a Pallas inner product from 6.3 to 4.5 ns
//! per term.
//!
//! # Sources
//!
//! - Zakura `deferred.rs`, whose `Product::mul_accumulate`, `accumulate` and
//!   `partial_reduce` this generalizes over `N`:
//!   <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/deferred.rs>
//! - Zakura `deferred/aarch64.rs` for the AArch64 kernels, cited in `deferred/aarch64.rs`
//! - Zakura Udon `x86_64_asm/products.rs` for the x86-64 kernels, cited in `deferred/x86_64.rs`
//! - Their [blog](https://zakura.com/engineering/deferred-montgomery-products/) on the technique

use super::{mont29::Mont29Params, Fp, MontBackend, MontConfig};
use crate::{biginteger::arithmetic as fa, const_helpers::MulBuffer, BigInt};
use ark_std::marker::PhantomData;

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

/// Whether the 4-limb kernels are usable on this CPU, which on x86-64 needs BMI2 and ADX.
#[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn kernel_available() -> bool {
    cfg!(target_arch = "aarch64") || super::asm_backend_active()
}

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
        #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
        if N == 4 && kernel_available() {
            self.mul_accumulate_kernel(a, b);
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

    /// Adds `sum(a_i * b_i)` over a block to the running sum, without reducing. Comba order:
    /// output limb `k` gathers every `a_i[l] * b_i[k - l]` of the block into four independent
    /// three-word lanes, merged once per limb, so the carry into limb `k + 1` is resolved once per
    /// block instead of once per product. Three words hold any realizable column, since fewer than
    /// `2^61` products of `2^128` fit in memory. Generalizes Zakura's 4-limb
    /// [`mul_accumulate_block`](https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/deferred.rs#L188-L286) (#280) over `N`.
    #[inline]
    pub fn mul_accumulate_block(
        &mut self,
        a: &[Fp<MontBackend<T, N>, N>],
        b: &[Fp<MontBackend<T, N>, N>],
    ) {
        assert_eq!(a.len(), b.len());
        let (a_quads, a_rest) = a.split_at(a.len() - a.len() % 4);
        let (b_quads, b_rest) = b.split_at(a_quads.len());
        let (mut carry_low, mut carry_high) = (0u64, 0u64);
        for k in 0..(2 * N - 1) {
            let mut lanes = [
                start_column(self.buf[k], carry_low, carry_high),
                [0; 3],
                [0; 3],
                [0; 3],
            ];
            for l in k.saturating_sub(N - 1)..=k.min(N - 1) {
                let m = k - l;
                for (a, b) in a_quads.chunks_exact(4).zip(b_quads.chunks_exact(4)) {
                    for lane in 0..4 {
                        add_product(&mut lanes[lane], ((a[lane].0).0)[l], ((b[lane].0).0)[m]);
                    }
                }
                for (a, b) in a_rest.iter().zip(b_rest) {
                    add_product(&mut lanes[0], (a.0).0[l], (b.0).0[m]);
                }
            }
            let column = merge_columns(lanes);
            *self.buf.get_mut(k) = column[0];
            (carry_low, carry_high) = (column[1], column[2]);
        }
        let column = start_column(self.buf[2 * N - 1], carry_low, carry_high);
        *self.buf.get_mut(2 * N - 1) = column[0];
        debug_assert_eq!(column[2], 0, "accumulator block overflow");
        self.add_carry(column[1]);
    }

    #[inline(always)]
    fn add_carry(&mut self, overflow: u64) {
        let (carry, wrapped) = self.carry.overflowing_add(overflow);
        debug_assert!(!wrapped, "accumulator carry overflow");
        self.carry = carry;
    }

    /// [`Self::mul_accumulate`] by the 4-limb kernel. Requires `N = 4` and [`kernel_available`].
    #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[inline(always)]
    fn mul_accumulate_kernel(&mut self, a: &[u64; N], b: &[u64; N]) {
        let (limbs, overflow) = kernel::mul_accumulate(
            self.limbs8(),
            a[..4].try_into().unwrap(),
            b[..4].try_into().unwrap(),
        );
        self.set_limbs8(limbs);
        self.add_carry(overflow);
    }

    /// The `2N = 8` limbs, for the 4-limb kernels.
    #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[inline(always)]
    fn limbs8(&self) -> [u64; 8] {
        core::array::from_fn(|i| self.buf[i])
    }

    #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
    #[inline(always)]
    fn set_limbs8(&mut self, limbs: [u64; 8]) {
        for (i, limb) in limbs.into_iter().enumerate() {
            *self.buf.get_mut(i) = limb;
        }
    }

    /// Whether the 4-limb fold kernel's bounds hold, `B_HIGH < 2^253` and `R2 < 2^252`.
    #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
    const ASM_FOLD: bool = N == 4 && T::B_HIGH.0[N - 1] < 1 << 61 && T::R2.0[N - 1] < 1 << 60;

    /// Reduces the running sum to a canonical field element.
    pub fn reduce(mut self) -> Fp<MontBackend<T, N>, N> {
        // Fold the top limb and the carry back in through their residues, leaving a value
        // below `2^{64(2N-1)} + 2^65 m`, which `T::CAN_DEFER` keeps under `R*m`.
        #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
        let folded = Self::ASM_FOLD && kernel_available() && {
            let limbs = kernel::partial_reduce(
                self.limbs8(),
                self.carry,
                T::B_HIGH.0[..4].try_into().unwrap(),
                T::R2.0[..4].try_into().unwrap(),
            );
            self.set_limbs8(limbs);
            true
        };
        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
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

/// `column += lhs * rhs` on a three-word column. This and the two helpers below are Zakura's
/// [`add_product`, `start_column` and `merge_columns`](https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/deferred.rs#L375-L419).
#[inline(always)]
fn add_product(column: &mut [u64; 3], lhs: u64, rhs: u64) {
    let product = (lhs as u128) * (rhs as u128);
    let (low, low_carry) = column[0].overflowing_add(product as u64);
    let (middle, high_carry) = column[1].overflowing_add((product >> 64) as u64);
    let (middle, middle_carry) = middle.overflowing_add(low_carry as u64);
    let high = column[2] + high_carry as u64 + middle_carry as u64;
    *column = [low, middle, high];
}

/// A column seeded with an accumulator limb and the two-word carry out of the previous column.
#[inline(always)]
const fn start_column(limb: u64, carry_low: u64, carry_high: u64) -> [u64; 3] {
    let (low, carry) = limb.overflowing_add(carry_low);
    let (middle, high) = carry_high.overflowing_add(carry as u64);
    [low, middle, high as u64]
}

#[inline(always)]
fn merge_columns(lanes: [[u64; 3]; 4]) -> [u64; 3] {
    let [mut result, lane1, lane2, lane3] = lanes;
    for lane in [lane1, lane2, lane3] {
        let mut carry = 0;
        for w in 0..3 {
            carry = fa::adc(&mut result[w], lane[w], carry);
        }
        debug_assert_eq!(carry, 0);
    }
    result
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
    let mut blocked = 0;
    if uses_blocks::<N>() {
        let tail = a.len() % BLOCK_LEN;
        blocked = if tail >= MIN_BLOCK_LEN {
            a.len()
        } else {
            a.len() - tail
        };
        for (a, b) in a[..blocked]
            .chunks(BLOCK_LEN)
            .zip(b[..blocked].chunks(BLOCK_LEN))
        {
            acc.mul_accumulate_block(a, b);
        }
    }
    let (a, b) = (&a[blocked..], &b[blocked..]);
    // One CPU check for the whole sum instead of one per term.
    #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
    if N == 4 && kernel_available() {
        for (a, b) in a.iter().zip(b) {
            acc.mul_accumulate_kernel(&(a.0).0, &(b.0).0);
        }
        return acc.reduce();
    }
    for (a, b) in a.iter().zip(b) {
        acc.mul_accumulate(a, b);
    }
    acc.reduce()
}

/// Terms per [`MontAccumulator::mul_accumulate_block`] call in [`inner_product`]. On an Apple M3
/// Max a Pallas inner product costs 6.4 ns per term per product and 5.0 in blocks of 128 (blocks
/// of 32 give 1.08x, blocks past 128 nothing more), and BLS12-381 `Fq` 13.1 and 11.2.
const BLOCK_LEN: usize = 128;

/// Shortest tail given a block of its own; below it the per-product kernel wins (0.94x blocked at
/// 20 terms, 1.07x at 32).
const MIN_BLOCK_LEN: usize = 32;

/// Whether [`inner_product`] runs in blocks: on AArch64, except for 4 limbs under the `asm`
/// feature, whose per-product kernel is 1.1x faster than the blocks, as Zakura #522 found. In
/// wasmtime the blocks are 0.6x, their `u128` columns being emulated; x86-64 is unmeasured.
const fn uses_blocks<const N: usize>() -> bool {
    cfg!(target_arch = "aarch64") && !(cfg!(feature = "asm") && N == 4)
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

    /// The blocked path leaves exactly the per-product path's state, on saturating limbs.
    fn check_block<T: MontConfig<4>>() {
        type F<T> = Fp<MontBackend<T, 4>, 4>;
        let mut rng = test_rng();
        fn pick<T: MontConfig<4>, R: ark_std::rand::Rng>(i: usize, rng: &mut R) -> F<T> {
            let limbs = match i % 3 {
                0 => [u64::MAX; 4],
                1 => [u64::MAX, 0, u64::MAX, 0],
                _ => rng.gen(),
            };
            F::<T>::new_unchecked(BigInt(limbs))
        }
        for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 100] {
            let a: Vec<F<T>> = (0..len).map(|i| pick::<T, _>(i, &mut rng)).collect();
            let b: Vec<F<T>> = (0..len).map(|i| pick::<T, _>(i + 1, &mut rng)).collect();
            let (mut per_product, mut blocked) =
                (MontAccumulator::<T, 4>::ZERO, MontAccumulator::<T, 4>::ZERO);
            for (x, y) in a.iter().zip(&b) {
                per_product.mul_accumulate(x, y);
            }
            // Seed both with the same nonzero state so the block's carry-in path is exercised.
            let seed: F<T> = pick::<T, _>(0, &mut rng);
            per_product.mul_accumulate(&seed, &seed);
            blocked.mul_accumulate(&seed, &seed);
            for (x, y) in a.chunks(32).zip(b.chunks(32)) {
                blocked.mul_accumulate_block(x, y);
            }
            assert_eq!(
                (per_product.buf.b0, per_product.buf.b1, per_product.carry),
                (blocked.buf.b0, blocked.buf.b1, blocked.carry),
                "len {len}"
            );
            assert_eq!(per_product.reduce(), blocked.reduce(), "len {len}");
        }
    }

    /// `inner_product` against the naive sum at lengths around the block and tail boundaries.
    #[test]
    fn inner_product_across_blocks() {
        fn run<T: MontConfig<4>>() {
            let mut rng = test_rng();
            for len in [31usize, 32, 33, 127, 128, 129, 159, 160, 161, 300] {
                let a: Vec<_> = (0..len)
                    .map(|_| Fp::<MontBackend<T, 4>, 4>::rand(&mut rng))
                    .collect();
                let b: Vec<_> = (0..len)
                    .map(|_| Fp::<MontBackend<T, 4>, 4>::rand(&mut rng))
                    .collect();
                let naive = a.iter().zip(&b).map(|(a, b)| *a * b).sum();
                assert_eq!(inner_product::<T, 4>(&a, &b), naive, "len {len}");
            }
        }
        run::<PallasBase>();
        run::<Secp256k1Scalar>();
    }

    #[test]
    fn block_matches_per_product() {
        check_block::<PallasBase>();
        check_block::<Secp256k1Scalar>();
    }

    #[test]
    fn accumulator_is_exact() {
        #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
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
