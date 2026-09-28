//! Montgomery multiplication and squaring in radix `2^29` for 4-limb fields with
//! `p < 2^255`, the wasm32 path of [`MontConfig`]. wasm has no `u64 x u64 -> u128`
//! product, so the 64-bit CIOS splits every limb product into four `i64.mul`s plus carry
//! detection; here the operands are split into 9 limbs of 29 bits, the 81 limb products
//! (each below `2^58`) are summed into 17 `u64` columns with no carries, and the
//! reduction runs on the columns.
//!
//! Each of 9 reduction steps clears the low column. A standard Montgomery step adds
//! `k p 2^{29 i}` with `k = -t_i p^{-1} mod 2^29`. A Yuval step instead adds
//! `k 2^{29(i + 1)} (2^{-29} mod p)` for `k = t_i mod 2^29`, which needs no product for `k`.
//! Dense moduli (BN254) take 7 Yuval steps; sparse ones (Pasta) take standard steps,
//! where the zero limbs of `p` cost nothing. The last two steps are standard, of 29 and
//! 24 bits, so the result is `a b 2^{-256}`, the same Montgomery form as the 64-bit
//! path, and below `p (p / 2^256 + 1 + 2^{-22}) < 2p`, so one conditional subtraction
//! makes it canonical. Every column stays below `2^63`.
//!
//! After barretenberg's wasm `montgomery_mul`, which uses radix `R = 2^261`:
//! <https://github.com/AztecProtocol/aztec-packages/blob/86dfbab44d953e52a2544f79797c525177e4ed53/barretenberg/cpp/src/barretenberg/ecc/fields/field_impl_generic.hpp#L768-L897>,
//! design notes in `field_docs.md` next to it.


use super::MontConfig;
use crate::BigInt;
use ark_ff_macros::unroll_for_loops;

const MASK29: u64 = (1 << 29) - 1;
const MASK24: u64 = (1 << 24) - 1;

/// The radix-`2^29` constants of a [`MontConfig`], derived from `MODULUS` and `INV`.
/// Meaningful only when [`Self::APPLIES`].
pub(super) trait Mont29Params<const N: usize>: MontConfig<N> {
    /// `N == 4` and `p < 2^255`.
    const APPLIES: bool = N == 4 && Self::MODULUS.0[N - 1] >> 63 == 0;
    /// `p` in 29-bit limbs.
    const P29: [u64; 9] = split29(&low4(&Self::MODULUS));
    /// `2^{-29} mod p` in 29-bit limbs.
    const TWO_INV29: [u64; 9] = split29(&two_pow_minus_29(&low4(&Self::MODULUS), Self::INV));
    /// `-p^{-1} mod 2^29`, the low bits of [`MontConfig::INV`].
    const INV29: u64 = Self::INV & MASK29;
    /// A Yuval step costs 9 products; a standard step one per nonzero limb of `p` and one
    /// for `k` unless `-p^{-1} = -1 mod 2^29`.
    const USE_YUVAL: bool = nonzero(&Self::P29) + (Self::INV29 != MASK29) as usize > 9;
}

impl<T: MontConfig<N>, const N: usize> Mont29Params<N> for T {}

const fn low4<const N: usize>(x: &BigInt<N>) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut i = 0;
    while i < 4 && i < N {
        out[i] = x.0[i];
        i += 1;
    }
    out
}

#[inline(always)]
const fn split29(d: &[u64; 4]) -> [u64; 9] {
    [
        d[0] & MASK29,
        (d[0] >> 29) & MASK29,
        (d[0] >> 58) | ((d[1] << 6) & MASK29),
        (d[1] >> 23) & MASK29,
        (d[1] >> 52) | ((d[2] << 12) & MASK29),
        (d[2] >> 17) & MASK29,
        (d[2] >> 46) | ((d[3] << 18) & MASK29),
        (d[3] >> 11) & MASK29,
        d[3] >> 40,
    ]
}

/// `(p m + 1) / 2^29` with `m = -p^{-1} mod 2^29` (the low bits of `inv`), which is
/// `2^{-29} mod p` and below `p`.
const fn two_pow_minus_29(p: &[u64; 4], inv: u64) -> [u64; 4] {
    let m = (inv & MASK29) as u128;
    let mut x = [0u64; 5];
    let mut carry = 1u128;
    let mut i = 0;
    while i < 4 {
        let v = p[i] as u128 * m + carry;
        x[i] = v as u64;
        carry = v >> 64;
        i += 1;
    }
    x[4] = carry as u64;
    let mut out = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        out[i] = (x[i] >> 29) | (x[i + 1] << 35);
        i += 1;
    }
    out
}

const fn nonzero(x: &[u64; 9]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < 9 {
        n += (x[i] != 0) as usize;
        i += 1;
    }
    n
}

/// Column sums of `l r` by one level of Karatsuba on the 5 + 4 limb halves. The middle
/// column differences are exact in wrapping arithmetic because their true values are
/// nonnegative and below `2^63`.
#[inline(always)]
fn products(l: &[u64; 9], r: &[u64; 9]) -> [u64; 17] {
    let pl0 = l[0] * r[0];
    let pl1 = l[0] * r[1] + l[1] * r[0];
    let pl2 = l[0] * r[2] + l[1] * r[1] + l[2] * r[0];
    let pl3 = l[0] * r[3] + l[1] * r[2] + l[2] * r[1] + l[3] * r[0];
    let pl4 = l[0] * r[4] + l[1] * r[3] + l[2] * r[2] + l[3] * r[1] + l[4] * r[0];
    let pl5 = l[1] * r[4] + l[2] * r[3] + l[3] * r[2] + l[4] * r[1];
    let pl6 = l[2] * r[4] + l[3] * r[3] + l[4] * r[2];
    let pl7 = l[3] * r[4] + l[4] * r[3];
    let pl8 = l[4] * r[4];
    let ph0 = l[5] * r[5];
    let ph1 = l[5] * r[6] + l[6] * r[5];
    let ph2 = l[5] * r[7] + l[6] * r[6] + l[7] * r[5];
    let ph3 = l[5] * r[8] + l[6] * r[7] + l[7] * r[6] + l[8] * r[5];
    let ph4 = l[6] * r[8] + l[7] * r[7] + l[8] * r[6];
    let ph5 = l[7] * r[8] + l[8] * r[7];
    let ph6 = l[8] * r[8];
    let (s0, s1, s2, s3, s4) = (l[0] + l[5], l[1] + l[6], l[2] + l[7], l[3] + l[8], l[4]);
    let (u0, u1, u2, u3, u4) = (r[0] + r[5], r[1] + r[6], r[2] + r[7], r[3] + r[8], r[4]);
    let pc0 = s0 * u0;
    let pc1 = s0 * u1 + s1 * u0;
    let pc2 = s0 * u2 + s1 * u1 + s2 * u0;
    let pc3 = s0 * u3 + s1 * u2 + s2 * u1 + s3 * u0;
    let pc4 = s0 * u4 + s1 * u3 + s2 * u2 + s3 * u1 + s4 * u0;
    let pc5 = s1 * u4 + s2 * u3 + s3 * u2 + s4 * u1;
    let pc6 = s2 * u4 + s3 * u3 + s4 * u2;
    let pc7 = s3 * u4 + s4 * u3;
    let pc8 = s4 * u4;
    let mid = |c: u64, a: u64, b: u64| c.wrapping_sub(a).wrapping_sub(b);
    [
        pl0,
        pl1,
        pl2,
        pl3,
        pl4,
        pl5.wrapping_add(mid(pc0, pl0, ph0)),
        pl6.wrapping_add(mid(pc1, pl1, ph1)),
        pl7.wrapping_add(mid(pc2, pl2, ph2)),
        pl8.wrapping_add(mid(pc3, pl3, ph3)),
        mid(pc4, pl4, ph4),
        mid(pc5, pl5, ph5).wrapping_add(ph0),
        mid(pc6, pl6, ph6).wrapping_add(ph1),
        pc7.wrapping_sub(pl7).wrapping_add(ph2),
        pc8.wrapping_sub(pl8).wrapping_add(ph3),
        ph4,
        ph5,
        ph6,
    ]
}

/// Column sums of `l^2`: 45 products, cross terms doubled.
#[inline(always)]
fn square_products(l: &[u64; 9]) -> [u64; 17] {
    [
        l[0] * l[0],
        (l[0] * l[1]) << 1,
        ((l[0] * l[2]) << 1) + l[1] * l[1],
        (l[0] * l[3] + l[1] * l[2]) << 1,
        ((l[0] * l[4] + l[1] * l[3]) << 1) + l[2] * l[2],
        (l[0] * l[5] + l[1] * l[4] + l[2] * l[3]) << 1,
        ((l[0] * l[6] + l[1] * l[5] + l[2] * l[4]) << 1) + l[3] * l[3],
        (l[0] * l[7] + l[1] * l[6] + l[2] * l[5] + l[3] * l[4]) << 1,
        ((l[0] * l[8] + l[1] * l[7] + l[2] * l[6] + l[3] * l[5]) << 1) + l[4] * l[4],
        (l[1] * l[8] + l[2] * l[7] + l[3] * l[6] + l[4] * l[5]) << 1,
        ((l[2] * l[8] + l[3] * l[7] + l[4] * l[6]) << 1) + l[5] * l[5],
        (l[3] * l[8] + l[4] * l[7] + l[5] * l[6]) << 1,
        ((l[4] * l[8] + l[5] * l[7]) << 1) + l[6] * l[6],
        (l[5] * l[8] + l[6] * l[7]) << 1,
        ((l[6] * l[8]) << 1) + l[7] * l[7],
        (l[7] * l[8]) << 1,
        l[8] * l[8],
    ]
}

/// Montgomery reduction of the columns `t` by `2^256`, then one conditional subtraction.
#[unroll_for_loops(12)]
#[inline(always)]
fn reduce<T: Mont29Params<N>, const N: usize>(t: [u64; 17]) -> [u64; 4] {
    let p = T::P29;
    let mut t = t;
    if T::USE_YUVAL {
        let w = T::TWO_INV29;
        for i in 0..7 {
            let k = t[i] & MASK29;
            t[i + 1] += t[i] >> 29;
            for j in 0..9 {
                t[i + 1 + j] += k * w[j];
            }
        }
        standard_step::<T, N>(&mut t, 7, MASK29);
    } else {
        for i in 0..8 {
            standard_step::<T, N>(&mut t, i, MASK29);
        }
    }
    // A 24-bit step completes the division by `2^{8 * 29 + 24} = 2^256`.
    let k = t[8].wrapping_mul(T::INV29) & MASK24;
    for j in 0..9 {
        t[8 + j] += k * p[j];
    }
    for i in 8..16 {
        t[i + 1] += t[i] >> 29;
        t[i] &= MASK29;
    }
    let mut r = [
        (t[8] >> 24) | (t[9] << 5) | (t[10] << 34) | (t[11] << 63),
        (t[11] >> 1) | (t[12] << 28) | (t[13] << 57),
        (t[13] >> 7) | (t[14] << 22) | (t[15] << 51),
        (t[15] >> 13) | (t[16] << 16),
    ];
    let m = low4(&T::MODULUS);
    let mut borrow = 0u64;
    let mut d = [0u64; 4];
    for i in 0..4 {
        let (v, b1) = r[i].overflowing_sub(m[i]);
        let (v, b2) = v.overflowing_sub(borrow);
        d[i] = v;
        borrow = (b1 | b2) as u64;
    }
    if borrow == 0 {
        r = d;
    }
    r
}

/// Adds `k p 2^{29 i}` with `k = -t_i p^{-1} mod 2^29`, clearing column `i` and carrying
/// it into column `i + 1`.
#[unroll_for_loops(12)]
#[inline(always)]
fn standard_step<T: Mont29Params<N>, const N: usize>(t: &mut [u64; 17], i: usize, mask: u64) {
    let p = T::P29;
    let k = t[i].wrapping_mul(T::INV29) & mask;
    t[i] += k * p[0];
    t[i + 1] += t[i] >> 29;
    for j in 1..9 {
        t[i + j] += k * p[j];
    }
}

/// `a b 2^{-256} mod p` for canonical `a`, `b`.
#[inline(always)]
pub(super) fn mul<T: Mont29Params<N>, const N: usize>(a: &BigInt<N>, b: &BigInt<N>) -> [u64; 4] {
    reduce::<T, N>(products(&split29(&low4(a)), &split29(&low4(b))))
}

/// `a^2 2^{-256} mod p` for canonical `a`.
#[inline(always)]
pub(super) fn square<T: Mont29Params<N>, const N: usize>(a: &BigInt<N>) -> [u64; 4] {
    reduce::<T, N>(square_products(&split29(&low4(a))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fields::models::fp::{Fp256, MontBackend},
        AdditiveGroup, BigInteger, Field, MontConfig, PrimeField, UniformRand,
    };
    use ark_std::{test_rng, vec, vec::Vec};

    #[derive(MontConfig)]
    #[modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583"]
    #[generator = "3"]
    pub struct Bn254FqConfig;

    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337"]
    #[generator = "5"]
    pub struct PallasFqConfig;

    #[derive(MontConfig)]
    #[modulus = "52435875175126190479447740508185965837690552500527637822603658699938581184513"]
    #[generator = "7"]
    pub struct Bls12381FrConfig;

    #[derive(MontConfig)]
    #[modulus = "57896044618658097711785492504343953926634992332820282019728792003956564819949"]
    #[generator = "2"]
    pub struct Curve25519FqConfig;

    fn check<C: MontConfig<4>>(yuval: bool) {
        type F<C> = Fp256<MontBackend<C, 4>>;
        assert!(<C as Mont29Params<4>>::APPLIES);
        assert_eq!(<C as Mont29Params<4>>::USE_YUVAL, yuval);
        // 2^29 * (2^{-29} mod p) = 1 mod p.
        let w = F::<C>::from_bigint(BigInt(two_pow_minus_29(&C::MODULUS.0, C::INV))).unwrap();
        assert_eq!(w * F::<C>::from(1u64 << 29), F::<C>::ONE);
        let mut rng = test_rng();
        let mut edge = vec![F::<C>::ZERO, F::<C>::ONE, -F::<C>::ONE, -F::<C>::from(2u64)];
        edge.push(F::<C>::from_bigint(BigInt([u64::MAX, u64::MAX, 0, 0])).unwrap());
        let mut values: Vec<F<C>> = (0..20_000).map(|_| F::<C>::rand(&mut rng)).collect();
        values.extend(edge.iter().copied());
        for (i, a) in values.iter().enumerate() {
            let b = values[(i * 7919 + 13) % values.len()];
            assert_eq!(mul::<C, 4>(&a.0, &b.0), (*a * b).0 .0);
            assert_eq!(square::<C, 4>(&a.0), a.square().0 .0);
        }
        for a in &edge {
            for b in &edge {
                assert_eq!(mul::<C, 4>(&a.0, &b.0), (*a * b).0 .0);
            }
        }
        // The raw maximum canonical limbs, p - 1, in both operands.
        let mut max = C::MODULUS;
        max.sub_with_borrow(&BigInt::from(1u64));
        let max = F::<C>::from_bigint(max).unwrap();
        assert_eq!(mul::<C, 4>(&max.0, &max.0), (max * max).0 .0);
    }

    #[test]
    fn matches_cios() {
        check::<Bn254FqConfig>(true);
        check::<PallasFqConfig>(false);
        check::<Bls12381FrConfig>(false);
        check::<Curve25519FqConfig>(true);
    }
}
