//! Loose arithmetic for Pasta-shape moduli: every result is an integer below `2p` congruent to the
//! field value, and only [`reduce`] restores the canonical representative below `p`. Multiplication
//! and squaring skip their closing conditional subtraction; addition and subtraction work modulo
//! `2p`. Products of inputs below `2p` stay below `2p` by the closure argument above Zakura Udon
//! [`square_run`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/pasta/montgomery.rs#L262-L272).
//! The portable kernels are Udon's
//! [`montgomery.rs`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/pasta/montgomery.rs)
//! and `square_wide` from
//! [`word.rs`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/pasta/word.rs#L126-L158).
//! The functions take the limbs of an `N`-limb element and apply only when [`applies`] holds, which
//! the callers check first.

use super::super::{Fp, MontBackend, MontConfig};
use crate::BigInt;

type Limbs = [u64; 4];
type F<T, const N: usize> = Fp<MontBackend<T, N>, N>;

/// Whether `T` is a 4-limb modulus of the Pasta shape, `modulus[2] = 0` and `modulus[3] = 2^62`,
/// with `modulus[1] < 2^63`, so that `2 * modulus` has the shape `[t0, t1, 0, 2^63]`.
#[inline(always)]
pub const fn applies<T: MontConfig<N>, const N: usize>() -> bool {
    N == 4 && T::MODULUS.0[1] >> 63 == 0 && T::MODULUS.0[2] == 0 && T::MODULUS.0[3] == 1 << 62
}

/// Whether the loose kernels serve [`applies`] moduli on this target and CPU. On wasm32 the
/// canonical radix-`2^29` products are faster than these 64-bit ones, and on x86-64 with the
/// MULX/ADX kernels in use the x86-64 loose kernels are unmeasured against them, so both keep
/// canonical arithmetic. The x86-64 answer comes from a cached CPUID check, so it is fixed for the
/// process.
#[inline(always)]
pub fn active() -> bool {
    #[cfg(target_family = "wasm")]
    {
        false
    }
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    {
        !super::pasta_asm_available()
    }
    #[cfg(not(any(target_family = "wasm", all(feature = "asm", target_arch = "x86_64"))))]
    {
        true
    }
}

/// The kernel constants of a Pasta-shape modulus, zero for every other modulus.
trait LooseParams<const N: usize>: MontConfig<N> {
    /// `[modulus[0..4], INV]`.
    const PARAMS: [u64; 5] = if applies::<Self, N>() {
        [
            <Self as MontConfig<N>>::MODULUS.0[0],
            <Self as MontConfig<N>>::MODULUS.0[1],
            <Self as MontConfig<N>>::MODULUS.0[2],
            <Self as MontConfig<N>>::MODULUS.0[3],
            Self::INV,
        ]
    } else {
        [0; 5]
    };
    const MODULUS_LIMBS: Limbs = [Self::PARAMS[0], Self::PARAMS[1], Self::PARAMS[2], Self::PARAMS[3]];
    /// `2 * modulus`, of the shape `[t0, t1, 0, 2^63]`.
    const TWICE_MODULUS: Limbs = {
        let m = Self::MODULUS_LIMBS;
        [
            m[0] << 1,
            (m[1] << 1) | (m[0] >> 63),
            (m[2] << 1) | (m[1] >> 63),
            (m[3] << 1) | (m[2] >> 63),
        ]
    };
}

impl<T: MontConfig<N>, const N: usize> LooseParams<N> for T {}

#[inline(always)]
fn limbs<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> Limbs {
    debug_assert!(applies::<T, N>());
    core::array::from_fn(|i| (a.0).0[i])
}

#[inline(always)]
fn element<T: MontConfig<N>, const N: usize>(l: Limbs) -> F<T, N> {
    debug_assert!(lt(&l, &<T as LooseParams<N>>::TWICE_MODULUS));
    F::new_unchecked(BigInt(core::array::from_fn(|i| l[i])))
}

#[inline(always)]
const fn lt(a: &Limbs, b: &Limbs) -> bool {
    let mut i = 4;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

#[inline(always)]
const fn mac(acc: u64, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 * b as u128 + acc as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let t = a as u128 + b as u128 + carry as u128;
    (t as u64, (t >> 64) as u64)
}

#[inline(always)]
const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let t = (a as u128).wrapping_sub(b as u128 + borrow as u128);
    (t as u64, (t >> 127) as u64)
}

/// `a - b` modulo `2^256` and the borrow.
#[inline(always)]
const fn sub_limbs(a: &Limbs, b: &Limbs) -> (Limbs, u64) {
    let (r0, c) = sbb(a[0], b[0], 0);
    let (r1, c) = sbb(a[1], b[1], c);
    let (r2, c) = sbb(a[2], b[2], c);
    let (r3, c) = sbb(a[3], b[3], c);
    ([r0, r1, r2, r3], c)
}

/// `a + b` modulo `2^256` and the carry.
#[inline(always)]
const fn add_limbs(a: &Limbs, b: &Limbs) -> (Limbs, u64) {
    let (r0, c) = adc(a[0], b[0], 0);
    let (r1, c) = adc(a[1], b[1], c);
    let (r2, c) = adc(a[2], b[2], c);
    let (r3, c) = adc(a[3], b[3], c);
    ([r0, r1, r2, r3], c)
}

/// Selects `b` where `mask` is all ones and `a` where it is zero.
#[inline(always)]
fn select(a: &Limbs, b: &Limbs, mask: u64) -> Limbs {
    core::array::from_fn(|i| a[i] ^ ((a[i] ^ b[i]) & mask))
}

#[cfg_attr(
    all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline(always)]
fn reduce_portable(a: &Limbs, params: &[u64; 5]) -> Limbs {
    let (reduced, borrow) = sub_limbs(a, &[params[0], params[1], params[2], params[3]]);
    select(a, &reduced, (borrow ^ 1).wrapping_neg())
}

#[cfg_attr(all(feature = "asm", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn add_portable(a: &Limbs, b: &Limbs, twice: &Limbs) -> Limbs {
    let (sum, carry) = add_limbs(a, b);
    let (reduced, borrow) = sub_limbs(&sum, twice);
    select(&sum, &reduced, (carry | (borrow ^ 1)).wrapping_neg())
}

#[cfg_attr(
    all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")),
    allow(dead_code)
)]
#[inline(always)]
fn sub_portable(a: &Limbs, b: &Limbs, twice: &Limbs) -> Limbs {
    let (diff, borrow) = sub_limbs(a, b);
    let mask = borrow.wrapping_neg();
    add_limbs(&diff, &core::array::from_fn(|i| twice[i] & mask)).0
}

/// Udon `montgomery_multiply_loose_rust`.
#[cfg_attr(all(feature = "asm", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn mul_portable(a: &Limbs, b: &Limbs, params: &[u64; 5]) -> Limbs {
    let [a0, a1, a2, a3] = *a;
    let (mut r0, mut r1, mut r2, mut r3) = (0u64, 0u64, 0u64, 0u64);
    for &bj in b {
        let (s0, c) = mac(r0, a0, bj, 0);
        let (s1, c) = mac(r1, a1, bj, c);
        let (s2, c) = mac(r2, a2, bj, c);
        let (s3, s4) = mac(r3, a3, bj, c);
        let q = s0.wrapping_mul(params[4]);
        let (_, c) = mac(s0, q, params[0], 0);
        let (t1, c) = mac(s1, q, params[1], c);
        let (t2, c) = adc(s2, 0, c);
        let (t3, c) = adc(s3, q << 62, c);
        let (t4, c) = adc(s4, q >> 2, c);
        debug_assert!(c == 0);
        (r0, r1, r2, r3) = (t1, t2, t3, t4);
    }
    [r0, r1, r2, r3]
}

/// Udon `square_wide` followed by `montgomery_reduce_unreduced`.
#[cfg_attr(all(feature = "asm", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn square_portable(a: &Limbs, params: &[u64; 5]) -> Limbs {
    let (r1, c) = mac(0, a[0], a[1], 0);
    let (r2, c) = mac(0, a[0], a[2], c);
    let (r3, r4) = mac(0, a[0], a[3], c);
    let (r3, c) = mac(r3, a[1], a[2], 0);
    let (r4, r5) = mac(r4, a[1], a[3], c);
    let (r5, r6) = mac(r5, a[2], a[3], 0);
    let r7 = r6 >> 63;
    let r6 = (r6 << 1) | (r5 >> 63);
    let r5 = (r5 << 1) | (r4 >> 63);
    let r4 = (r4 << 1) | (r3 >> 63);
    let r3 = (r3 << 1) | (r2 >> 63);
    let r2 = (r2 << 1) | (r1 >> 63);
    let r1 = r1 << 1;
    let (r0, c) = mac(0, a[0], a[0], 0);
    let (r1, c) = adc(r1, 0, c);
    let (r2, c) = mac(r2, a[1], a[1], c);
    let (r3, c) = adc(r3, 0, c);
    let (r4, c) = mac(r4, a[2], a[2], c);
    let (r5, c) = adc(r5, 0, c);
    let (r6, c) = mac(r6, a[3], a[3], c);
    let (r7, _) = adc(r7, 0, c);

    let (mut l0, mut l1, mut l2, mut l3) = (r0, r1, r2, r3);
    for _ in 0..4 {
        let k = l0.wrapping_mul(params[4]);
        let (_, c) = mac(l0, k, params[0], 0);
        let (s0, c) = mac(l1, k, params[1], c);
        let (s1, c) = adc(l2, 0, c);
        let (s2, c) = adc(l3, k << 62, c);
        let s3 = (k >> 2) + c;
        (l0, l1, l2, l3) = (s0, s1, s2, s3);
    }
    let (l0, c) = adc(l0, r4, 0);
    let (l1, c) = adc(l1, r5, c);
    let (l2, c) = adc(l2, r6, c);
    let (l3, c) = adc(l3, r7, c);
    debug_assert!(c == 0);
    [l0, l1, l2, l3]
}

/// `(a * b + c * d) * R^{-1}` below `2p` by two products and one addition.
#[cfg_attr(all(feature = "asm", target_arch = "aarch64"), allow(dead_code))]
#[inline(always)]
fn sum_of_two_products_portable(
    a: &Limbs,
    b: &Limbs,
    c: &Limbs,
    d: &Limbs,
    params: &[u64; 5],
    twice: &Limbs,
) -> Limbs {
    add_portable(
        &mul_portable(a, b, params),
        &mul_portable(c, d, params),
        twice,
    )
}

/// The assembly kernel where one is compiled in, the portable function otherwise.
macro_rules! kernel {
    ($asm:ident ($($arg:expr),*), $portable:ident ($($parg:expr),*)) => {{
        #[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            super::kernel::$asm($($arg),*)
        }
        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        {
            $portable($($parg),*)
        }
    }};
}

/// [`kernel`] for the kernels that use MULX and ADCX/ADOX on x86-64, which take the portable
/// function on CPUs without BMI2 and ADX.
macro_rules! mulx_kernel {
    ($asm:ident ($($arg:expr),*), $portable:ident ($($parg:expr),*)) => {{
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            super::kernel::$asm($($arg),*)
        }
        #[cfg(all(feature = "asm", target_arch = "x86_64"))]
        {
            if super::pasta_asm_available() {
                super::kernel::$asm($($arg),*)
            } else {
                $portable($($parg),*)
            }
        }
        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        {
            $portable($($parg),*)
        }
    }};
}

/// `a + b` below `2p`.
#[inline(always)]
pub fn add<T: MontConfig<N>, const N: usize>(a: &F<T, N>, b: &F<T, N>) -> F<T, N> {
    let (a, b, t) = (limbs(a), limbs(b), &<T as LooseParams<N>>::TWICE_MODULUS);
    element(kernel!(add_loose(&a, &b, t), add_portable(&a, &b, t)))
}

/// `a - b` below `2p`.
#[inline(always)]
pub fn sub<T: MontConfig<N>, const N: usize>(a: &F<T, N>, b: &F<T, N>) -> F<T, N> {
    let (a, b, t) = (limbs(a), limbs(b), &<T as LooseParams<N>>::TWICE_MODULUS);
    element(kernel!(sub_loose(&a, &b, t), sub_portable(&a, &b, t)))
}

/// `2a` below `2p`.
#[inline(always)]
pub fn double<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> F<T, N> {
    add(a, a)
}

/// `-a` below `2p`.
#[inline(always)]
pub fn neg<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> F<T, N> {
    let (a, t) = (limbs(a), &<T as LooseParams<N>>::TWICE_MODULUS);
    element(kernel!(neg_loose(&a, t), sub_portable(&[0; 4], &a, t)))
}

/// `a * b * R^{-1}` below `2p`.
#[inline(always)]
pub fn mul<T: MontConfig<N>, const N: usize>(a: &F<T, N>, b: &F<T, N>) -> F<T, N> {
    let (a, b, p) = (limbs(a), limbs(b), &<T as LooseParams<N>>::PARAMS);
    element(mulx_kernel!(mul_loose(&a, &b, p), mul_portable(&a, &b, p)))
}

/// `a^2 * R^{-1}` below `2p`.
#[inline(always)]
pub fn square<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> F<T, N> {
    let (a, p) = (limbs(a), &<T as LooseParams<N>>::PARAMS);
    element(mulx_kernel!(square_loose(&a, p), square_portable(&a, p)))
}

/// `a[0] * b[0] + a[1] * b[1]` below `2p` for `a` below `p` and `b` below `2p`, by the fused
/// kernel where it exists.
#[inline(always)]
pub fn sum_of_two_products<T: MontConfig<N>, const N: usize>(
    a: &[F<T, N>; 2],
    b: &[F<T, N>; 2],
) -> F<T, N> {
    let (a0, a1, b0, b1) = (limbs(&a[0]), limbs(&a[1]), limbs(&b[0]), limbs(&b[1]));
    let p = &<T as LooseParams<N>>::PARAMS;
    debug_assert!(lt(&a0, &<T as LooseParams<N>>::MODULUS_LIMBS));
    debug_assert!(lt(&a1, &<T as LooseParams<N>>::MODULUS_LIMBS));
    element(mulx_kernel!(
        sum_of_two_products(&a0, &b0, &a1, &b1, p),
        sum_of_two_products_portable(&a0, &b0, &a1, &b1, p, &<T as LooseParams<N>>::TWICE_MODULUS)
    ))
}

/// The canonical representative of a value below `2p`.
#[inline(always)]
pub fn reduce<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> F<T, N> {
    let (a, p) = (limbs(a), &<T as LooseParams<N>>::PARAMS);
    let r = kernel!(reduce_loose(&a, p), reduce_portable(&a, p));
    F::new_unchecked(BigInt(core::array::from_fn(|i| r[i])))
}

/// Whether a value below `2p` is zero, that is `0` or `p`.
#[inline(always)]
pub fn is_zero<T: MontConfig<N>, const N: usize>(a: &F<T, N>) -> bool {
    let a = limbs(a);
    a == [0; 4] || a == <T as LooseParams<N>>::MODULUS_LIMBS
}

#[cfg(test)]
mod tests {
    use super::{super::tests::*, *};
    use crate::{BigInteger, UniformRand};
    use ark_std::{test_rng, vec::Vec};
    use num_bigint::BigUint;

    fn big(l: &Limbs) -> BigUint {
        l.iter().rev().fold(BigUint::from(0u8), |acc, &x| (acc << 64) | BigUint::from(x))
    }

    fn loose_values<T: MontConfig<4>>() -> Vec<F<T, 4>> {
        let p = T::MODULUS;
        let twice = <T as LooseParams<4>>::TWICE_MODULUS;
        let mut out = Vec::new();
        let mut rng = test_rng();
        for v in edge_values::<T>() {
            out.push(v);
            let (s, c) = add_limbs(&(v.0).0, &p.0);
            assert_eq!(c, 0);
            out.push(F::new_unchecked(BigInt(s)));
        }
        for k in 1..=4u64 {
            let (v, _) = sub_limbs(&twice, &[k, 0, 0, 0]);
            out.push(F::new_unchecked(BigInt(v)));
        }
        out.push(F::new_unchecked(p));
        for _ in 0..48 {
            let v = F::<T, 4>::rand(&mut rng);
            out.push(v);
            out.push(F::new_unchecked(BigInt(add_limbs(&(v.0).0, &p.0).0)));
        }
        let mut half = p;
        half.div2();
        out.push(F::new_unchecked(BigInt(add_limbs(&half.0, &p.0).0)));
        out
    }

    fn check<T: MontConfig<4>>() {
        assert!(applies::<T, 4>());
        let p = BigUint::from(T::MODULUS);
        let twice = &p * 2u8;
        let r_inv = (BigUint::from(1u8) << 256u32).modpow(&(&p - 2u8), &p);
        let values = loose_values::<T>();
        let loose = |x: &F<T, 4>, what: &str| {
            assert!(big(&(x.0).0) < twice, "{what} not below 2p");
            big(&(x.0).0)
        };
        for a in &values {
            let ab = big(&(a.0).0);
            assert_eq!(loose(&double(a), "double"), (&ab << 1) % &twice);
            assert_eq!(loose(&neg(a), "neg") % &p, (&twice - &ab) % &p);
            assert_eq!(loose(&square(a), "square") % &p, &ab * &ab * &r_inv % &p);
            assert_eq!(square_portable(&(a.0).0, &<T as LooseParams<4>>::PARAMS), (square(a).0).0);
            let r = reduce(a);
            assert!(r.0 < T::MODULUS);
            assert_eq!(big(&(r.0).0), &ab % &p);
            assert_eq!(is_zero(a), (&ab % &p) == BigUint::from(0u8));
            for b in &values {
                let bb = big(&(b.0).0);
                assert_eq!(loose(&add(a, b), "add"), (&ab + &bb) % &twice);
                assert_eq!(loose(&sub(a, b), "sub"), (&ab + &twice - &bb) % &twice);
                let m = mul(a, b);
                assert_eq!(loose(&m, "mul") % &p, &ab * &bb * &r_inv % &p);
                let params = &<T as LooseParams<4>>::PARAMS;
                assert_eq!(mul_portable(&(a.0).0, &(b.0).0, params), (m.0).0);
                let twice_l = &<T as LooseParams<4>>::TWICE_MODULUS;
                assert_eq!(add_portable(&(a.0).0, &(b.0).0, twice_l), (add(a, b).0).0);
                assert_eq!(sub_portable(&(a.0).0, &(b.0).0, twice_l), (sub(a, b).0).0);
            }
        }
    }

    fn check_sum_of_two_products<T: MontConfig<4>>() {
        let p = BigUint::from(T::MODULUS);
        let twice = &p * 2u8;
        let r_inv = (BigUint::from(1u8) << 256u32).modpow(&(&p - 2u8), &p);
        let values = loose_values::<T>();
        let canonical: Vec<_> = values.iter().filter(|v| v.0 < T::MODULUS).copied().collect();
        let (n, m) = (values.len(), canonical.len());
        for (i, a0) in canonical.iter().enumerate() {
            for (j, b0) in values.iter().enumerate() {
                let a1 = canonical[(i * 5 + j + 1) % m];
                for b1 in [values[(i + 3 * j + 7) % n], *b0, values[n - 1 - j % n]] {
                    let r = sum_of_two_products(&[*a0, a1], &[*b0, b1]);
                    let rb = big(&(r.0).0);
                    assert!(rb < twice);
                    let expected = (big(&(a0.0).0) * big(&(b0.0).0) + big(&(a1.0).0) * big(&(b1.0).0))
                        * &r_inv
                        % &p;
                    assert_eq!(rb % &p, expected);
                }
            }
        }
    }

    #[test]
    fn loose_sum_of_two_products_matches_biguint() {
        check_sum_of_two_products::<PallasBase>();
        check_sum_of_two_products::<VestaBase>();
    }

    #[test]
    fn loose_kernels_match_biguint() {
        check::<PallasBase>();
        check::<VestaBase>();
    }

    /// Long chains of loose products and squares started near `2p` stay below `2p`.
    #[test]
    fn loose_chains_stay_below_twice_modulus() {
        fn run<T: MontConfig<4>>() {
            let p = T::MODULUS;
            let twice = <T as LooseParams<4>>::TWICE_MODULUS;
            let mut rng = test_rng();
            for _ in 0..2_000 {
                let v = F::<T, 4>::rand(&mut rng);
                let top = F::<T, 4>::new_unchecked(BigInt(add_limbs(&(v.0).0, &p.0).0));
                let mut x = top;
                for i in 0..64 {
                    x = if i % 3 == 0 { square(&x) } else { mul(&x, &top) };
                    assert!(lt(&(x.0).0, &twice));
                }
                let canonical = (0..64).fold(v, |acc, i| if i % 3 == 0 { acc * acc } else { acc * v });
                assert_eq!(reduce(&x), canonical);
            }
        }
        run::<PallasBase>();
        run::<VestaBase>();
    }

    #[test]
    fn twice_modulus_shape() {
        for t in [
            <PallasBase as LooseParams<4>>::TWICE_MODULUS,
            <VestaBase as LooseParams<4>>::TWICE_MODULUS,
        ] {
            assert_eq!((t[2], t[3]), (0, 1 << 63));
        }
    }

    #[test]
    fn twice_modulus() {
        let p = BigUint::from(PallasBase::MODULUS);
        assert_eq!(big(&<PallasBase as LooseParams<4>>::TWICE_MODULUS), &p * 2u8);
        let p = BigUint::from(VestaBase::MODULUS);
        assert_eq!(big(&<VestaBase as LooseParams<4>>::TWICE_MODULUS), &p * 2u8);
        assert!(!applies::<Bn254Base, 4>());
    }

    #[derive(crate::MontConfig)]
    #[modulus = "21888242871839275222246405745257275088696311157297823662689037894645226208583"]
    #[generator = "3"]
    pub struct Bn254Base;
}
