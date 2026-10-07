//! Variable-time modular inversion by 62-bit divsteps (safegcd), for 4- and 6-limb
//! Montgomery fields.
//!
//! Port of libsecp256k1 `src/modinv64_impl.h` (`secp256k1_modinv64_var`, Copyright (c) 2020
//! Peter Dettman, MIT), generic over [`MontConfig`] instead of its `modinv64_modinfo` and over
//! the radix-`2^62` limb count, and seeded so that Montgomery input yields Montgomery output:
//! `e` starts at `R^2 mod m` rather than `1`, so `X = aR` gives `a^{-1}R` with no conversion
//! and no Montgomery reduction. The invariants across batches are `X*d = R^2*f (mod m)` and
//! `X*e = R^2*g (mod m)`; at termination `g = 0` and `f = s` for `s` in `{-1, 1}`, so
//! `s*d = R^2*X^{-1} = a^{-1}*R (mod m)`.
//!
//! Variable-time in the value being inverted: its trailing-zero counts, divstep branches,
//! batch count and active limb length all depend on it. The BEA it replaces
//! ([`Fp::bea_inverse`](super::Fp::bea_inverse)) is equally variable-time, but branches per
//! bit, so on fresh inputs it runs 2.4x slower than on a repeated one; this path does not.
//!
//! Bernstein, Yang, [Fast constant-time gcd computation and modular inversion](https://eprint.iacr.org/2019/266),
//! Theorem 11.2: `floor((49*b + 57)/17)` divsteps suffice for a `b`-bit modulus, 741 for
//! `b = 256` (12 batches of 62) and 1110 for `b = 384` (18 batches). A 4-limb value takes 5
//! radix-`2^62` limbs and a 6-limb value 7; the kernel bounds are the same for any limb
//! count (every `i128` accumulator stays below `2^126`).
//!
//! # Sources
//!
//! - libsecp256k1 `secp256k1_modinv64_var` and its kernels, MIT:
//!   <https://github.com/bitcoin-core/secp256k1/blob/v0.8.0/src/modinv64_impl.h#L637-L710>,
//!   with the design write-up at
//!   <https://github.com/bitcoin-core/secp256k1/blob/v0.8.0/doc/safegcd_implementation.md>
//! - Zakura's Pasta-specialized port, which the `R^2` seeding and the batch-count vectors come
//!   from:
//!   <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/fields/modinv62.rs>
//! - The constant-time alternative, Pornin, [Optimized Binary GCD for Modular Inversion](https://eprint.iacr.org/2020/972)

use super::MontConfig;
use crate::BigInt;

/// Mask of the low 62 bits of a word.
const MASK62: u64 = (1u64 << 62) - 1;

/// Radix-`2^62` limbs of the widest supported modulus, 6 limbs of 64 bits.
const MAX_L: usize = 7;

/// A signed integer in radix `2^62` over `L` limbs, least-significant limb first. Limbs below
/// the active length are in `[0, 2^62)` and the top active limb carries the sign.
type S62<const L: usize> = [i64; L];

/// An upstream `VERIFY_CHECK`. Live in this crate's unit tests, including `--release` runs
/// where `debug_assert!` is inert, and in debug builds; compiled out otherwise.
macro_rules! verify {
    ($cond:expr $(, $($arg:tt)+)?) => {
        if cfg!(any(test, debug_assertions)) {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// Whether [`invert`] handles `n`-limb moduli.
pub(super) const fn supports(n: usize) -> bool {
    n == 4 || n == 6
}

/// 62-divstep batches for an `n`-limb modulus: `ceil(floor((49 b + 57) / 17) / 62)` for
/// `b = 64 n`.
const fn batches(n: usize) -> u32 {
    let divsteps = (49 * 64 * n + 57) / 17;
    divsteps.div_ceil(62) as u32
}

/// The signed-radix-`2^62` constants of a [`MontConfig`], derived from `MODULUS`, `INV` and
/// `R2`, zero-padded to [`MAX_L`] limbs. Meaningful only when [`supports`] `N`.
pub(super) trait Inv62Params<const N: usize>: MontConfig<N> {
    /// `m` in signed radix `2^62`.
    const MODULUS62: [i64; MAX_L] = pack62(&Self::MODULUS);
    /// `m^{-1} mod 2^62`. Not [`MontConfig::INV`], which is `-m^{-1} mod 2^64`.
    const MU: u64 = Self::INV.wrapping_neg() & MASK62;
    /// `R^2 mod m` in signed radix `2^62`: the initial `e`.
    const R2_62: [i64; MAX_L] = pack62(&Self::R2);
    /// Pins `MU` against the low modulus limb. Forced by [`invert_counted`].
    const CHECK_MU: () = assert!(Self::MU.wrapping_mul(Self::MODULUS62[0] as u64) & MASK62 == 1);
}

impl<T: MontConfig<N>, const N: usize> Inv62Params<N> for T {}

/// Repacks a canonical little-endian value into signed radix `2^62`, zero-padded to
/// [`MAX_L`] limbs. Limb `j` holds bits `[62 j, 62 j + 62)`.
const fn pack62<const N: usize>(x: &BigInt<N>) -> [i64; MAX_L] {
    let mut out = [0i64; MAX_L];
    let mut j = 0;
    while j < MAX_L {
        let word = 62 * j / 64;
        let offset = 62 * j % 64;
        let mut v = 0u64;
        if word < N {
            v = x.0[word] >> offset;
            if offset > 2 && word + 1 < N {
                v |= x.0[word + 1] << (64 - offset);
            }
        }
        out[j] = (v & MASK62) as i64;
        j += 1;
    }
    out
}

/// The low `L` limbs of a [`MAX_L`]-limb constant.
fn low<const L: usize>(x: &[i64; MAX_L]) -> S62<L> {
    core::array::from_fn(|j| x[j])
}

/// Repacks a canonical signed-62 value in `[0, m)`, a [`normalize_62`] output, back into
/// `N` 64-bit little-endian limbs.
fn unpack62<const N: usize, const L: usize>(v: &S62<L>) -> BigInt<N> {
    verify!(
        v.iter().all(|&limb| limb >= 0 && limb as u64 <= MASK62),
        "unpack input must be canonical and reduced"
    );
    let mut out = [0u64; N];
    for (j, &limb) in v.iter().enumerate() {
        let limb = limb as u64;
        let word = 62 * j / 64;
        let offset = 62 * j % 64;
        if word < N {
            out[word] |= limb << offset;
            if offset > 2 && word + 1 < N {
                out[word + 1] |= limb >> (64 - offset);
            }
        }
    }
    BigInt(out)
}

/// The transition matrix for one batch of 62 divsteps. Row sums are bounded by
/// `|u| + |v| <= 2^62` and `|q| + |r| <= 2^62`, and the determinant is exactly `2^62`.
#[derive(Clone, Copy)]
struct Trans2x2 {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

/// Value-level bound checks, a port of upstream's `secp256k1_modinv64_mul_cmp_62`
/// machinery. Active only where [`verify!`] is.
#[cfg(any(test, debug_assertions))]
mod checks {
    use super::{low, Inv62Params, MASK62, S62};
    use core::cmp::Ordering;

    /// `a * factor` over `alen` limbs of `a`, with all but the top limb normalized.
    pub(super) fn mul_62<const L: usize>(a: &S62<L>, alen: usize, factor: i64) -> S62<L> {
        let mut c: i128 = 0;
        let mut r = [0i64; L];
        for i in 0..L - 1 {
            if i < alen {
                c += i128::from(a[i]) * i128::from(factor);
            }
            r[i] = (c as u64 & MASK62) as i64;
            c >>= 62;
        }
        if L - 1 < alen {
            c += i128::from(a[L - 1]) * i128::from(factor);
        }
        assert_eq!(c, i128::from(c as i64), "mul_62 top limb must fit i64");
        r[L - 1] = c as i64;
        r
    }

    /// Compares `a`, over `alen` limbs, against `b * factor`.
    pub(super) fn mul_cmp_62<const L: usize>(
        a: &S62<L>,
        alen: usize,
        b: &S62<L>,
        factor: i64,
    ) -> Ordering {
        let am = mul_62(a, alen, 1);
        let bm = mul_62(b, L, factor);
        for i in 0..L - 1 {
            assert!(
                am[i] >> 62 == 0 && bm[i] >> 62 == 0,
                "operand not normalized"
            );
        }
        for i in (0..L).rev() {
            if am[i] != bm[i] {
                return am[i].cmp(&bm[i]);
            }
        }
        Ordering::Equal
    }

    /// Asserts `-2m < x < m`, the coefficient range invariant.
    pub(super) fn assert_coeff_range<T: Inv62Params<N>, const N: usize, const L: usize>(
        x: &S62<L>,
        what: &str,
    ) {
        let m = low::<L>(&T::MODULUS62);
        assert_eq!(mul_cmp_62(x, L, &m, -2), Ordering::Greater, "{what} > -2m");
        assert_eq!(mul_cmp_62(x, L, &m, 1), Ordering::Less, "{what} < m");
    }

    /// Asserts `-m < f <= m` and `-m < g < m`.
    pub(super) fn assert_fg_range<T: Inv62Params<N>, const N: usize, const L: usize>(
        f: &S62<L>,
        g: &S62<L>,
        len: usize,
    ) {
        let m = low::<L>(&T::MODULUS62);
        assert_eq!(mul_cmp_62(f, len, &m, -1), Ordering::Greater, "f > -m");
        assert_ne!(mul_cmp_62(f, len, &m, 1), Ordering::Greater, "f <= m");
        assert_eq!(mul_cmp_62(g, len, &m, -1), Ordering::Greater, "g > -m");
        assert_eq!(mul_cmp_62(g, len, &m, 1), Ordering::Less, "g < m");
    }

    pub(super) fn assert_non_negative<const L: usize>(x: &S62<L>, what: &str) {
        assert_ne!(mul_cmp_62(x, L, &[0; L], 0), Ordering::Less, "{what} >= 0");
    }
}

/// The negative-eta cancellation multiplier. Out of line so the AArch64 backend does not
/// speculatively evaluate both cancellation formulas before selecting one.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
const fn negative_eta_multiplier(f: u64, g: u64, mask: u64) -> u64 {
    f.wrapping_mul(g)
        .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
        & mask
}

/// The transition matrix for 62 variable-time divsteps, read off the bottom limbs of `f`
/// and `g` alone, with the updated `eta`. Upstream's `secp256k1_modinv64_divsteps_62_var`:
/// trailing-zero counts batch the halvings of `g`, and small modular-inverse formulas
/// cancel up to 4 bits (`eta >= 0`) or 6 bits (`eta < 0`, after the swap) of `g` per
/// iteration.
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64, max_eta: i64) -> (Trans2x2, i64) {
    let mut u: u64 = 1;
    let mut v: u64 = 0;
    let mut q: u64 = 0;
    let mut r: u64 = 1;
    let mut f = f0;
    let mut g = g0;
    let mut i: u32 = 62;

    loop {
        // A sentinel bit counts zeros only up to i; those divsteps just halve g.
        let zeros = (g | (u64::MAX << i)).trailing_zeros();
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= i64::from(zeros);
        i -= zeros;
        if i == 0 {
            break;
        }
        verify!(f & 1 == 1, "f must be odd");
        verify!(g & 1 == 1, "g must be odd once its zeros are shifted out");
        verify!(
            u.wrapping_mul(f0).wrapping_add(v.wrapping_mul(g0)) == f << (62 - i),
            "top row of T must reproduce f"
        );
        verify!(
            q.wrapping_mul(f0).wrapping_add(r.wrapping_mul(g0)) == g << (62 - i),
            "bottom row of T must reproduce g"
        );
        // eta starts at -1 and moves by at most one per divstep.
        verify!((-max_eta..=max_eta).contains(&eta), "eta out of range");
        let w;
        let m;
        if eta < 0 {
            // Negate eta and replace f,g with g,-f.
            eta = -eta;
            let tmp = f;
            f = g;
            g = tmp.wrapping_neg();
            let tmp = u;
            u = q;
            q = tmp.wrapping_neg();
            let tmp = v;
            v = r;
            r = tmp.wrapping_neg();
            // f*(f*f - 2) is the inverse of -f modulo 64: cancels up to 6 bits of g.
            let limit = core::cmp::min(eta as u32 + 1, i);
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 63;
            #[cfg(target_arch = "aarch64")]
            {
                w = negative_eta_multiplier(f, g, m);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                w = f
                    .wrapping_mul(g)
                    .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                    & m;
            }
        } else {
            // f + (((f + 1) & 4) << 1) is the inverse of f modulo 16: cancels up to 4 bits.
            let limit = core::cmp::min(eta as u32 + 1, i);
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 15;
            let w0 = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = w0.wrapping_neg().wrapping_mul(g) & m;
        }
        g = g.wrapping_add(f.wrapping_mul(w));
        q = q.wrapping_add(u.wrapping_mul(w));
        r = r.wrapping_add(v.wrapping_mul(w));
        verify!(g & m == 0, "the masked low bits of g must cancel");
    }
    let t = Trans2x2 {
        u: u as i64,
        v: v as i64,
        q: q as i64,
        r: r as i64,
    };
    // A power-of-two determinant is what makes the matrix-vector products below preserve
    // the relative sizes of f and g.
    verify!(
        i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q) == 1 << 62,
        "determinant of T must be 2^62"
    );
    verify!(
        t.u.unsigned_abs() + t.v.unsigned_abs() <= 1 << 62,
        "|u| + |v| must not exceed 2^62"
    );
    verify!(
        t.q.unsigned_abs() + t.r.unsigned_abs() <= 1 << 62,
        "|q| + |r| must not exceed 2^62"
    );
    (t, eta)
}

/// Applies `t` to `f` and `g` over `len` active limbs, dividing exactly by `2^62`.
/// Upstream's `secp256k1_modinv64_update_fg_62_var`.
fn update_fg_62_var<const L: usize>(len: usize, f: &mut S62<L>, g: &mut S62<L>, t: &Trans2x2) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    verify!((1..=L).contains(&len), "active length out of range");
    let mut cf = i128::from(u) * i128::from(f[0]) + i128::from(v) * i128::from(g[0]);
    let mut cg = i128::from(q) * i128::from(f[0]) + i128::from(r) * i128::from(g[0]);
    // The bottom 62 bits of t*[f,g] are zero by construction of t.
    verify!(cf as u64 & MASK62 == 0, "low bits of new f must vanish");
    verify!(cg as u64 & MASK62 == 0, "low bits of new g must vanish");
    cf >>= 62;
    cg >>= 62;
    for j in 1..len {
        cf += i128::from(u) * i128::from(f[j]) + i128::from(v) * i128::from(g[j]);
        cg += i128::from(q) * i128::from(f[j]) + i128::from(r) * i128::from(g[j]);
        f[j - 1] = (cf as u64 & MASK62) as i64;
        cf >>= 62;
        g[j - 1] = (cg as u64 & MASK62) as i64;
        cg >>= 62;
    }
    verify!(cf == i128::from(cf as i64), "f tail must fit one limb");
    verify!(cg == i128::from(cg as i64), "g tail must fit one limb");
    f[len - 1] = cf as i64;
    g[len - 1] = cg as i64;
}

/// Applies `t` to the coefficients `d`, `e` modulo `m`, dividing exactly by `2^62`.
/// Upstream's `secp256k1_modinv64_update_de_62`. Maintains `-2m < d, e < m`: with the sign
/// corrections `(u & sd) + (v & se)` and the `MU`-derived cancellation term, each row's
/// numerator stays within `(-2^63*m, 2^62*m)`, so the exact shift by 62 lands back in
/// `(-2m, m)`. The bound depends only on the ranges of `d` and `e`, never on their values.
fn update_de_62<T: Inv62Params<N>, const N: usize, const L: usize>(
    d: &mut S62<L>,
    e: &mut S62<L>,
    t: &Trans2x2,
) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N, L>(d, "d");
        checks::assert_coeff_range::<T, N, L>(e, "e");
    }
    // [md, me] start at zero, plus [u, q] if d is negative, plus [v, r] if e is negative.
    let sd = d[L - 1] >> 63;
    let se = e[L - 1] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut me = (q & sd) + (r & se);
    let mut cd = i128::from(u) * i128::from(d[0]) + i128::from(v) * i128::from(e[0]);
    let mut ce = i128::from(q) * i128::from(d[0]) + i128::from(r) * i128::from(e[0]);
    // Correct md, me so that t*[d, e] + m*[md, me] has 62 zero bottom bits.
    md = md.wrapping_sub((T::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    me = me.wrapping_sub((T::MU.wrapping_mul(ce as u64).wrapping_add(me as u64) & MASK62) as i64);
    cd += i128::from(m[0]) * i128::from(md);
    ce += i128::from(m[0]) * i128::from(me);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    verify!(ce as u64 & MASK62 == 0, "low bits of new e must vanish");
    cd >>= 62;
    ce >>= 62;
    for j in 1..L {
        cd += i128::from(u) * i128::from(d[j])
            + i128::from(v) * i128::from(e[j])
            + i128::from(m[j]) * i128::from(md);
        ce += i128::from(q) * i128::from(d[j])
            + i128::from(r) * i128::from(e[j])
            + i128::from(m[j]) * i128::from(me);
        d[j - 1] = (cd as u64 & MASK62) as i64;
        cd >>= 62;
        e[j - 1] = (ce as u64 & MASK62) as i64;
        ce >>= 62;
    }
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    verify!(ce == i128::from(ce as i64), "e tail must fit one limb");
    d[L - 1] = cd as i64;
    e[L - 1] = ce as i64;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N, L>(d, "new d");
        checks::assert_coeff_range::<T, N, L>(e, "new e");
    }
}

/// [`update_de_62`] restricted to the top matrix row: the terminal batch, where only `d`
/// feeds the result and `e` is dead.
fn update_d_only_62<T: Inv62Params<N>, const N: usize, const L: usize>(
    d: &mut S62<L>,
    e: &S62<L>,
    u: i64,
    v: i64,
) {
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N, L>(d, "d");
        checks::assert_coeff_range::<T, N, L>(e, "e");
    }
    let sd = d[L - 1] >> 63;
    let se = e[L - 1] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut cd = i128::from(u) * i128::from(d[0]) + i128::from(v) * i128::from(e[0]);
    md = md.wrapping_sub((T::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    cd += i128::from(m[0]) * i128::from(md);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    cd >>= 62;
    for j in 1..L {
        cd += i128::from(u) * i128::from(d[j])
            + i128::from(v) * i128::from(e[j])
            + i128::from(m[j]) * i128::from(md);
        d[j - 1] = (cd as u64 & MASK62) as i64;
        cd >>= 62;
    }
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    d[L - 1] = cd as i64;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<T, N, L>(d, "new d");
}

/// Brings `r` from `(-2m, m)` to `[0, m)`, negating first if `sign` is negative.
/// Upstream's `secp256k1_modinv64_normalize_62`.
fn normalize_62<T: Inv62Params<N>, const N: usize, const L: usize>(r: &mut S62<L>, sign: i64) {
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<T, N, L>(r, "normalize input");
    // Add the modulus if the input is negative, bringing r to (-m, m), then negate if
    // asked; still (-m, m), and every limb still fits i64.
    let mut cond_add = r[L - 1] >> 63;
    let cond_negate = sign >> 63;
    for j in 0..L {
        r[j] = ((r[j] + (m[j] & cond_add)) ^ cond_negate) - cond_negate;
    }
    for j in 0..L - 1 {
        r[j + 1] += r[j] >> 62;
        r[j] &= MASK62 as i64;
    }
    // Add the modulus again if still negative, bringing r to [0, m).
    cond_add = r[L - 1] >> 63;
    for j in 0..L {
        r[j] += m[j] & cond_add;
    }
    for j in 0..L - 1 {
        r[j + 1] += r[j] >> 62;
        r[j] &= MASK62 as i64;
    }
    #[cfg(any(test, debug_assertions))]
    {
        assert!(
            r.iter().all(|&x| x >> 62 == 0),
            "normalize output must be canonical"
        );
        checks::assert_non_negative(r, "normalize output");
        assert_eq!(
            checks::mul_cmp_62(r, L, &low::<L>(&m), 1),
            core::cmp::Ordering::Less,
            "normalize output must be reduced"
        );
    }
}

/// Whether the `len` active limbs of `g` are all zero.
fn is_zero<const L: usize>(g: &S62<L>, len: usize) -> bool {
    if g[0] != 0 {
        return false;
    }
    g[1..len].iter().fold(0, |acc, &limb| acc | limb) == 0
}

/// Shrinks the active length of `f` and `g` by one when both top limbs are pure sign
/// extensions of the limb below, folding their signs down.
fn shrink_len<const L: usize>(f: &mut S62<L>, g: &mut S62<L>, len: &mut usize) {
    let l = *len;
    let fn_ = f[l - 1];
    let gn = g[l - 1];
    let mut cond = ((l as i64) - 2) >> 63;
    cond |= fn_ ^ (fn_ >> 63);
    cond |= gn ^ (gn >> 63);
    if cond == 0 {
        f[l - 2] = (f[l - 2] as u64 | (fn_ as u64) << 62) as i64;
        g[l - 2] = (g[l - 2] as u64 | (gn as u64) << 62) as i64;
        *len = l - 1;
    }
}

/// Inverts a nonzero canonical Montgomery representation, returning the Montgomery
/// representation of the inverse. `None` for zero. Requires [`supports`] `N`.
pub(super) fn invert<T: Inv62Params<N>, const N: usize>(x: &BigInt<N>) -> Option<BigInt<N>> {
    match N {
        4 => invert_counted::<T, N, 5>(x),
        6 => invert_counted::<T, N, 7>(x),
        _ => unreachable!("modinv62 supports 4- and 6-limb moduli"),
    }
    .map(|(r, _)| r)
}

/// [`invert`] over `L` radix-`2^62` limbs, also reporting how many 62-divstep batches ran.
fn invert_counted<T: Inv62Params<N>, const N: usize, const L: usize>(
    x: &BigInt<N>,
) -> Option<(BigInt<N>, u32)> {
    let () = T::CHECK_MU;
    if x.0.iter().all(|&limb| limb == 0) {
        return None;
    }
    let max_batches = batches(N);
    let max_eta = 62 * i64::from(max_batches) + 1;
    let mut f = low::<L>(&T::MODULUS62);
    let mut g = low::<L>(&pack62(x));
    let mut d: S62<L> = [0; L];
    let mut e = low::<L>(&T::R2_62);
    let mut eta: i64 = -1;
    let mut len: usize = L;

    for batch in 1..=max_batches {
        let (t, new_eta) = divsteps_62_var(eta, f[0] as u64, g[0] as u64, max_eta);
        eta = new_eta;
        // f,g first: a terminating batch then needs only the top row of the d,e update.
        update_fg_62_var(len, &mut f, &mut g, &t);
        if is_zero(&g, len) {
            update_d_only_62::<T, N, L>(&mut d, &e, t.u, t.v);
            #[cfg(any(test, debug_assertions))]
            {
                let mut one = [0i64; L];
                one[0] = 1;
                let cmp = |s| checks::mul_cmp_62(&f, len, &one, s) == core::cmp::Ordering::Equal;
                assert!(cmp(1) || cmp(-1), "f must be +/-1 at termination");
            }
            // The sign of f lives in its top active limb.
            normalize_62::<T, N, L>(&mut d, f[len - 1]);
            return Some((unpack62(&d), batch));
        }
        update_de_62::<T, N, L>(&mut d, &mut e, &t);
        #[cfg(any(test, debug_assertions))]
        checks::assert_fg_range::<T, N, L>(&f, &g, len);
        shrink_len(&mut f, &mut g, &mut len);
    }
    panic!("safegcd inversion exceeded its divstep bound");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fields::models::fp::{Fp, MontBackend},
        AdditiveGroup, BigInteger, Field, One,
    };
    use ark_std::{test_rng, vec, vec::Vec, UniformRand};

    // Zakura's `Fp` and `Fq`: the Pallas and Vesta base fields, the two fields its
    // regression vectors were produced for.
    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337"]
    #[generator = "5"]
    pub struct PallasFq;

    #[derive(MontConfig)]
    #[modulus = "28948022309329048855892746252171976963363056481941647379679742748393362948097"]
    #[generator = "5"]
    pub struct VestaFq;

    // A dense 256-bit modulus: no spare bit, `MODULUS62[4] == 0xff`, none of the sparse
    // limbs the Pasta shape gives the kernels.
    #[derive(MontConfig)]
    #[modulus = "115792089237316195423570985008687907853269984665640564039457584007908834671663"]
    #[generator = "3"]
    pub struct Secp256k1Fq;

    // 6-limb fields: the BLS12-381 and BLS12-377 base fields, and the dense P-384 prime
    // (top bit set, no spare bit).
    #[derive(MontConfig)]
    #[modulus = "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"]
    #[generator = "2"]
    pub struct Bls12381Fq;

    #[derive(MontConfig)]
    #[modulus = "258664426012969094010652733694893533536393512754914660539884262666720468348340822774968888139573360124440321458177"]
    #[generator = "15"]
    pub struct Bls12377Fq;

    #[derive(MontConfig)]
    #[modulus = "39402006196394479212279040100143613805079739270465446667948293404245721771496870329047266088258938001861606973112319"]
    #[generator = "19"]
    pub struct P384Fq;

    type F<C, const N: usize> = Fp<MontBackend<C, N>, N>;

    /// `(internal little-endian representation, 62-divstep batch count)` for the Pallas base
    /// field, from Zakura `fields/modinv62.rs` `PALLAS_VECTORS`, produced there by an
    /// independently validated simulation of this algorithm. A drifting count means the
    /// control flow deviated; investigate rather than re-pin.
    const PALLAS_VECTORS: &[([u64; 4], u32)] = &[
        (
            [
                0x34786d38fffffffd,
                0x992c350be41914ad,
                0xffffffffffffffff,
                0x3fffffffffffffff,
            ],
            7,
        ),
        ([0x00000000000005d1, 0, 0, 0], 8),
        (
            [
                0xe096c0a18679d7ae,
                0x2cc4e34bc6b6f06a,
                0x20eddc12b5a7661d,
                0x0c3c5e66537210ad,
            ],
            8,
        ),
        ([1, 0, 0, 0], 9),
        (
            [
                0xa162a7d34ad63d62,
                0xba71beb748b1fa25,
                0x68dc1330fab3847b,
                0x2dc205e082f2c197,
            ],
            10,
        ),
        (
            [
                0x7ceb4d8e9534361d,
                0x8d7879adc97a8e79,
                0x19ed0538ef7b15eb,
                0x35fb58b0ee06c5b4,
            ],
            10,
        ),
    ];

    /// [`PALLAS_VECTORS`] for the Vesta base field, from Zakura's `VESTA_VECTORS`.
    const VESTA_VECTORS: &[([u64; 4], u32)] = &[
        (
            [
                0x5b2b3e9cfffffffd,
                0x992c350be3420567,
                0xffffffffffffffff,
                0x3fffffffffffffff,
            ],
            7,
        ),
        (
            [
                0x3ea12df55a259593,
                0xd9c58668e724391e,
                0xfcf9d370dd78552a,
                0x1f7fd517b4f7efeb,
            ],
            8,
        ),
        (
            [
                0x4973c2bd762bc27a,
                0x9085d079ecab3a12,
                0x9f66128174a6731a,
                0x3e7853d1fb18fcca,
            ],
            8,
        ),
        ([1, 0, 0, 0], 9),
        ([0x00000ba5f061296c, 0, 0, 0], 10),
        (
            [
                0xc00e79606e554fa8,
                0x13f9947f444f41d3,
                0xd5a780db63e83468,
                0x337e275425a385f3,
            ],
            10,
        ),
    ];

    /// Internal representations exercising the divstep control flow: small values, the
    /// Montgomery constants, values just under the modulus, every power of two, every
    /// trailing-zero count, and saturated limb patterns. Candidates outside `(0, m)` drop out.
    fn edge_reprs<C: MontConfig<N>, const N: usize>() -> Vec<F<C, N>> {
        let one = BigInt::from(1u64);
        let mut raw = vec![one, BigInt::from(2u64), C::R, C::R2];
        for sub in [&one, &C::R, &C::R2] {
            let mut v = C::MODULUS;
            v.sub_with_borrow(sub);
            raw.push(v);
        }
        for k in 0..64 * N {
            let mut limbs = [0u64; N];
            limbs[k / 64] = 1u64 << (k % 64);
            raw.push(BigInt(limbs));
            // The modulus minus one with its low k bits cleared: k trailing zeros.
            let mut v = C::MODULUS;
            v.sub_with_borrow(&one);
            for b in 0..k {
                v.0[b / 64] &= !(1u64 << (b % 64));
            }
            raw.push(v);
        }
        for pat in [0u64, u64::MAX, MASK62] {
            for i in 0..N {
                let mut limbs = [pat; N];
                limbs[i] = !pat;
                raw.push(BigInt(limbs));
            }
        }
        raw.into_iter()
            .filter(|v| v < &C::MODULUS && v != &BigInt::from(0u64))
            .map(F::<C, N>::new_unchecked)
            .collect()
    }

    /// Bit identity with the BEA path `inverse` replaces, plus `a * a^-1 == 1`, over the
    /// edge representations and 20000 random elements.
    fn matches_bea<C: MontConfig<N>, const N: usize, const L: usize>() {
        assert!(F::<C, N>::ZERO.inverse().is_none());
        let mut rng = test_rng();
        let random = (0..20_000).map(|_| F::<C, N>::rand(&mut rng));
        let mut max_batches = 0;
        for a in edge_reprs::<C, N>().into_iter().chain(random) {
            let (inv, batches) = invert_counted::<C, N, L>(&a.0).unwrap();
            let inv = F::<C, N>::new_unchecked(inv);
            assert_eq!(inv, a.inverse().unwrap(), "dispatch did not reach modinv62");
            assert_eq!(inv, a.bea_inverse().unwrap(), "divstep disagrees with BEA");
            assert_eq!(a * inv, F::<C, N>::one(), "a * a^-1 != 1");
            max_batches = max_batches.max(batches);
        }
        assert!(
            max_batches <= super::batches(N),
            "batch bound exceeded: {max_batches}"
        );
    }

    fn check_vectors<C: MontConfig<4>>(vectors: &[([u64; 4], u32)]) {
        for &(limbs, batches) in vectors {
            let a = F::<C, 4>::new_unchecked(BigInt(limbs));
            let (inv, count) = invert_counted::<C, 4, 5>(&a.0).unwrap();
            assert_eq!(count, batches, "batch count drifted for {limbs:x?}");
            assert_eq!(a * F::<C, 4>::new_unchecked(inv), F::<C, 4>::one());
        }
    }

    #[test]
    fn pallas_base() {
        assert_eq!(
            PallasFq::MODULUS62,
            [0x192d30ed00000001, 0x091a63f02533e46e, 2, 0, 0x40, 0, 0]
        );
        assert_eq!(PallasFq::MU, 0x26d2cf1300000001);
        assert_eq!(
            PallasFq::R2_62,
            [
                0x0c78ecb30000000f,
                0x1f4c36f62c37839e,
                0x397a99bc3c95d18d,
                0x1b506bdee72dc51d,
                9,
                0,
                0
            ]
        );
        matches_bea::<PallasFq, 4, 5>();
        check_vectors::<PallasFq>(PALLAS_VECTORS);
    }

    #[test]
    fn vesta_base() {
        assert_eq!(
            VestaFq::MODULUS62,
            [0x0c46eb2100000001, 0x091a63f02652a376, 2, 0, 0x40, 0, 0]
        );
        assert_eq!(VestaFq::MU, 0x33b914df00000001);
        matches_bea::<VestaFq, 4, 5>();
        check_vectors::<VestaFq>(VESTA_VECTORS);
    }

    #[test]
    fn dense_modulus() {
        matches_bea::<Secp256k1Fq, 4, 5>();
    }

    #[test]
    fn six_limbs() {
        assert_eq!((batches(4), batches(6)), (12, 18));
        // The packing round-trips through the 7-limb representation.
        let m = Bls12381Fq::MODULUS;
        let packed: [i64; 7] = low::<7>(&Bls12381Fq::MODULUS62);
        assert_eq!(unpack62::<6, 7>(&packed), m);
        matches_bea::<Bls12381Fq, 6, 7>();
        matches_bea::<Bls12377Fq, 6, 7>();
        matches_bea::<P384Fq, 6, 7>();
    }
}
