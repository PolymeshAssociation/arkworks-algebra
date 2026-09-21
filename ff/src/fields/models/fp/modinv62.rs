//! Variable-time modular inversion by 62-bit divsteps (safegcd), for 4-limb Montgomery
//! fields.
//!
//! Port of libsecp256k1 `src/modinv64_impl.h` (`secp256k1_modinv64_var`, Copyright (c) 2020
//! Peter Dettman, MIT), generic over [`MontConfig`] instead of its `modinv64_modinfo`, and
//! seeded so that Montgomery input yields Montgomery output: `e` starts at `R^2 mod m`
//! rather than `1`, so `X = aR` gives `a^{-1}R` with no conversion and no Montgomery
//! reduction. The invariants across batches are `X*d = R^2*f (mod m)` and
//! `X*e = R^2*g (mod m)`; at termination `g = 0` and `f = s` for `s` in `{-1, 1}`, so
//! `s*d = R^2*X^{-1} = a^{-1}*R (mod m)`.
//!
//! Variable-time in the value being inverted: its trailing-zero counts, divstep branches,
//! batch count and active limb length all depend on it. The BEA it replaces
//! ([`Fp::bea_inverse`](super::Fp::bea_inverse)) is equally variable-time.
//!
//! Bernstein, Yang, [Fast constant-time gcd computation and modular inversion](https://eprint.iacr.org/2019/266),
//! Theorem 11.2: `floor((49*b + 57)/17)` divsteps suffice for a `b`-bit modulus, 741 for
//! `b = 256`, so 12 batches of 62 cover every 4-limb field.
//!
//! # Sources
//!
//! - libsecp256k1 `secp256k1_modinv64_var` and its kernels, MIT:
//!   <https://github.com/bitcoin-core/secp256k1/blob/master/src/modinv64_impl.h>, with the
//!   design write-up at
//!   <https://github.com/bitcoin-core/secp256k1/blob/master/doc/safegcd_implementation.md>
//! - Zakura's Pasta-specialized port, which the `R^2` seeding and the batch-count vectors come
//!   from:
//!   <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/fields/modinv62.rs>
//! - The constant-time alternative, Pornin, [Optimized Binary GCD for Modular Inversion](https://eprint.iacr.org/2020/972)

use super::MontConfig;
use crate::BigInt;

/// Mask of the low 62 bits of a word.
const MASK62: u64 = (1u64 << 62) - 1;

/// A signed integer in radix `2^62`, least-significant limb first. Limbs below the active
/// length are in `[0, 2^62)` and the top active limb carries the sign.
type S62 = [i64; 5];

/// An upstream `VERIFY_CHECK`. Live in this crate's unit tests, including `--release` runs
/// where `debug_assert!` is inert, and in debug builds; compiled out otherwise.
macro_rules! verify {
    ($cond:expr $(, $($arg:tt)+)?) => {
        if cfg!(any(test, debug_assertions)) {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// The signed-radix-`2^62` constants of a [`MontConfig`], derived from `MODULUS`, `INV` and
/// `R2`. Meaningful only for `N == 4`; for other `N` the low 4 limbs are packed and nothing
/// reads the result.
pub(super) trait Inv62Params<const N: usize>: MontConfig<N> {
    /// `m` in signed radix `2^62`.
    const MODULUS62: S62 = pack62(&low4(&Self::MODULUS));
    /// `m^{-1} mod 2^62`. Not [`MontConfig::INV`], which is `-m^{-1} mod 2^64`.
    const MU: u64 = Self::INV.wrapping_neg() & MASK62;
    /// `R^2 mod m` in signed radix `2^62`: the initial `e`.
    const R2_62: S62 = pack62(&low4(&Self::R2));
    /// Pins `MU` against the low modulus limb. Forced by [`invert_counted`].
    const CHECK_MU: () = assert!(Self::MU.wrapping_mul(Self::MODULUS62[0] as u64) & MASK62 == 1);
}

impl<T: MontConfig<N>, const N: usize> Inv62Params<N> for T {}

/// The low 4 limbs of `x`, zero-padded.
const fn low4<const N: usize>(x: &BigInt<N>) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut i = 0;
    while i < 4 && i < N {
        out[i] = x.0[i];
        i += 1;
    }
    out
}

/// Repacks a canonical 4x64-bit little-endian value into signed radix `2^62`.
const fn pack62(x: &[u64; 4]) -> S62 {
    [
        (x[0] & MASK62) as i64,
        (((x[0] >> 62) | (x[1] << 2)) & MASK62) as i64,
        (((x[1] >> 60) | (x[2] << 4)) & MASK62) as i64,
        (((x[2] >> 58) | (x[3] << 6)) & MASK62) as i64,
        (x[3] >> 56) as i64,
    ]
}

/// Repacks a canonical signed-62 value in `[0, m)`, a [`normalize_62`] output, back into
/// `N` 64-bit little-endian limbs.
fn unpack62<const N: usize>(v: &S62) -> BigInt<N> {
    let [v0, v1, v2, v3, v4] = [
        v[0] as u64,
        v[1] as u64,
        v[2] as u64,
        v[3] as u64,
        v[4] as u64,
    ];
    verify!(
        v0 <= MASK62 && v1 <= MASK62 && v2 <= MASK62 && v3 <= MASK62 && v4 <= 0xff,
        "unpack input must be canonical and reduced"
    );
    let x = [
        v0 | (v1 << 62),
        (v1 >> 2) | (v2 << 60),
        (v2 >> 4) | (v3 << 58),
        (v3 >> 6) | (v4 << 56),
    ];
    let mut out = [0u64; N];
    let mut i = 0;
    while i < 4 && i < N {
        out[i] = x[i];
        i += 1;
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
    use super::{Inv62Params, MASK62, S62};
    use core::cmp::Ordering;

    /// `a * factor` over `alen` limbs of `a`, with all but the top limb normalized.
    pub(super) fn mul_62(a: &S62, alen: usize, factor: i64) -> S62 {
        let mut c: i128 = 0;
        let mut r = [0i64; 5];
        for i in 0..4 {
            if i < alen {
                c += i128::from(a[i]) * i128::from(factor);
            }
            r[i] = (c as u64 & MASK62) as i64;
            c >>= 62;
        }
        if 4 < alen {
            c += i128::from(a[4]) * i128::from(factor);
        }
        assert_eq!(c, i128::from(c as i64), "mul_62 top limb must fit i64");
        r[4] = c as i64;
        r
    }

    /// Compares `a`, over `alen` limbs, against `b * factor`.
    pub(super) fn mul_cmp_62(a: &S62, alen: usize, b: &S62, factor: i64) -> Ordering {
        let am = mul_62(a, alen, 1);
        let bm = mul_62(b, 5, factor);
        for i in 0..4 {
            assert!(
                am[i] >> 62 == 0 && bm[i] >> 62 == 0,
                "operand not normalized"
            );
        }
        for i in (0..5).rev() {
            if am[i] != bm[i] {
                return am[i].cmp(&bm[i]);
            }
        }
        Ordering::Equal
    }

    /// Asserts `-2m < x < m`, the coefficient range invariant.
    pub(super) fn assert_coeff_range<T: Inv62Params<N>, const N: usize>(x: &S62, what: &str) {
        let m = T::MODULUS62;
        assert_eq!(mul_cmp_62(x, 5, &m, -2), Ordering::Greater, "{what} > -2m");
        assert_eq!(mul_cmp_62(x, 5, &m, 1), Ordering::Less, "{what} < m");
    }

    /// Asserts `-m < f <= m` and `-m < g < m`.
    pub(super) fn assert_fg_range<T: Inv62Params<N>, const N: usize>(f: &S62, g: &S62, len: usize) {
        let m = T::MODULUS62;
        assert_eq!(mul_cmp_62(f, len, &m, -1), Ordering::Greater, "f > -m");
        assert_ne!(mul_cmp_62(f, len, &m, 1), Ordering::Greater, "f <= m");
        assert_eq!(mul_cmp_62(g, len, &m, -1), Ordering::Greater, "g > -m");
        assert_eq!(mul_cmp_62(g, len, &m, 1), Ordering::Less, "g < m");
    }

    pub(super) fn assert_non_negative(x: &S62, what: &str) {
        assert_ne!(mul_cmp_62(x, 5, &[0; 5], 0), Ordering::Less, "{what} >= 0");
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
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64) -> (Trans2x2, i64) {
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
        // eta starts at -1 and moves by at most one per divstep, over at most 744 divsteps.
        verify!((-745..=745).contains(&eta), "eta out of range");
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
fn update_fg_62_var(len: usize, f: &mut S62, g: &mut S62, t: &Trans2x2) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    verify!((1..=5).contains(&len), "active length out of range");
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
fn update_de_62<T: Inv62Params<N>, const N: usize>(d: &mut S62, e: &mut S62, t: &Trans2x2) {
    let (u, v, q, r) = (t.u, t.v, t.q, t.r);
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N>(d, "d");
        checks::assert_coeff_range::<T, N>(e, "e");
    }
    // [md, me] start at zero, plus [u, q] if d is negative, plus [v, r] if e is negative.
    let sd = d[4] >> 63;
    let se = e[4] >> 63;
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
    for j in 1..5 {
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
    d[4] = cd as i64;
    e[4] = ce as i64;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N>(d, "new d");
        checks::assert_coeff_range::<T, N>(e, "new e");
    }
}

/// [`update_de_62`] restricted to the top matrix row: the terminal batch, where only `d`
/// feeds the result and `e` is dead.
fn update_d_only_62<T: Inv62Params<N>, const N: usize>(d: &mut S62, e: &S62, u: i64, v: i64) {
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<T, N>(d, "d");
        checks::assert_coeff_range::<T, N>(e, "e");
    }
    let sd = d[4] >> 63;
    let se = e[4] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut cd = i128::from(u) * i128::from(d[0]) + i128::from(v) * i128::from(e[0]);
    md = md.wrapping_sub((T::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    cd += i128::from(m[0]) * i128::from(md);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    cd >>= 62;
    for j in 1..5 {
        cd += i128::from(u) * i128::from(d[j])
            + i128::from(v) * i128::from(e[j])
            + i128::from(m[j]) * i128::from(md);
        d[j - 1] = (cd as u64 & MASK62) as i64;
        cd >>= 62;
    }
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    d[4] = cd as i64;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<T, N>(d, "new d");
}

/// Brings `r` from `(-2m, m)` to `[0, m)`, negating first if `sign` is negative.
/// Upstream's `secp256k1_modinv64_normalize_62`.
fn normalize_62<T: Inv62Params<N>, const N: usize>(r: &mut S62, sign: i64) {
    let m = T::MODULUS62;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<T, N>(r, "normalize input");
    // Add the modulus if the input is negative, bringing r to (-m, m), then negate if
    // asked; still (-m, m), and every limb still fits i64.
    let mut cond_add = r[4] >> 63;
    let cond_negate = sign >> 63;
    for j in 0..5 {
        r[j] = ((r[j] + (m[j] & cond_add)) ^ cond_negate) - cond_negate;
    }
    for j in 0..4 {
        r[j + 1] += r[j] >> 62;
        r[j] &= MASK62 as i64;
    }
    // Add the modulus again if still negative, bringing r to [0, m).
    cond_add = r[4] >> 63;
    for j in 0..5 {
        r[j] += m[j] & cond_add;
    }
    for j in 0..4 {
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
            checks::mul_cmp_62(r, 5, &m, 1),
            core::cmp::Ordering::Less,
            "normalize output must be reduced"
        );
    }
}

/// Whether the `len` active limbs of `g` are all zero.
fn is_zero(g: &S62, len: usize) -> bool {
    if g[0] != 0 {
        return false;
    }
    g[1..len].iter().fold(0, |acc, &limb| acc | limb) == 0
}

/// Shrinks the active length of `f` and `g` by one when both top limbs are pure sign
/// extensions of the limb below, folding their signs down.
fn shrink_len(f: &mut S62, g: &mut S62, len: &mut usize) {
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
/// representation of the inverse. `None` for zero. Requires `N == 4`.
pub(super) fn invert<T: Inv62Params<N>, const N: usize>(x: &BigInt<N>) -> Option<BigInt<N>> {
    invert_counted::<T, N>(x).map(|(r, _)| r)
}

/// [`invert`], also reporting how many 62-divstep batches ran.
fn invert_counted<T: Inv62Params<N>, const N: usize>(x: &BigInt<N>) -> Option<(BigInt<N>, u32)> {
    let () = T::CHECK_MU;
    let x = low4(x);
    if x == [0u64; 4] {
        return None;
    }
    let mut f = T::MODULUS62;
    let mut g = pack62(&x);
    let mut d: S62 = [0; 5];
    let mut e = T::R2_62;
    let mut eta: i64 = -1;
    let mut len: usize = 5;

    // 12 * 62 = 744 divsteps cover the floor((49*256 + 57)/17) = 741 the bound asks for.
    for batch in 1..=12u32 {
        let (t, new_eta) = divsteps_62_var(eta, f[0] as u64, g[0] as u64);
        eta = new_eta;
        // f,g first: a terminating batch then needs only the top row of the d,e update.
        update_fg_62_var(len, &mut f, &mut g, &t);
        if is_zero(&g, len) {
            update_d_only_62::<T, N>(&mut d, &e, t.u, t.v);
            #[cfg(any(test, debug_assertions))]
            {
                let one = [1, 0, 0, 0, 0];
                let cmp = |s| checks::mul_cmp_62(&f, len, &one, s) == core::cmp::Ordering::Equal;
                assert!(cmp(1) || cmp(-1), "f must be +/-1 at termination");
            }
            // The sign of f lives in its top active limb.
            normalize_62::<T, N>(&mut d, f[len - 1]);
            return Some((unpack62(&d), batch));
        }
        update_de_62::<T, N>(&mut d, &mut e, &t);
        #[cfg(any(test, debug_assertions))]
        checks::assert_fg_range::<T, N>(&f, &g, len);
        shrink_len(&mut f, &mut g, &mut len);
    }
    panic!("safegcd inversion exceeded its 744-divstep bound");
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

    type F<C> = Fp<MontBackend<C, 4>, 4>;

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
    fn edge_reprs<C: MontConfig<4>>() -> Vec<F<C>> {
        let one = BigInt::from(1u64);
        let mut raw = vec![one, BigInt::from(2u64), C::R, C::R2];
        for sub in [&one, &C::R, &C::R2] {
            let mut v = C::MODULUS;
            v.sub_with_borrow(sub);
            raw.push(v);
        }
        for k in 0..256 {
            let mut limbs = [0u64; 4];
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
            for i in 0..4 {
                let mut limbs = [pat; 4];
                limbs[i] = !pat;
                raw.push(BigInt(limbs));
            }
        }
        raw.into_iter()
            .filter(|v| v < &C::MODULUS && v != &BigInt::from(0u64))
            .map(F::<C>::new_unchecked)
            .collect()
    }

    /// Bit identity with the BEA path `inverse` replaces, plus `a * a^-1 == 1`, over the
    /// edge representations and 2000 random elements.
    fn matches_bea<C: MontConfig<4>>() {
        assert!(F::<C>::ZERO.inverse().is_none());
        let mut rng = test_rng();
        let random = (0..20_000).map(|_| F::<C>::rand(&mut rng));
        let mut max_batches = 0;
        for a in edge_reprs::<C>().into_iter().chain(random) {
            let (inv, batches) = invert_counted::<C, 4>(&a.0).unwrap();
            let inv = F::<C>::new_unchecked(inv);
            assert_eq!(inv, a.inverse().unwrap(), "dispatch did not reach modinv62");
            assert_eq!(inv, a.bea_inverse().unwrap(), "divstep disagrees with BEA");
            assert_eq!(a * inv, F::<C>::one(), "a * a^-1 != 1");
            max_batches = max_batches.max(batches);
        }
        assert!(max_batches <= 12, "batch bound exceeded: {max_batches}");
    }

    fn check_vectors<C: MontConfig<4>>(vectors: &[([u64; 4], u32)]) {
        for &(limbs, batches) in vectors {
            let a = F::<C>::new_unchecked(BigInt(limbs));
            let (inv, count) = invert_counted::<C, 4>(&a.0).unwrap();
            assert_eq!(count, batches, "batch count drifted for {limbs:x?}");
            assert_eq!(a * F::<C>::new_unchecked(inv), F::<C>::one());
        }
    }

    #[test]
    fn pallas_base() {
        assert_eq!(
            PallasFq::MODULUS62,
            [0x192d30ed00000001, 0x091a63f02533e46e, 2, 0, 0x40]
        );
        assert_eq!(PallasFq::MU, 0x26d2cf1300000001);
        assert_eq!(
            PallasFq::R2_62,
            [
                0x0c78ecb30000000f,
                0x1f4c36f62c37839e,
                0x397a99bc3c95d18d,
                0x1b506bdee72dc51d,
                9
            ]
        );
        matches_bea::<PallasFq>();
        check_vectors::<PallasFq>(PALLAS_VECTORS);
    }

    #[test]
    fn vesta_base() {
        assert_eq!(
            VestaFq::MODULUS62,
            [0x0c46eb2100000001, 0x091a63f02652a376, 2, 0, 0x40]
        );
        assert_eq!(VestaFq::MU, 0x33b914df00000001);
        matches_bea::<VestaFq>();
        check_vectors::<VestaFq>(VESTA_VECTORS);
    }

    #[test]
    fn dense_modulus() {
        matches_bea::<Secp256k1Fq>();
    }
}
