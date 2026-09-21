//! Joint width-3 non-adjacent form over the Eisenstein integers `Z[\omega]`, for the GLV single
//! scalar multiplication.
//!
//! Port of Zakura `glv.rs` (`DELTA`, `JOINT_DIGITS`, `joint_digits`, `Table`, `Decomposed`),
//! itself the hexagonal-lattice analogue of the joint sparse form. A GLV decomposition
//! `k = a + b\lambda` is one Eisenstein integer `a + b\omega`, and recoding it in radix 2 against the 48
//! digits `U*\Delta` (the six units `{\pm 1, \pm\omega, \pm\omega^2}` times the eight orbit representatives
//! `DELTA`) leaves about 38.5 nonzero digits over about 127 columns, against the joint
//! sparse form's about 64 nonzero columns over 128. One 8-entry table serves all 48 digits,
//! because a unit acts on `(x, y)` as an x-rotation by `\zeta` and a y-negation.
//!
//! The correspondence the whole module rests on: `\omega -> \lambda` is a ring homomorphism
//! `Z[\omega] -> Z/n` because `\lambda` has order 3, hence `\lambda^2 + \lambda + 1 = 0`; and `\omega` acts on points as
//! [`GLVConfig::endomorphism`], which multiplies `x` by `\zeta = ENDO_COEFFS[0]` and is `[\lambda]`.
//!
//! # Sources
//!
//! Ported from Zakura `glv.rs`,
//! <https://github.com/zakura-core/common/blob/main/crates/pasta_curves/src/glv.rs>:
//!
//! - `DELTA` and `JOINT_DIGITS`, the orbit representatives and the residue-class lookup
//! - `joint_digits`, the recoding
//! - `Table` and its affine addition chain, with `affine_add_sub_pairs` and `batch_affine_ladder_raw`
//! - `Decomposed`
//! - `strauss_multiexp`, the multiexponentiation
//!
//! Their write-up of why the hexagonal lattice beats the joint sparse form:
//! <https://zakura.com/engineering/key-agreement/>. The GLV decomposition itself is from
//! [Faster Point Multiplication on Elliptic Curves with Efficient Endomorphisms](https://www.iacr.org/archive/crypto2001/21390189.pdf).

use super::{binary_scalar_mul_jsf, binary_scalar_mul_jsf_affine, GLVConfig};
use crate::{
    short_weierstrass::{Affine, Projective},
    AdditiveGroup, AffineRepr, CurveGroup,
};
use ark_ff::{serial_batch_inversion_and_mul_with_scratch, Field, One, PrimeField, Zero};
use ark_std::{marker::PhantomData, vec, vec::Vec};
use educe::Educe;

/// The eight Eisenstein digit-orbit representatives, as coefficient pairs `(a, b)` of
/// `a + b\omega` (norms 1, 3, 7, 7, 9, 13, 13, 19). The 48 nonzero digits are `U*\Delta` for the six
/// units `U = {\pm 1, \pm\omega, \pm\omega^2}`, and tile the odd residue classes mod `8Z[\omega]` exactly: 64
/// classes, minus the 16 divisible by 2, each hit once.
const DELTA: [(i8, i8); 8] = [
    (1, 0),
    (1, -1),
    (2, -1),
    (1, -2),
    (3, 0),
    (3, -1),
    (1, -3),
    (2, -3),
];

/// Digit lookup for the joint recoding, indexed by `((a mod 8) << 3) | (b mod 8)`: entry
/// `(da, db, code)` is the unique element `da + db*\omega` of `U*\Delta` congruent to `a + b\omega` mod
/// `8Z[\omega]`, or `(0, 0, 0)` for the 16 classes divisible by 2, where the recoder emits a zero
/// digit. `code` packs `1 + 6*orbit + unit` with units ordered `[+1, -1, +\omega, -\omega, +\omega^2, -\omega^2]`,
/// so `unit >> 1` is the rotation exponent and `unit & 1` the negation.
#[rustfmt::skip]
const JOINT_DIGITS: [(i8, i8, u8); 64] = [
    (0, 0, 0), (0, 1, 3), (0, 0, 0), (0, 3, 27), (0, 0, 0), (0, -3, 28), (0, 0, 0), (0, -1, 4),
    (1, 0, 1), (1, 1, 6), (1, 2, 9), (1, 3, 15), (1, 4, 33), (1, -3, 37), (1, -2, 19), (1, -1, 7),
    (0, 0, 0), (2, 1, 12), (0, 0, 0), (2, 3, 21), (0, 0, 0), (2, -3, 43), (0, 0, 0), (2, -1, 13),
    (3, 0, 25), (3, 1, 24), (3, 2, 18), (3, 3, 30), (3, 4, 39), (3, 5, 45), (-5, -2, 47), (3, -1, 31),
    (0, 0, 0), (4, 1, 42), (0, 0, 0), (4, 3, 36), (0, 0, 0), (-4, -3, 35), (0, 0, 0), (-4, -1, 41),
    (-3, 0, 26), (-3, 1, 32), (5, 2, 48), (-3, -5, 46), (-3, -4, 40), (-3, -3, 29), (-3, -2, 17), (-3, -1, 23),
    (0, 0, 0), (-2, 1, 14), (0, 0, 0), (-2, 3, 44), (0, 0, 0), (-2, -3, 22), (0, 0, 0), (-2, -1, 11),
    (-1, 0, 2), (-1, 1, 8), (-1, 2, 20), (-1, 3, 38), (-1, -4, 34), (-1, -3, 16), (-1, -2, 10), (-1, -1, 5),
];

/// Upper bound on the number of joint digit positions. The coefficients start below `2^128`
/// and each step at least halves them up to a coefficient-5 digit (`r' <= (r + 5)/2`), so 128
/// steps reach the box `max(|a|, |b|) <= 5`, which needs at most 5 further positions.
pub const MAX_JOINT_DIGITS: usize = 133;

/// Bit bound each GLV half must satisfy. Pasta's lattice basis keeps the halves below
/// `2^127`; secp256k1's does not, reaching 128 bits for about 5% of halves, which is why the
/// first recoding position runs on a sign-and-magnitude pair rather than an `i128`.
const HALF_BITS: u32 = 128;

/// Scalar-field width above which a GLV half cannot fit [`HALF_BITS`], since a half is about
/// half the field's width. Wider fields (bw6_761's 377-bit `Fr`) never reach the ladder, so
/// this const-folded check keeps their fallback free.
const MAX_SCALAR_BITS: u32 = 2 * HALF_BITS;

/// Splits a nonzero digit code into `(orbit, rotation, negate)`.
fn decode_digit(code: u8) -> (usize, usize, bool) {
    debug_assert!((1..=48).contains(&code), "invalid digit code");
    let v = usize::from(code - 1);
    (v / 6, (v % 6) >> 1, (v % 6) & 1 == 1)
}

/// The Eisenstein coefficients `(a, b)` of a nonzero digit code, i.e. the digit as `a + b\omega`.
/// Both are at most 5 in magnitude.
fn digit_coeffs(code: u8) -> (i8, i8) {
    let (orbit, e, negate) = decode_digit(code);
    let (mut a, mut b) = DELTA[orbit];
    // (a + b\omega)\omega = -b + (a - b)\omega.
    for _ in 0..e {
        let (ra, rb) = (-b, a - b);
        a = ra;
        b = rb;
    }
    if negate {
        a = -a;
        b = -b;
    }
    (a, b)
}

/// The scalar a nonzero digit multiplies the base point by, `a + b\lambda`. Never zero: the digit's
/// Eisenstein norm `a^2 - ab + b^2` is at most 75.
pub fn digit_scalar<P: GLVConfig>(code: u8) -> P::ScalarField {
    let (da, db) = digit_coeffs(code);
    let signed = |v: i8| {
        let m = P::ScalarField::from(u64::from(v.unsigned_abs()));
        if v < 0 {
            -m
        } else {
            m
        }
    };
    signed(da) + signed(db) * P::LAMBDA
}

/// A signed GLV half: `sign * magnitude`, with the magnitude up to `2^128 - 1`.
type Half = (bool, u128);

/// The value's residue mod 8.
fn residue((negative, magnitude): Half) -> u128 {
    if negative {
        0u128.wrapping_sub(magnitude) & 7
    } else {
        magnitude & 7
    }
}

/// `(v - d) / 2` in sign-and-magnitude form, exact because `v \equiv d (mod 2)`. With `p = m >> 1`,
/// `q = d >> 1` (floor) and the shared parity bit `r`, `(m - d)/2 = p - q` and
/// `(-m - d)/2 = -(p + q + r)`, evaluated without forming `m \pm d`, so no magnitude below `2^128`
/// overflows; the sign flips when the small term outweighs `p`. The result's magnitude is at
/// most `2^127 + 2`.
fn step_half((negative, magnitude): Half, d: i8) -> Half {
    let p = magnitude >> 1;
    let q = i128::from(d >> 1);
    let r = i128::from(d & 1);
    debug_assert_eq!((magnitude & 1) as i128, r, "digit parity must match the value");
    // `p - s` for a positive value, `-(p + s)` for a negative one.
    let s = if negative { q + r } else { q };
    let (grow, shrink) = if negative { (s >= 0, s < 0) } else { (s <= 0, s > 0) };
    let small = s.unsigned_abs();
    if grow {
        (negative, p + small)
    } else if shrink && p >= small {
        (negative, p - small)
    } else {
        (!negative, small - p)
    }
}

/// A sign-and-magnitude value whose magnitude fits `i128`.
fn to_i128((negative, magnitude): Half) -> i128 {
    debug_assert!(magnitude >> 127 == 0, "magnitude must fit i128");
    if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    }
}

/// Joint width-3 NAF recoding of `a + b\omega`, lowest position first: while the value is nonzero,
/// emit 0 if it is divisible by 2 (both coefficients even), else the unique `U*\Delta` digit
/// congruent to it mod `8Z[\omega]`; then subtract and halve. A nonzero digit leaves a multiple of
/// 8, so the next two positions are forced zeros.
///
/// The first two positions run on sign-and-magnitude pairs, since a half can be 128 bits wide:
/// one step leaves magnitudes up to `2^127 + 2`, which `i128` cannot hold, and the second
/// brings them under `2^126 + 4`; the rest of the loop is `i128`. `None` if the digit budget
/// is exhausted, which the input bound rules out.
fn joint_digits(a: Half, b: Half) -> Option<([u8; MAX_JOINT_DIGITS], usize)> {
    let mut digits = [0u8; MAX_JOINT_DIGITS];
    let mut n = 0;
    let (mut a, mut b) = (a, b);
    for _ in 0..2 {
        if a.1 == 0 && b.1 == 0 {
            return Some((digits, n));
        }
        let (da, db, code) = JOINT_DIGITS[((residue(a) << 3) | residue(b)) as usize];
        digits[n] = code;
        n += 1;
        a = step_half(a, da);
        b = step_half(b, db);
    }
    let (mut a, mut b) = (to_i128(a), to_i128(b));
    while a != 0 || b != 0 {
        if n == MAX_JOINT_DIGITS {
            return None;
        }
        let idx = (((a & 7) << 3) | (b & 7)) as usize;
        let (da, db, code) = JOINT_DIGITS[idx];
        digits[n] = code;
        a = (a >> 1) - (i128::from(da) >> 1);
        b = (b >> 1) - (i128::from(db) >> 1);
        n += 1;
    }
    // A leading zero digit only happens when the value was a multiple of 2 all the way down,
    // which the zero shortcut above already excludes.
    debug_assert!(digits[n - 1] != 0);
    Some((digits, n))
}

/// A scalar recoded as one joint Eisenstein digit string, reusable across any number of
/// [`Table`]s.
#[derive(Clone, Copy, Debug)]
pub struct Decomposed<P: GLVConfig> {
    /// Digit codes, lowest position first; zero at `len` and beyond.
    digits: [u8; MAX_JOINT_DIGITS],
    len: usize,
    phantom: PhantomData<P>,
}

impl<P: GLVConfig> Decomposed<P> {
    /// Decomposes `k` and recodes the halves. `None` when a half does not fit the recoding's
    /// 128-bit bound, which sends the caller back to the joint sparse form: scalar fields
    /// wider than 256 bits always, and any curve whose rounding pushes a half over.
    pub fn new(k: P::ScalarField) -> Option<Self> {
        // A scalar already narrow enough to be one half is its own decomposition, `k = k + 0w`,
        // so the lattice reduction is dead work. Zero takes this path too. The miss costs the
        // one `into_bigint` that `half` pays on the hit anyway. Generalizes two narrower forms of
        // the same skip in Zakura `glv/zero.rs`: canonical 10-bit table values recoded as
        // `(q, 0)`, and exact zeros.
        if let Some(d) = Self::new_given_halves(true, k, true, P::ScalarField::ZERO) {
            return Some(d);
        }
        let ((sgn_a, a), (sgn_b, b)) = P::scalar_decomposition(k);
        Self::new_given_halves(sgn_a, a, sgn_b, b)
    }

    /// Same as [`Self::new`] but on an already computed decomposition.
    pub fn new_given_halves(
        sgn_a: bool,
        a: P::ScalarField,
        sgn_b: bool,
        b: P::ScalarField,
    ) -> Option<Self> {
        if P::ScalarField::MODULUS_BIT_SIZE > MAX_SCALAR_BITS {
            return None;
        }
        let (digits, len) = joint_digits(half(sgn_a, a)?, half(sgn_b, b)?)?;
        Some(Self {
            digits,
            len,
            phantom: PhantomData,
        })
    }

    /// Digit positions in use.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the scalar was zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The digit code at `position`, zero past [`Self::len`].
    pub fn digit(&self, position: usize) -> u8 {
        self.digits[position]
    }
}

/// Whether `k`'s column schedule avoids every exceptional case of the batch-affine ladder,
/// tracking the scalar `s` that multiplies the base point in the accumulator. The top digit
/// makes `s` nonzero (a digit's Eisenstein norm is at most 75, far below the group order) and
/// doubling preserves that in an odd-order field, so doubling columns never hit `2y = 0`. An
/// active column computes `2P + D` with `P = [s]B`, `D = [d]B`; the direct denominator
/// `(x(D) - x(P))^2 (2x(P) + x(D)) - (y(D) - y(P))^2` vanishes exactly when `d = s` or `d = -2s`
/// (the latter also being when the output is the identity), and unlike the two-step formula it
/// handles `d = -s` (`D = -P`) directly. The conditions depend only on the schedule, so one check
/// covers the whole batch; random scalars fail it with probability about `2^-124`.
///
/// The ~100 field multiplies here (a `digit_scalar` per nonzero digit) run once per
/// [`Table::mul_decomposed_batch`] call. The 48 possible digit scalars are built once and looked up.
fn affine_ladder_safe<P: GLVConfig>(k: &Decomposed<P>) -> bool {
    debug_assert!(k.len > 0);
    let mut digit_scalars = [P::ScalarField::ZERO; 49];
    for (code, slot) in digit_scalars.iter_mut().enumerate().skip(1) {
        *slot = digit_scalar::<P>(code as u8);
    }
    let mut s = digit_scalars[usize::from(k.digits[k.len - 1])];
    for &code in k.digits[..k.len - 1].iter().rev() {
        if code == 0 {
            s = s.double();
        } else {
            let d = digit_scalars[usize::from(code)];
            let s2 = s.double();
            if d == s || d == -s2 {
                return false;
            }
            s = s2 + d;
        }
    }
    true
}

/// A GLV half as a sign and a magnitude, or `None` if it exceeds `HALF_BITS` bits.
fn half<F: PrimeField>(positive: bool, v: F) -> Option<Half> {
    let repr = v.into_bigint();
    let limbs = repr.as_ref();
    if limbs.len() < 2 || limbs[2..].iter().any(|&l| l != 0) {
        return None;
    }
    debug_assert_eq!(HALF_BITS, 128);
    Some((
        !positive,
        u128::from(limbs[0]) | (u128::from(limbs[1]) << 64),
    ))
}

/// Fewest points for [`Table::batch`]'s affine addition chain. Below it the five layers'
/// inversions and their temporaries cost more than the projective build they replace. On Pallas
/// (`batch_affine_bench`) the two are level around 3 and the affine chain pulls ahead from 4 up;
/// Zakura's own threshold is 8, and the safegcd inversion moves it down.
pub const TABLE_BATCH_AFFINE_MIN_POINTS: usize = 4;

/// Fewest points for the synchronized batch-affine ladder [`Table::mul_decomposed_batch`].
/// Below it the per-column batch inversions and the schedule scan cost more than the per-point
/// projective ladders they replace. Local kernel-vs-fallback crossover on Pallas (aarch64, safegcd
/// inversion): the ladder loses at 16 (0.87x), is marginal at 24 (1.03x), and wins from 32 up
/// (1.10x, rising to 1.38x by 96), so 32 is the smallest robustly winning size. Matches Zakura's 32.
const AFFINE_LADDER_MIN_POINTS: usize = 32;

/// The eight orbit representatives `[\Delta_i]P` in affine form, with each x-coordinate's three
/// `\zeta`-rotations. One table serves all 48 digits.
#[derive(Educe)]
#[educe(Clone, Copy, Debug)]
pub struct Table<P: GLVConfig> {
    /// `xs[e][i] = \zeta^e * x([\Delta_i]P)`.
    xs: [[P::BaseField; 8]; 3],
    /// `ys[i] = y([\Delta_i]P)`.
    ys: [P::BaseField; 8],
}

impl<P: GLVConfig> Table<P> {
    /// One table, at the cost of one field inversion. Amortize that with [`Self::batch`].
    pub fn new(p: &Projective<P>) -> Self {
        Self::from_window_points(&Projective::normalize_batch(&Self::window_points(p)))
    }

    /// One table per point. Above [`TABLE_BATCH_AFFINE_MIN_POINTS`] points the addition
    /// chain runs in affine form, one shared inversion per layer, which replaces the seven
    /// projective additions and the `8n`-entry normalization with five inversions over `n`.
    /// Below it, or once a layer hits an exceptional pair, the projective build stands.
    /// Identity inputs give identity tables and may be mixed in.
    pub fn batch(points: &[Projective<P>]) -> Vec<Self> {
        let n = points.len();
        if n < TABLE_BATCH_AFFINE_MIN_POINTS {
            return Self::batch_projective(points);
        }
        // Affine chords have no identity case, so those lanes sit out and keep their identity
        // tables.
        let mut non_zero_point_indices = Vec::with_capacity(n);
        let mut non_zero_points = Vec::with_capacity(n);
        for (i, p) in points.iter().enumerate() {
            if !p.is_zero() {
                non_zero_point_indices.push(i);
                non_zero_points.push(*p);
            }
        }
        if non_zero_points.len() < TABLE_BATCH_AFFINE_MIN_POINTS {
            return Self::batch_projective(points);
        }

        let endo = P::endomorphism_affine;
        let p = Projective::normalize_batch(&non_zero_points);
        let len = p.len();
        let mut tables = vec![Self::from_window_points(&[Affine::<P>::zero(); 8]); n];
        let mut den = Vec::with_capacity(len);
        // One prefix-product buffer for the five layers' batch inversions, reused across layers.
        let mut inv_scratch = Vec::with_capacity(len);
        let mut spare = vec![Affine::<P>::zero(); len];
        let mut free = true;

        let phi_p = p.iter().map(endo).collect::<Vec<_>>();
        let mut d1 = vec![Affine::<P>::zero(); len];
        free = affine_add_sub_pairs::<P>(&p, &phi_p, &mut spare, &mut d1, &mut den, &mut inv_scratch, free);

        let d1_endo = d1.iter().map(endo).collect::<Vec<_>>();
        let mut b = vec![Affine::<P>::zero(); len];
        free = affine_add_sub_pairs::<P>(&d1, &d1_endo, &mut spare, &mut b, &mut den, &mut inv_scratch, free);

        let b_endo = b.iter().map(endo).collect::<Vec<_>>();
        let m3 = b_endo.iter().map(endo).collect::<Vec<_>>();
        let mut t3a = vec![Affine::<P>::zero(); len];
        let mut t3b = vec![Affine::<P>::zero(); len];
        free = affine_add_sub_pairs::<P>(&phi_p, &m3, &mut t3a, &mut t3b, &mut den, &mut inv_scratch, free);

        let r3 = b_endo.iter().map(|q| -*q).collect::<Vec<_>>();
        let mut t4a = vec![Affine::<P>::zero(); len];
        let mut t4b = vec![Affine::<P>::zero(); len];
        free = affine_add_sub_pairs::<P>(&phi_p, &r3, &mut t4a, &mut t4b, &mut den, &mut inv_scratch, free);

        let mut t19 = vec![Affine::<P>::zero(); len];
        affine_add_sub_pairs::<P>(&phi_p, &t4b, &mut t19, &mut spare, &mut den, &mut inv_scratch, free);

        for (lane, index) in non_zero_point_indices.into_iter().enumerate() {
            tables[index] = Self::from_window_points(&[
                p[lane],
                d1[lane],
                endo(&t4a[lane]),
                -endo(&t3b[lane]),
                -m3[lane],
                -t3a[lane],
                endo(&endo(&t4b[lane])),
                endo(&endo(&t19[lane])),
            ]);
        }
        tables
    }

    /// [`Self::batch`]'s projective build: `8n` window entries through one shared inversion.
    /// Public for the benchmark that fixes [`TABLE_BATCH_AFFINE_MIN_POINTS`].
    pub fn batch_projective(points: &[Projective<P>]) -> Vec<Self> {
        let mut proj = Vec::with_capacity(points.len() * 8);
        for p in points {
            proj.extend_from_slice(&Self::window_points(p));
        }
        Projective::normalize_batch(&proj)
            .chunks_exact(8)
            .map(Self::from_window_points)
            .collect()
    }

    /// The eight projective `[\Delta_i]P`, in orbit order: seven additions plus endomorphism
    /// applications (one base-field multiply each) and negations, no inversion, so the
    /// entries ride along in the caller's shared normalization. Conjugate pairs share their
    /// intermediates; the trailing unit of each chain output is stripped with rotations and
    /// negations.
    fn window_points(p: &Projective<P>) -> [Projective<P>; 8] {
        let endo = P::endomorphism;
        let phi_p = endo(p); // \omega
        let d1 = *p - phi_p; // 1 - \omega
        let b = d1 - endo(&d1); // (1 - \omega)^2 = -3\omega
        let b_endo = endo(&b); // -3\omega^2
        let m3 = endo(&b_endo); // -3
        let t3a = phi_p + m3; // -3 + \omega
        let t3b = phi_p - m3; // 3 + \omega
        let r3 = -b_endo; // 3\omega^2
        let t4a = phi_p + r3; // -3 - 2\omega
        let t4b = phi_p - r3; // 3 + 4\omega
        let t19 = phi_p + t4b; // 3 + 5\omega
        [
            *p,                        // \Delta_0 = 1
            d1,                        // \Delta_1 = 1 - \omega
            endo(&t4a),                // \Delta_2 = 2 - \omega = \omega(-3 - 2\omega)
            -endo(&t3b),               // \Delta_3 = 1 - 2\omega = -\omega(3 + \omega)
            -m3,                       // \Delta_4 = 3
            -t3a,                      // \Delta_5 = 3 - \omega
            endo(&endo(&t4b)),         // \Delta_6 = 1 - 3\omega = \omega^2(3 + 4\omega)
            endo(&endo(&t19)),         // \Delta_7 = 2 - 3\omega = \omega^2(3 + 5\omega)
        ]
    }

    /// Assembles a table from one normalized 8-entry window, materializing the `\zeta`-rotations
    /// of each x-coordinate. `\zeta` is a nontrivial cube root of unity, so `\zeta^2 x = -x - \zeta x`.
    fn from_window_points(w: &[Affine<P>]) -> Self {
        let zeta = P::ENDO_COEFFS[0];
        let mut xs = [[P::BaseField::ZERO; 8]; 3];
        let mut ys = [P::BaseField::ZERO; 8];
        for (i, p) in w.iter().enumerate() {
            if p.is_zero() {
                continue;
            }
            let xz = p.x * zeta;
            xs[0][i] = p.x;
            xs[1][i] = xz;
            xs[2][i] = -p.x - xz;
            ys[i] = p.y;
        }
        Self { xs, ys }
    }

    /// Whether this is the identity's table. Identity windows are all `(0, 0)`, and the group
    /// has odd prime order so no valid point has `y = 0`.
    pub fn is_identity(&self) -> bool {
        self.ys[0].is_zero()
    }

    /// The affine coordinates a nonzero digit contributes, `\pm\phi^e([\Delta_i]P)`: one lookup and
    /// at most one negation.
    fn digit_coords(&self, code: u8) -> (P::BaseField, P::BaseField) {
        let (orbit, e, negate) = decode_digit(code);
        let y = if negate {
            -self.ys[orbit]
        } else {
            self.ys[orbit]
        };
        (self.xs[e][orbit], y)
    }

    /// The affine point a nonzero digit contributes.
    pub fn digit_point(&self, code: u8) -> Affine<P> {
        let (x, y) = self.digit_coords(code);
        Affine::new_unchecked(x, y)
    }

    /// `k * P` for the point this table encodes: one doubling per column and one mixed
    /// addition per nonzero digit.
    pub fn mul_decomposed(&self, k: &Decomposed<P>) -> Projective<P> {
        if self.is_identity() || k.len == 0 {
            return Projective::zero();
        }
        let mut acc = Projective::zero();
        for (i, &code) in k.digits[..k.len].iter().enumerate().rev() {
            // The accumulator is still the identity at the top column.
            if i + 1 < k.len {
                acc.double_in_place();
            }
            if code != 0 {
                acc += self.digit_point(code);
            }
        }
        acc
    }

    /// One `k * P` per table, sharing the whole column schedule across the batch. Equal, table by
    /// table, to [`Self::mul_decomposed`]. With at least [`AFFINE_LADDER_MIN_POINTS`]
    /// non-identity tables and an [`affine_ladder_safe`] schedule, the ladder runs on affine
    /// accumulators with one batch inversion per column; otherwise every table falls back to its
    /// own per-point ladder.
    pub fn mul_decomposed_batch(tables: &[Self], k: &Decomposed<P>) -> Vec<Projective<P>> {
        if k.len == 0 {
            return vec![Projective::zero(); tables.len()];
        }
        let non_zero: Vec<&Self> = tables.iter().filter(|t| !t.is_identity()).collect();
        if non_zero.len() < AFFINE_LADDER_MIN_POINTS || !affine_ladder_safe(k) {
            return tables.iter().map(|t| t.mul_decomposed(k)).collect();
        }
        let (xs, ys) = batch_affine_ladder_raw(&non_zero, k);
        let mut products = xs
            .into_iter()
            .zip(ys)
            .map(|(x, y)| Affine::<P>::new_unchecked(x, y).into_group());
        tables
            .iter()
            .map(|t| {
                if t.is_identity() {
                    Projective::zero()
                } else {
                    products.next().expect("one product per non-zero table")
                }
            })
            .collect()
    }
}

/// `left_i + right_i` into `sums` and `left_i - right_i` into `diffs`, one inversion for the
/// whole slice. Both chords are `(\pm y_R - y_L) / (x_R - x_L)`, so a single batch inversion of
/// the shared denominators serves them.
///
/// `identity_free` carries the caller's knowledge that no operand can be the identity; while it
/// is false every layer rescans, because affine chords have no identity case and no doubling
/// case. Returns it unchanged on success and `false` after a fallback.
fn affine_add_sub_pairs<P: GLVConfig>(
    left: &[Affine<P>],
    right: &[Affine<P>],
    sums: &mut [Affine<P>],
    diffs: &mut [Affine<P>],
    den: &mut Vec<P::BaseField>,
    inv_scratch: &mut Vec<P::BaseField>,
    identity_free: bool,
) -> bool {
    if !identity_free
        && left
            .iter()
            .zip(right)
            .any(|(l, r)| l.is_zero() || r.is_zero())
    {
        projective_add_sub_pairs::<P>(left, right, sums, diffs);
        return false;
    }
    den.clear();
    den.extend(left.iter().zip(right).map(|(l, r)| r.x - l.x));
    // Equal x means a doubling or a vertical line, neither of which the chord below computes.
    // Checked before anything is written, so the fallback sees untouched outputs.
    if den.iter().any(Zero::is_zero) {
        projective_add_sub_pairs::<P>(left, right, sums, diffs);
        return false;
    }
    serial_batch_inversion_and_mul_with_scratch(den, &P::BaseField::one(), inv_scratch);

    for (i, (l, r)) in left.iter().zip(right).enumerate() {
        let x_sum = l.x + r.x;
        let chord = |num: P::BaseField| {
            let lambda = num * den[i];
            let x = lambda.square() - x_sum;
            Affine::new_unchecked(x, lambda * (l.x - x) - l.y)
        };
        sums[i] = chord(r.y - l.y);
        diffs[i] = chord(-r.y - l.y);
    }
    identity_free
}

/// [`affine_add_sub_pairs`] for the pairs its chords cannot take: one projective addition and
/// one subtraction each, through a shared normalization.
fn projective_add_sub_pairs<P: GLVConfig>(
    left: &[Affine<P>],
    right: &[Affine<P>],
    sums: &mut [Affine<P>],
    diffs: &mut [Affine<P>],
) {
    let mut proj = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right) {
        let l = l.into_group();
        proj.extend_from_slice(&[l + r, l - r]);
    }
    for (i, pair) in Projective::normalize_batch(&proj).chunks_exact(2).enumerate() {
        sums[i] = pair[0];
        diffs[i] = pair[1];
    }
}

/// Multiplies field values in place, stage by stage across a batch.
#[inline(always)]
fn mul_assign_batch<F: Field>(lhs: &mut [F], rhs: &[F]) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        *lhs *= rhs;
    }
}

/// Completes a batch of one-inversion affine `2P + Q` formulas stage by stage from the inverse of
/// the direct denominator `(u - x)^2 (2x + u) - (v - y)^2`. `h = u - x`, `r = v - y`, `h_squared`
/// and `denominator_inverse` enter with their column values; each intermediate overwrites an input
/// after its last use, and `denominator_inverse` leaves holding the addend for the accumulator's
/// new y.
fn double_add_finish_batch<F: Field>(
    y: &[F],
    h: &mut [F],
    r: &mut [F],
    h_squared: &mut [F],
    denominator_inverse: &mut [F],
) {
    mul_assign_batch(denominator_inverse, y);
    mul_assign_batch(h, denominator_inverse);
    mul_assign_batch(h_squared, h);
    for i in 0..r.len() {
        r[i] = h_squared[i] - r[i];
    }
    mul_assign_batch(h, r);
    mul_assign_batch(denominator_inverse, r);
    for ((factor, square), lambda) in denominator_inverse
        .iter_mut()
        .zip(h_squared.iter_mut())
        .zip(r)
    {
        *factor = F::ONE + factor.double().double();
        *square += *lambda;
    }
    mul_assign_batch(denominator_inverse, h_squared);
}

/// The synchronized batch-affine ladder kernel. Every table steps the shared digit schedule
/// in lockstep on affine accumulators (structure-of-arrays `xs`/`ys`), sharing each column's
/// inversion: a zero digit is a batched affine doubling, a nonzero digit a direct affine `2P + D`
/// whose two dependent chord denominators are algebraically combined into one inversion batch.
/// Returns the raw accumulator coordinates. Callers guarantee `k.len > 0`, every table
/// non-identity, and an [`affine_ladder_safe`] schedule, so no denominator is zero and no
/// accumulator is the identity.
///
/// Ported from Zakura `glv.rs` `batch_affine_ladder_raw`.
fn batch_affine_ladder_raw<P: GLVConfig>(
    tables: &[&Table<P>],
    k: &Decomposed<P>,
) -> (Vec<P::BaseField>, Vec<P::BaseField>) {
    // The doubling numerator is `3x^2`, omitting `+ a`; every curve carrying this endomorphism has
    // `a = 0`, which the whole module already relies on.
    debug_assert!(P::COEFF_A.is_zero(), "affine ladder assumes COEFF_A = 0");
    let n = tables.len();
    // Affine accumulators, seeded from the top digit; the ladder's first column is the digit itself.
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for t in tables {
        let (x, y) = t.digit_coords(k.digits[k.len - 1]);
        xs.push(x);
        ys.push(y);
    }
    let mut h_squares = vec![P::BaseField::ZERO; n];
    let mut h = vec![P::BaseField::ZERO; n];
    let mut r = vec![P::BaseField::ZERO; n];
    let mut a = vec![P::BaseField::ZERO; n];
    // One prefix-product buffer for the per-column batch inversion, reused across all columns
    // instead of allocated per column.
    let mut inv_scratch = Vec::with_capacity(n);
    let one = P::BaseField::one();

    for &code in k.digits[..k.len - 1].iter().rev() {
        if code == 0 {
            // Batched affine doubling: m = 3x^2 / 2y, x' = m^2 - 2x, y' = m(x - x') - y.
            for (a, y) in a.iter_mut().zip(&ys) {
                *a = y.double();
            }
            serial_batch_inversion_and_mul_with_scratch(&mut a, &one, &mut inv_scratch);
            for i in 0..n {
                let x_sqr = xs[i].square();
                // m = 3x^2 + a
                let m = (x_sqr.double() + x_sqr) * a[i];
                let x2 = m.square() - xs[i].double();
                ys[i] = m * (xs[i] - x2) - ys[i];
                xs[i] = x2;
            }
        } else {
            let (orbit, e, negate) = decode_digit(code);
            // Direct affine 2P + D: one inversion batch of the combined denominator
            // (u - x)^2 (2x + u) - (v - y)^2.
            for i in 0..n {
                let u = tables[i].xs[e][orbit];
                let v = if negate {
                    -tables[i].ys[orbit]
                } else {
                    tables[i].ys[orbit]
                };
                h[i] = u - xs[i];
                r[i] = v - ys[i];
                h_squares[i] = h[i].square();
                a[i] = xs[i].double() + u;
            }
            mul_assign_batch(&mut a, &h_squares);
            for i in 0..n {
                a[i] -= r[i].square();
            }
            serial_batch_inversion_and_mul_with_scratch(&mut a, &one, &mut inv_scratch);
            double_add_finish_batch(&ys, &mut h, &mut r, &mut h_squares, &mut a);
            for i in 0..n {
                let u = tables[i].xs[e][orbit];
                xs[i] = u + h[i].double().double();
                ys[i] = -ys[i] - a[i];
            }
        }
    }
    (xs, ys)
}

/// `sum(bases_i * scalars_i)` by Straus over the joint Eisenstein digit strings: every scalar
/// is recoded once, one [`Table::batch`] covers every base with a single inversion for all
/// `8n` window entries, and one ladder of `max len` columns pays one doubling per column and
/// one mixed addition per nonzero digit. The bucket algorithm's 255 doublings and 85 window
/// reductions are replaced by about 126 doublings shared across the whole sum.
///
/// `None` when any decomposition misses the recoding's bound, which sends the caller back to
/// the bucket path.
///
/// Ported from Zakura `strauss_multiexp`.
pub fn eisenstein_msm<P: GLVConfig>(
    bases: &[Affine<P>],
    scalars: &[P::ScalarField],
) -> Option<Projective<P>> {
    let n = bases.len().min(scalars.len());
    let mut decomposed = Vec::with_capacity(n);
    for k in &scalars[..n] {
        decomposed.push(Decomposed::<P>::new(*k)?);
    }
    let columns = decomposed.iter().map(Decomposed::len).max().unwrap_or(0);
    if columns == 0 {
        return Some(Projective::zero());
    }

    let proj = bases[..n].iter().map(|p| p.into_group()).collect::<Vec<_>>();
    let tables = Table::<P>::batch(&proj);
    let non_zero = tables
        .iter()
        .zip(&decomposed)
        .filter(|(t, _)| !t.is_identity())
        .collect::<Vec<_>>();

    let mut acc = Projective::zero();
    for column in (0..columns).rev() {
        if column + 1 < columns {
            acc.double_in_place();
        }
        for (table, digits) in &non_zero {
            let code = digits.digit(column);
            if code != 0 {
                acc += table.digit_point(code);
            }
        }
    }
    Some(acc)
}

/// `k * p` through the Eisenstein ladder, falling back to the joint sparse form on the same
/// decomposition when a half misses the recoding's bound.
pub fn eisenstein_mul_projective<P: GLVConfig>(
    p: Projective<P>,
    k: P::ScalarField,
) -> Projective<P> {
    if let Some(d) = Decomposed::<P>::new(k) {
        return Table::new(&p).mul_decomposed(&d);
    }
    // Only the fallback needs the halves, so it pays the decomposition a second time.
    let ((sa, a), (sb, b)) = P::scalar_decomposition(k);
    let mut b1 = p;
    let mut b2 = P::endomorphism(&p);
    if !sa {
        b1 = -b1;
    }
    if !sb {
        b2 = -b2;
    }
    binary_scalar_mul_jsf(b1, a, b2, b)
}

/// [`eisenstein_mul_projective`] for an affine base.
pub fn eisenstein_mul_affine<P: GLVConfig>(p: Affine<P>, k: P::ScalarField) -> Projective<P> {
    if let Some(d) = Decomposed::<P>::new(k) {
        return Table::new(&p.into_group()).mul_decomposed(&d);
    }
    // Only the fallback needs the halves, so it pays the decomposition a second time.
    let ((sa, a), (sb, b)) = P::scalar_decomposition(k);
    let mut b1 = p;
    let mut b2 = P::endomorphism_affine(&p);
    if !sa {
        b1 = -b1;
    }
    if !sb {
        b2 = -b2;
    }
    binary_scalar_mul_jsf_affine(b1, a, b2, b)
}

/// `k * p_i` for every point, decomposing `k` once and sharing one inversion across all the tables.
pub fn glv_mul_same_scalar<P: GLVConfig>(
    points: &[Affine<P>],
    k: P::ScalarField,
) -> Vec<Projective<P>> {
    let Some(d) = Decomposed::<P>::new(k) else {
        return points
            .iter()
            .map(|p| eisenstein_mul_affine::<P>(*p, k))
            .collect();
    };
    let proj = points.iter().map(|p| p.into_group()).collect::<Vec<_>>();
    let tables = Table::batch(&proj);
    Table::mul_decomposed_batch(&tables, &d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::vec;
    use num_bigint::BigInt;

    /// `(a + b\omega) * \omega = -b + (a - b)\omega`.
    fn mul_omega((a, b): (i32, i32)) -> (i32, i32) {
        (-b, a - b)
    }

    /// The 48 nonzero digits, as `(code, (a, b))`, built from the definitions rather than the
    /// table: `code = 1 + 6*orbit + unit` with units `[+1, -1, +\omega, -\omega, +\omega^2, -\omega^2]`.
    fn all_digits() -> Vec<(u8, (i32, i32))> {
        let mut out = Vec::with_capacity(48);
        for (orbit, &(da, db)) in DELTA.iter().enumerate() {
            let mut d = (i32::from(da), i32::from(db));
            for rotation in 0..3 {
                for negate in [false, true] {
                    let unit = 2 * rotation + usize::from(negate);
                    let code = (1 + 6 * orbit + unit) as u8;
                    let v = if negate { (-d.0, -d.1) } else { d };
                    out.push((code, v));
                }
                d = mul_omega(d);
            }
        }
        out
    }

    /// Rebuilds [`JOINT_DIGITS`] from the definitions: every odd residue class mod `8Z[\omega]` is
    /// hit by exactly one of the 48 digits, and the 16 classes divisible by 2 by none.
    #[test]
    fn joint_digit_table_matches_first_principles() {
        let mut seen = vec![None; 64];
        for (code, (a, b)) in all_digits() {
            assert_eq!(digit_coeffs(code), (a as i8, b as i8), "code {code}");
            let idx = (((a & 7) << 3) | (b & 7)) as usize;
            assert!(seen[idx].is_none(), "class {idx} hit twice");
            seen[idx] = Some((a as i8, b as i8, code));
            assert_eq!(
                JOINT_DIGITS[idx],
                (a as i8, b as i8, code),
                "class {idx}"
            );
        }
        for (idx, entry) in JOINT_DIGITS.iter().enumerate() {
            let even = (idx >> 3) % 2 == 0 && (idx & 7) % 2 == 0;
            assert_eq!(seen[idx].is_none(), even, "class {idx}");
            assert_eq!(*entry == (0, 0, 0), even, "class {idx}");
        }
    }

    /// The digits must reconstruct the value: `sum_i 2^i * digit_i = a + b\omega`.
    #[test]
    fn joint_recoding_reconstructs_the_value() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut cases: Vec<(Half, Half)> = Vec::new();
        for a in -20i128..=20 {
            for b in -20i128..=20 {
                cases.push((
                    (a < 0, a.unsigned_abs()),
                    (b < 0, b.unsigned_abs()),
                ));
            }
        }
        for _ in 0..4000 {
            let pick = |x: u64, y: u64| ((x & 1) == 1, (u128::from(x) << 64) | u128::from(y));
            cases.push((pick(next(), next()), pick(next(), next())));
        }
        // The corners of the 128-bit magnitude range, where the wide first step matters.
        for a in [0u128, 1, (1 << 127) - 1, 1 << 127, u128::MAX - 4] {
            for b in [0u128, 1, 1 << 127, u128::MAX - 4] {
                for signs in [(false, false), (true, false), (false, true), (true, true)] {
                    cases.push(((signs.0, a), (signs.1, b)));
                }
            }
        }
        // Every residue class at the top of the range, so each of the 48 digits, the `\pm 5`
        // coefficients included, meets a magnitude within 8 of `2^128`.
        for ra in 0u128..8 {
            for rb in 0u128..8 {
                for signs in [(false, false), (true, false), (false, true), (true, true)] {
                    cases.push(((signs.0, u128::MAX - 7 + ra), (signs.1, u128::MAX - 7 + rb)));
                }
            }
        }
        for (a, b) in cases {
            let (digits, len) = joint_digits(a, b).expect("within the digit budget");
            assert!(len <= MAX_JOINT_DIGITS);
            assert!(len == 0 || digits[len - 1] != 0, "trailing zero digit");
            // Exact reconstruction: a full 128-bit magnitude does not fit `i128`, and wrapping
            // arithmetic would hide a first step that wrapped the same way.
            let (mut ra, mut rb) = (BigInt::from(0), BigInt::from(0));
            for &code in digits[..len].iter().rev() {
                ra *= 2;
                rb *= 2;
                if code != 0 {
                    let (da, db) = digit_coeffs(code);
                    ra += da;
                    rb += db;
                }
            }
            let expect = |v: Half| {
                let m = BigInt::from(v.1);
                if v.0 {
                    -m
                } else {
                    m
                }
            };
            assert_eq!(
                (ra, rb),
                (expect(a), expect(b)),
                "reconstruction of ({a:?}, {b:?})"
            );
        }
    }

    /// A nonzero digit leaves a multiple of 8, so the next two positions are zero.
    #[test]
    fn nonzero_digits_are_spaced() {
        let (digits, len) =
            joint_digits((false, u128::from(u64::MAX) << 60), (true, 1234567890123456789)).unwrap();
        for i in 0..len {
            if digits[i] != 0 {
                assert!(i + 1 >= len || digits[i + 1] == 0, "position {i}");
                assert!(i + 2 >= len || digits[i + 2] == 0, "position {i}");
            }
        }
    }
}
