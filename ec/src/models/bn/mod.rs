use crate::{
    models::{fp12_lines as lines, short_weierstrass::SWCurveConfig, CurveConfig},
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    scalar_mul::glv::mul_shift_round_bigint,
};
use ark_ff::{
    fields::{
        fp12_2over3over2::{Fp12, Fp12Config},
        fp2::Fp2Config,
        fp6_3over2::Fp6Config,
        Field, Fp2, PrimeField,
    },
    AdditiveGroup, CyclotomicMultSubgroup,
};
use ark_std::{cfg_chunks_mut, marker::PhantomData, vec::*};
use educe::Educe;
use itertools::Itertools;
use num_traits::One;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use super::fp12_lines::TwistType;

pub trait BnConfig: 'static + Sized {
    /// The absolute value of the BN curve parameter `X`
    /// (as in `q = 36 X^4 + 36 X^3 + 24 X^2 + 6 X + 1`).
    const X: &[u64];

    /// Whether or not `X` is negative.
    const X_IS_NEGATIVE: bool;

    /// The absolute value of `6X + 2`.
    const ATE_LOOP_COUNT: &[i8];

    const TWIST_TYPE: TwistType;
    const TWIST_MUL_BY_Q_X: Fp2<Self::Fp2Config>;
    const TWIST_MUL_BY_Q_Y: Fp2<Self::Fp2Config>;
    type Fp: PrimeField + Into<<Self::Fp as PrimeField>::BigInt>;
    type Fp2Config: Fp2Config<Fp = Self::Fp>;
    type Fp6Config: Fp6Config<Fp2Config = Self::Fp2Config>;
    type Fp12Config: Fp12Config<Fp6Config = Self::Fp6Config>;
    type G1Config: SWCurveConfig<BaseField = Self::Fp>;
    type G2Config: SWCurveConfig<
        BaseField = Fp2<Self::Fp2Config>,
        ScalarField = <Self::G1Config as CurveConfig>::ScalarField,
    >;

    /// Returns `3 * b' * c`, where `b'` is the G2 twist coefficient. Override
    /// when `b'` has structure that beats a general `Fp2` multiplication.
    fn mul_by_3b_twist(c: Fp2<Self::Fp2Config>) -> Fp2<Self::Fp2Config> {
        <Self::G2Config as SWCurveConfig>::COEFF_B * (c.double() + c)
    }

    /// Returns `f^x` for `f` in the cyclotomic subgroup. Override with a fixed addition
    /// chain for `Self::X`.
    fn exp_by_x(f: Fp12<Self::Fp12Config>) -> Fp12<Self::Fp12Config> {
        let mut f = f.cyclotomic_exp(Self::X);
        if Self::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }
        f
    }

    /// Constants for the 4-dimensional Galbraith-Scott split ([`gls4_digits`]) that GT
    /// exponentiation and G2 scalar multiplication share, or `None` to fall back to the
    /// windowed cyclotomic exponentiation.
    const GT_GLS: Option<GtGlsParams> = None;

    /// Returns `f^scalar` for `f` in GT. With [`Self::GT_GLS`] set, uses the
    /// Frobenius 4-dimensional GLS; otherwise the windowed cyclotomic exp.
    fn gt_exp(f: Fp12<Self::Fp12Config>, scalar: &[u64]) -> Fp12<Self::Fp12Config> {
        match Self::GT_GLS {
            Some(params) => gt_gls_exp::<Self>(f, scalar, &params),
            None => f.cyclotomic_exp(scalar),
        }
    }

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<G1Prepared<Self>>>,
        b: impl IntoIterator<Item = impl Into<G2Prepared<Self>>>,
    ) -> MillerLoopOutput<Bn<Self>> {
        let pairs = a
            .into_iter()
            .zip_eq(b)
            .filter_map(|(p, q)| {
                let (p, q) = (p.into(), q.into());
                match !p.is_zero() && !q.is_zero() {
                    true => Some((p, q)),
                    false => None,
                }
            })
            .collect::<Vec<_>>();

        // `(1/P.y, P.x/P.y)` for each pair whose `Q` has normalized lines
        // (`G2Prepared::normalize_lines`). A zero `P.y` keeps the pair on raw lines.
        let scales = lines::line_scales(pairs.iter().map(|(p, q)| {
            let normalized = q
                .ell_coeffs
                .first()
                .is_some_and(|c| lines::py_coeff::<Self::Fp12Config>(Self::TWIST_TYPE, c).is_one());
            (p.0.x, p.0.y, normalized)
        }));
        let mut pairs = pairs
            .into_iter()
            .zip(scales)
            .map(|((p, q), scale)| (p, q.ell_coeffs.into_iter(), scale))
            .collect::<Vec<_>>();

        // Amortize the shared squaring across all pairs: serial builds keep every
        // pair in one Miller loop (no duplicated squarings), parallel builds split
        // into chunks so the loops run concurrently.
        let chunk_size = if cfg!(feature = "parallel") {
            4
        } else {
            pairs.len().max(1)
        };

        let mut f = cfg_chunks_mut!(pairs, chunk_size)
            .map(|pairs| {
                let mut f = <Bn<Self> as Pairing>::TargetField::one();
                for i in (1..Self::ATE_LOOP_COUNT.len()).rev() {
                    // `f` starts at 1, so the first squaring is skipped (2010/526 section 4).
                    if i != Self::ATE_LOOP_COUNT.len() - 1 {
                        f.square_in_place();
                    }

                    let bit = Self::ATE_LOOP_COUNT[i - 1];
                    let has_add = bit == 1 || bit == -1;
                    // Pair every raw line in this iteration two at a time, across pairs
                    // and across doubling/addition.
                    let mut pending = None;
                    for (p, coeffs, scale) in pairs.iter_mut() {
                        for _ in 0..1 + usize::from(has_add) {
                            let c = coeffs.next().unwrap();
                            lines::feed_line(
                                Self::TWIST_TYPE,
                                &mut f,
                                &mut pending,
                                &c,
                                &p.0.x,
                                &p.0.y,
                                scale,
                            );
                        }
                    }
                    if let Some(line) = pending {
                        lines::mul_line(Self::TWIST_TYPE, &mut f, &line);
                    }
                }
                f
            })
            .product::<<Bn<Self> as Pairing>::TargetField>();

        if Self::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }

        // The two Frobenius steps contribute two lines per pair; pair them too.
        let mut pending = None;
        for (p, coeffs, scale) in &mut pairs {
            for c in coeffs.take(2) {
                lines::feed_line(
                    Self::TWIST_TYPE,
                    &mut f,
                    &mut pending,
                    &c,
                    &p.0.x,
                    &p.0.y,
                    scale,
                );
            }
        }
        if let Some(line) = pending {
            lines::mul_line(Self::TWIST_TYPE, &mut f, &line);
        }

        MillerLoopOutput(f)
    }

    fn final_exponentiation(f: MillerLoopOutput<Bn<Self>>) -> Option<PairingOutput<Bn<Self>>> {
        lines::final_exp_easy_part(f.0).map(|mut r| {
            // If the easy part already yields 1, the pairing is trivial.
            if r.is_one() {
                return PairingOutput(r);
            }

            // Hard part follows Laura Fuentes-Castaneda et al. "Faster hashing to G2"
            // (https://cacr.uwaterloo.ca/techreports/2011/cacr2011-26.pdf, section 4.1):
            // 3 exponentiations by x, 3 squarings and 10 multiplications, computing:
            //
            // result = elt^(q^3 * (12*z^3 + 6z^2 + 4z - 1) +
            //               q^2 * (12*z^3 + 6z^2 + 6z) +
            //               q   * (12*z^3 + 6z^2 + 4z) +
            //               1   * (12*z^3 + 12z^2 + 6z + 1))
            // which equals
            //
            // result = elt^( 2z * ( 6z^2 + 3z + 1 ) * (q^4 - q^2 + 1)/r ).

            let y0 = Bn::<Self>::exp_by_neg_x(r);
            let y1 = y0.cyclotomic_square();
            let y2 = y1.cyclotomic_square();
            let mut y3 = y2 * &y1;
            let y4 = Bn::<Self>::exp_by_neg_x(y3);
            let y5 = y4.cyclotomic_square();
            let mut y6 = Bn::<Self>::exp_by_neg_x(y5);
            y3.cyclotomic_inverse_in_place();
            y6.cyclotomic_inverse_in_place();
            let y7 = y6 * &y4;
            let mut y8 = y7 * &y3;
            let y9 = y8 * &y1;
            let y10 = y8 * &y4;
            let y11 = y10 * &r;
            let mut y12 = y9;
            y12.frobenius_map_in_place(1);
            let y13 = y12 * &y11;
            y8.frobenius_map_in_place(2);
            let y14 = y8 * &y13;
            r.cyclotomic_inverse_in_place();
            let mut y15 = r * &y9;
            y15.frobenius_map_in_place(3);
            let y16 = y15 * &y14;

            PairingOutput(y16)
        })
    }
}

pub mod g1;
pub mod g2;

pub use self::{
    g1::{G1Affine, G1Prepared, G1Projective},
    g2::{G2Affine, G2Prepared, G2Projective},
};

#[derive(Educe)]
#[educe(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Bn<P: BnConfig>(PhantomData<fn() -> P>);

impl<P: BnConfig> Bn<P> {
    fn exp_by_neg_x(f: Fp12<P::Fp12Config>) -> Fp12<P::Fp12Config> {
        let mut f = P::exp_by_x(f);
        f.cyclotomic_inverse_in_place();
        f
    }
}

impl<P: BnConfig> Pairing for Bn<P> {
    type BaseField = <P::G1Config as CurveConfig>::BaseField;
    type ScalarField = <P::G1Config as CurveConfig>::ScalarField;
    type G1 = G1Projective<P>;
    type G1Affine = G1Affine<P>;
    type G1Prepared = G1Prepared<P>;
    type G2 = G2Projective<P>;
    type G2Affine = G2Affine<P>;
    type G2Prepared = G2Prepared<P>;
    type TargetField = Fp12<P::Fp12Config>;

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> MillerLoopOutput<Self> {
        P::multi_miller_loop(a, b)
    }

    fn final_exponentiation(f: MillerLoopOutput<Self>) -> Option<PairingOutput<Self>> {
        P::final_exponentiation(f)
    }

    fn gt_exp(f: &Fp12<P::Fp12Config>, scalar: &[u64]) -> Fp12<P::Fp12Config> {
        P::gt_exp(*f, scalar)
    }

    fn is_in_gt(f: &Fp12<P::Fp12Config>) -> bool {
        // Scott, https://eprint.iacr.org/2021/1130. GT is the order-r subgroup of the
        // cyclotomic subgroup.
        if !lines::is_cyclotomic(f) {
            return false;
        }
        // Order-r subgroup, via Dai-Lin-Zhao-Zhou https://eprint.iacr.org/2022/348, as MIRACL
        // core `gtmember` (https://github.com/miracl/core/blob/a6df6733c1ad1ad0918306abd0c3983b4cd4a58c/rust/pair.rs#L1006-L1038):
        // one exponentiation f^x plus Frobenius, checking
        // f^(2 x p^3) == f^(1 + x + x p + x p^2). On GT the Frobenius is
        // [p mod r] = [6x^2], and 1 + x + x p + x p^2 == 2 x p^3 (mod r) there, so the
        // check is necessary. Dai et al. show it is sufficient on the cyclotomic
        // subgroup.
        let mut t = P::exp_by_x(*f); // f^x
        let mut r = t;
        r.frobenius_map_in_place(1); // f^(x p)
        t *= f; // f^x * f
        t *= &r; // * f^(x p)
        r.frobenius_map_in_place(1); // f^(x p^2)
        t *= &r; // * f^(x p^2)
        r.frobenius_map_in_place(1); // f^(x p^3)
        r.cyclotomic_square_in_place(); // f^(2 x p^3)
        r == t
    }
}

/// The 4x4 short lattice basis and first adjugate row for the GT 4-dimensional
/// GLS decomposition of a curve. With `lambda = p mod r`, the Frobenius eigenvalue on
/// GT (not `GLVConfig::LAMBDA`, the curve endomorphism eigenvalue on G1/G2), the
/// decompositions of 0 form the lattice
/// `L = { v : v0 + v1 lambda + v2 lambda^2 + v3 lambda^3 == 0 (mod r) }`, generated by
/// the rows `(r, 0, 0, 0)`, `(-lambda, 1, 0, 0)`, `(-lambda^2, 0, 1, 0)`,
/// `(-lambda^3, 0, 0, 1)` of determinant `r`. `basis` is an LLL reduction of these rows
/// (Sage `matrix(ZZ, rows).LLL()`, in any row order) with `det(basis) = r` and
/// entries about `x`, and `adj0` is the first row of `adj(basis)`, so
/// `basis^-1 = adj(basis) / r`. Basis entries fit `i128`. Each `adj0[j]` is
/// `(is_negative, magnitude)`, the sign convention of the digits of [`gls4_digits`], with the
/// magnitude in little-endian `u64` limbs below the scalar field modulus.
/// `adj0_div_r[j] = round(2^384 |adj0[j]| / r)` in five limbs, for a four-limb scalar field, so
/// [`gls4_digits`] divides by `r` with a shift. `scripts/bn_gls_decomp.py` checks `basis` and
/// `adj0` and prints `adj0_div_r`.
#[derive(Copy, Clone, Debug)]
pub struct GtGlsParams {
    pub basis: [[i128; 4]; 4],
    pub adj0: [(bool, [u64; 4]); 4],
    pub adj0_div_r: [[u64; 5]; 4],
}

/// Digits `k_i` with `k == \sum_i k_i p^i (mod r)`, each about `r^{1/4}`, by Babai rounding
/// against `params`. On BN curves `p mod r = 6x^2` is about twice as wide as `x`, so unlike
/// BLS12 (`p == x (mod r)` with a single-limb `x`) there are no base-`p` digits to read off,
/// and the digits come from the lattice of [`GtGlsParams`]. Babai rounding writes
/// `(k, 0, 0, 0) = beta * basis` with `beta_j = k * adj0[j] / r`, and the digits are
/// `(k, 0, 0, 0) - sum_j round(beta_j) basis_j`, still congruent to `k` because every basis
/// row lies in the lattice. Each rounding error is at most 1/2, so
/// `|k_i| <= (1/2) sum_j |basis[j][i]|`, which `params` must keep below `2^64` (BN254's is
/// `3x`, about `2^63.7`). The rounding is `round(k adj0_div_r[j] / 2^384)`, a multiplication and
/// a shift whose error against `k adj0[j] / r` is below `2^{-130}`, so the bound grows by that
/// much at most. The digits are that small, so they come out exactly from wrapping `i128`
/// arithmetic on the low 128 bits of `k` and of each `round(beta_j)`. On BN254 this takes
/// 0.07 us, against 2.67 us for the same rounding in `num-bigint`. The Frobenius on GT and
/// `psi` on G2 both act as `[p mod r]`, so the same digits serve [`Pairing::gt_exp`] and
/// [`crate::scalar_mul::gls`]. Galbraith-Scott, <https://eprint.iacr.org/2008/117>, section 3
/// (the lattice and Babai rounding); the rounded lattice decomposition is Gallant, Lambert,
/// Vanstone, CRYPTO 2001.
pub fn gls4_digits<P: BnConfig>(scalar: &[u64], params: &GtGlsParams) -> [(bool, u64); 4] {
    let k = scalar_mod_r::<ScalarField<P>>(scalar);
    let k = k.as_ref();
    debug_assert_eq!(
        k.len(),
        4,
        "`adj0_div_r` is sized for a four-limb scalar field"
    );
    let low_128 = |x: &[u64]| u128::from(x[0]) | (u128::from(x[1]) << 64);
    let mut digits = [low_128(k) as i128, 0, 0, 0];
    for j in 0..4 {
        let beta = mul_shift_round_bigint::<ScalarField<P>>(k, &params.adj0_div_r[j], k.len() + 2);
        let mut beta = low_128(beta.as_ref());
        if params.adj0[j].0 {
            beta = beta.wrapping_neg();
        }
        let beta = beta as i128;
        for (digit, b) in digits.iter_mut().zip(params.basis[j]) {
            *digit = digit.wrapping_sub(beta.wrapping_mul(b));
        }
    }
    digits.map(|d| {
        assert!(d.unsigned_abs() < 1 << 64, "GLS digit wider than 64 bits");
        (d < 0, d.unsigned_abs() as u64)
    })
}

type ScalarField<P> = <<P as BnConfig>::G1Config as CurveConfig>::ScalarField;

/// `k mod r` as canonical limbs, for `k` in little-endian limbs of any length.
fn scalar_mod_r<F: PrimeField>(k: &[u64]) -> F::BigInt {
    let len = k.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
    let mut repr = F::BigInt::default();
    if len <= repr.as_ref().len() {
        repr.as_mut()[..len].copy_from_slice(&k[..len]);
        if repr < F::MODULUS {
            return repr;
        }
    }
    let two_64 = F::from(1u128 << 64);
    k[..len]
        .iter()
        .rev()
        .fold(F::zero(), |acc, &l| acc * two_64 + F::from(l))
        .into_bigint()
}

/// `f^k` for `f` in GT via the Frobenius 4-dimensional GLS: the digits of [`gls4_digits`], then
/// `prod_i (f^(p^i))^{k_i}` with [`crate::pairing::gt_multiexp`]. On GT the Frobenius `pi` acts
/// as `[p mod r]` and `pi^4 - pi^2 + 1 = 0`, since `r` divides `Phi_12(p) = p^4 - p^2 + 1`, so
/// four digits suffice. About 64 cyclotomic squarings instead of about 254. Galbraith-Scott,
/// <https://eprint.iacr.org/2008/117>, section 4 (the Frobenius on GT). MIRACL core
/// [`gtpow`](https://github.com/miracl/core/blob/a6df6733c1ad1ad0918306abd0c3983b4cd4a58c/rust/pair.rs#L875-L909) does the same with its own basis.
fn gt_gls_exp<P: BnConfig>(
    f: Fp12<P::Fp12Config>,
    scalar: &[u64],
    params: &GtGlsParams,
) -> Fp12<P::Fp12Config> {
    let digits = gls4_digits::<P>(scalar, params);
    let mut g = [f; 4];
    for i in 1..4 {
        g[i] = g[i - 1];
        g[i].frobenius_map_in_place(1);
    }
    crate::pairing::gt_multiexp(g, digits)
}
