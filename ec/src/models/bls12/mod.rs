use crate::{
    models::{short_weierstrass::SWCurveConfig, CurveConfig},
    pairing::{MillerLoopOutput, Pairing, PairingOutput},
    AffineRepr,
};
use ark_ff::{
    fields::{
        fp12_2over3over2::{Fp12, Fp12Config},
        fp2::Fp2Config,
        fp6_3over2::Fp6Config,
        Fp2,
    },
    AdditiveGroup, BitIteratorBE, CyclotomicMultSubgroup, Field, PrimeField,
};
use ark_std::{cfg_chunks_mut, marker::PhantomData, vec::*};
use educe::Educe;
use num_traits::{One, Zero};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A particular BLS12 group can have G2 being either a multiplicative or a
/// divisive twist.
pub enum TwistType {
    M,
    D,
}

pub trait Bls12Config: 'static + Sized {
    /// Parameterizes the BLS12 family.
    const X: &[u64];
    /// Is `Self::X` negative?
    const X_IS_NEGATIVE: bool;
    /// What kind of twist is this?
    const TWIST_TYPE: TwistType;

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

    /// Returns `f^x`. Override with a fixed addition chain for `Self::X`.
    fn exp_by_x(f: Fp12<Self::Fp12Config>) -> Fp12<Self::Fp12Config> {
        let mut res = f.cyclotomic_exp(Self::X);
        if Self::X_IS_NEGATIVE {
            res.cyclotomic_inverse_in_place();
        }
        res
    }

    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<G1Prepared<Self>>>,
        b: impl IntoIterator<Item = impl Into<G2Prepared<Self>>>,
    ) -> MillerLoopOutput<Bls12<Self>> {
        use itertools::Itertools;

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
        // (`G2Prepared::normalize_lines`), from one batched inversion. A zero `P.y` keeps the
        // pair on raw lines.
        let mut yinv: Vec<Self::Fp> = pairs
            .iter()
            .map(|(p, q)| match q.ell_coeffs.first() {
                Some(c) if g2::py_coeff::<Self>(c).is_one() => p.0.xy().unwrap().1,
                _ => Self::Fp::zero(),
            })
            .collect();
        if yinv.iter().any(|y| !y.is_zero()) {
            ark_ff::batch_inversion(&mut yinv);
        }
        let mut pairs = pairs
            .into_iter()
            .zip(yinv)
            .map(|((p, q), yinv)| {
                let scale = (!yinv.is_zero()).then(|| (yinv, p.0.xy().unwrap().0 * yinv));
                (p, q.ell_coeffs.into_iter(), scale)
            })
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
                let mut f = <Bls12<Self> as Pairing>::TargetField::one();
                // `f` starts at 1, so the first squaring is skipped (2010/526 section 4).
                let mut first = true;
                for i in BitIteratorBE::without_leading_zeros(Self::X).skip(1) {
                    if !first {
                        f.square_in_place();
                    }
                    first = false;
                    // Pair every raw line in this iteration two at a time, across pairs
                    // and across doubling/addition, halving the sparse-by-full mults.
                    let mut pending = None;
                    for (p, coeffs, scale) in pairs.iter_mut() {
                        Bls12::<Self>::feed_line(
                            &mut f,
                            &mut pending,
                            &coeffs.next().unwrap(),
                            &p.0,
                            scale,
                        );
                        if i {
                            Bls12::<Self>::feed_line(
                                &mut f,
                                &mut pending,
                                &coeffs.next().unwrap(),
                                &p.0,
                                scale,
                            );
                        }
                    }
                    if let Some(line) = pending {
                        Bls12::<Self>::mul_line(&mut f, &line);
                    }
                }
                f
            })
            .product::<<Bls12<Self> as Pairing>::TargetField>();

        if Self::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }
        MillerLoopOutput(f)
    }

    fn final_exponentiation(
        f: MillerLoopOutput<Bls12<Self>>,
    ) -> Option<PairingOutput<Bls12<Self>>> {
        // Computing the final exponentiation following
        // https://eprint.iacr.org/2020/875
        // Adapted from the implementation in https://github.com/ConsenSys/gurvy/pull/29

        // f1 = r.cyclotomic_inverse_in_place() = f^(p^6)
        let f = f.0;
        let mut f1 = f;
        f1.cyclotomic_inverse_in_place();

        f.inverse().map(|mut f2| {
            // f2 = f^(-1);
            // r = f^(p^6 - 1)
            let mut r = f1 * &f2;

            // f2 = f^(p^6 - 1)
            f2 = r;
            // r = f^((p^6 - 1)(p^2))
            r.frobenius_map_in_place(2);

            // r = f^((p^6 - 1)(p^2) + (p^6 - 1))
            // r = f^((p^6 - 1)(p^2 + 1))
            r *= &f2;

            // If the easy part already yields 1, the pairing is trivial.
            if r.is_one() {
                return PairingOutput(r);
            }

            // Hard part of the final exponentiation:
            // t[0].CyclotomicSquare(&result)
            let mut y0 = r.cyclotomic_square();
            // t[1].Expt(&result)
            let mut y1 = Self::exp_by_x(r);
            // t[2].InverseUnitary(&result)
            let mut y2 = r;
            y2.cyclotomic_inverse_in_place();
            // t[1].Mul(&t[1], &t[2])
            y1 *= &y2;
            // t[2].Expt(&t[1])
            y2 = Self::exp_by_x(y1);
            // t[1].InverseUnitary(&t[1])
            y1.cyclotomic_inverse_in_place();
            // t[1].Mul(&t[1], &t[2])
            y1 *= &y2;
            // t[2].Expt(&t[1])
            y2 = Self::exp_by_x(y1);
            // t[1].Frobenius(&t[1])
            y1.frobenius_map_in_place(1);
            // t[1].Mul(&t[1], &t[2])
            y1 *= &y2;
            // result.Mul(&result, &t[0])
            r *= &y0;
            // t[0].Expt(&t[1])
            y0 = Self::exp_by_x(y1);
            // t[2].Expt(&t[0])
            y2 = Self::exp_by_x(y0);
            // t[0].FrobeniusSquare(&t[1])
            y0 = y1;
            y0.frobenius_map_in_place(2);
            // t[1].InverseUnitary(&t[1])
            y1.cyclotomic_inverse_in_place();
            // t[1].Mul(&t[1], &t[2])
            y1 *= &y2;
            // t[1].Mul(&t[1], &t[0])
            y1 *= &y0;
            // result.Mul(&result, &t[1])
            r *= &y1;
            PairingOutput(r)
        })
    }
}

/// Digits `k_i` with `k = \sum_i k_i x^i`, from the base-`|x|` digits of `k`, negating the odd
/// ones when `x < 0`. `None` when `X` spans more than one limb or `k >= |x|^4`. With `p == x
/// (mod r)`, these are the four-dimensional GLS digits for the Frobenius on GT
/// ([`Pairing::gt_exp`]) and for `psi` on G2 ([`crate::scalar_mul::gls`]).
pub fn gls4_digits<P: Bls12Config>(k: &[u64]) -> Option<[(bool, u64); 4]> {
    let [x] = P::X else {
        return None;
    };
    let x = u128::from(*x);
    let mut rem = k.to_vec();
    let mut digits = [(false, 0u64); 4];
    for (i, d) in digits.iter_mut().enumerate() {
        let mut carry = 0u128;
        for limb in rem.iter_mut().rev() {
            let cur = (carry << 64) | u128::from(*limb);
            *limb = (cur / x) as u64;
            carry = cur % x;
        }
        *d = (P::X_IS_NEGATIVE && i % 2 == 1, carry as u64);
    }
    rem.iter().all(|&l| l == 0).then_some(digits)
}

pub mod g1;
pub mod g2;

pub use self::{
    g1::{G1Affine, G1Prepared, G1Projective},
    g2::{G2Affine, G2Prepared, G2PreparedFixed, G2Projective},
};

#[derive(Educe)]
#[educe(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Bls12<P: Bls12Config>(PhantomData<fn() -> P>);

/// A line, scaled by a G1 point, in the three coefficient slots its sparse
/// multiplication reads.
type EllLine<P> = (
    Fp2<<P as Bls12Config>::Fp2Config>,
    Fp2<<P as Bls12Config>::Fp2Config>,
    Fp2<<P as Bls12Config>::Fp2Config>,
);

impl<P: Bls12Config> Bls12<P> {
    // Scale a raw line by `p`, positioning the coefficients at their sparse slots
    // (0,1,4 for an M-twist, 0,3,4 for a D-twist).
    fn scale_line(coeffs: &g2::EllCoeff<P>, p: &G1Affine<P>) -> EllLine<P> {
        let (px, py) = p.xy().unwrap();
        let (mut c0, mut c1, mut c2) = (coeffs.0, coeffs.1, coeffs.2);
        match P::TWIST_TYPE {
            TwistType::M => {
                c2.mul_assign_by_fp(&py);
                c1.mul_assign_by_fp(&px);
            },
            TwistType::D => {
                c0.mul_assign_by_fp(&py);
                c1.mul_assign_by_fp(&px);
            },
        }
        (c0, c1, c2)
    }

    // Multiply `f` by one scaled line.
    fn mul_line(f: &mut Fp12<P::Fp12Config>, a: &EllLine<P>) {
        match P::TWIST_TYPE {
            TwistType::M => f.mul_by_014(&a.0, &a.1, &a.2),
            TwistType::D => f.mul_by_034(&a.0, &a.1, &a.2),
        }
    }

    // Multiply `f` by two scaled lines, which may come from the same pair (doubling
    // and addition) or from different pairs. `Fp12::mul_by_014_pair` /
    // `Fp12::mul_by_034_pair` pick between two sparse products and one line-by-line
    // product followed by a semi-sparse one. Pairing lines across pairs follows MIRACL core
    // `ate2` (https://github.com/miracl/core/blob/a6df6733c1ad1ad0918306abd0c3983b4cd4a58c/rust/pair.rs#L522-L541).
    fn mul_line_pair(f: &mut Fp12<P::Fp12Config>, a: &EllLine<P>, b: &EllLine<P>) {
        match P::TWIST_TYPE {
            TwistType::M => f.mul_by_014_pair(&a.0, &a.1, &a.2, &b.0, &b.1, &b.2),
            TwistType::D => f.mul_by_034_pair(&a.0, &a.1, &a.2, &b.0, &b.1, &b.2),
        }
    }

    // Feed a scaled line into `f`, pairing it with a held-back line when one waits.
    fn push_line(f: &mut Fp12<P::Fp12Config>, pending: &mut Option<EllLine<P>>, line: EllLine<P>) {
        match pending.take() {
            Some(prev) => Self::mul_line_pair(f, &prev, &line),
            None => *pending = Some(line),
        }
    }

    // Multiply `f` by one line at `p`. A normalized line, whose `P.y` coefficient is 1, takes the
    // fixed-Q product when `scale = (1/P.y, P.x/P.y)` is known. Any other line goes through
    // `push_line`.
    fn feed_line(
        f: &mut Fp12<P::Fp12Config>,
        pending: &mut Option<EllLine<P>>,
        coeffs: &g2::EllCoeff<P>,
        p: &G1Affine<P>,
        scale: &Option<(P::Fp, P::Fp)>,
    ) {
        match scale {
            Some((yinv, pxyinv)) if g2::py_coeff::<P>(coeffs).is_one() => {
                Self::mul_fixed_line(f, &g2::fixed_line::<P>(coeffs), yinv, pxyinv)
            },
            _ => Self::push_line(f, pending, Self::scale_line(coeffs, p)),
        }
    }

    // Multiply `f` by one fixed-Q two-coefficient line reconstructed at `p`, whose
    // third slot is 1 after the per-`p` rescale by `yinv`/`pxyinv`.
    fn mul_fixed_line(
        f: &mut Fp12<P::Fp12Config>,
        line: &(Fp2<P::Fp2Config>, Fp2<P::Fp2Config>),
        yinv: &P::Fp,
        pxyinv: &P::Fp,
    ) {
        match P::TWIST_TYPE {
            TwistType::M => {
                let mut s0 = line.0;
                s0.mul_assign_by_fp(yinv);
                let mut s1 = line.1;
                s1.mul_assign_by_fp(pxyinv);
                f.mul_by_014_c4_one(&s0, &s1);
            },
            TwistType::D => {
                let mut s3 = line.0;
                s3.mul_assign_by_fp(pxyinv);
                let mut s4 = line.1;
                s4.mul_assign_by_fp(yinv);
                f.mul_by_034_c0_one(&s3, &s4);
            },
        }
    }

    /// Miller loop with each `Q` fixed and preprocessed to two-coefficient lines
    /// ([`G2PreparedFixed`]). Panics if `a` and `b` differ in length.
    pub fn multi_miller_loop_fixed(
        a: impl IntoIterator<Item = impl Into<G1Affine<P>>>,
        b: &[G2PreparedFixed<P>],
    ) -> MillerLoopOutput<Self> {
        use itertools::Itertools;

        let mut g1s = Vec::new();
        let mut lines = Vec::new();
        for (p, prep) in a.into_iter().zip_eq(b) {
            let p = p.into();
            if p.is_zero() || prep.infinity {
                continue;
            }
            g1s.push(p);
            lines.push(prep.lines.iter());
        }
        if g1s.is_empty() {
            return MillerLoopOutput(Fp12::one());
        }

        // yinv = 1/P.y, pxyinv = P.x/P.y, batched over the G1 points.
        let mut yinv: Vec<P::Fp> = g1s.iter().map(|p| p.xy().unwrap().1).collect();
        ark_ff::batch_inversion(&mut yinv);
        let pxyinv: Vec<P::Fp> = g1s
            .iter()
            .zip(&yinv)
            .map(|(p, yi)| p.xy().unwrap().0 * yi)
            .collect();

        let mut f = Fp12::<P::Fp12Config>::one();
        let mut first = true;
        for i in BitIteratorBE::without_leading_zeros(P::X).skip(1) {
            if !first {
                f.square_in_place();
            }
            first = false;
            for (idx, it) in lines.iter_mut().enumerate() {
                Self::mul_fixed_line(&mut f, it.next().unwrap(), &yinv[idx], &pxyinv[idx]);
                if i {
                    Self::mul_fixed_line(&mut f, it.next().unwrap(), &yinv[idx], &pxyinv[idx]);
                }
            }
        }
        if P::X_IS_NEGATIVE {
            f.cyclotomic_inverse_in_place();
        }
        MillerLoopOutput(f)
    }
}

impl<P: Bls12Config> Pairing for Bls12<P> {
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
        // 4-dimensional Galbraith-Scott on GT (https://eprint.iacr.org/2008/117,
        // section 4), as MIRACL core `gtpow` (https://github.com/miracl/core/blob/a6df6733c1ad1ad0918306abd0c3983b4cd4a58c/rust/pair.rs#L875-L909). The Frobenius pi(f) = f^p acts as [p mod r] on GT and satisfies
        // pi^4 - pi^2 + 1 = 0 there, since r divides Phi_12(p) = p^4 - p^2 + 1, so
        // f^k = prod_i (f^(p^i))^{k_i} for any k = sum_i k_i p^i (mod r) with four
        // digits. For BLS12, p == x (mod r) and |x| is a single limb, so the digits are
        // the base-|x| digits of k, from repeated division by a u64 ([`gls4_digits`]). The four
        // ~64-bit digits need about 64 cyclotomic squarings instead of about 255.
        let Some(signed) = gls4_digits::<P>(scalar) else {
            return f.cyclotomic_exp(scalar);
        };

        // g[i] = f^(p^i).
        let mut g = [*f; 4];
        for i in 1..4 {
            g[i] = g[i - 1];
            g[i].frobenius_map_in_place(1);
        }
        crate::pairing::gt_multiexp(g, signed)
    }

    fn is_in_gt(f: &Fp12<P::Fp12Config>) -> bool {
        // Scott, https://eprint.iacr.org/2021/1130. `f` is in GT iff it is in the
        // cyclotomic subgroup, of order Phi_12(p) = p^4 - p^2 + 1, and the order-r
        // subgroup within it. On GT the Frobenius is [p mod r] = [x], so f^p == f^x is
        // necessary, and Scott shows it is sufficient on the cyclotomic subgroup of a
        // BLS12 curve. This costs one exponentiation by x instead of one by r.
        if f.is_zero() {
            return false;
        }
        // Cyclotomic: f^(p^2) == f^(p^4) * f, i.e. f^(p^4 - p^2 + 1) == 1.
        let mut a = *f;
        a.frobenius_map_in_place(2);
        let mut b = a;
        b.frobenius_map_in_place(2);
        b *= f;
        if a != b {
            return false;
        }
        // Order-r subgroup: f^p == f^x, since p == x (mod r).
        let mut fp = *f;
        fp.frobenius_map_in_place(1);
        fp == P::exp_by_x(*f)
    }
}
