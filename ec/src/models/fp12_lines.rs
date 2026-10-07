//! Line arithmetic shared by the BN and BLS12 Miller loops. A G2 line has three `Fp2`
//! coefficients at the sparse slots 0, 1, 4 of `Fp12` for an M-twist and 0, 3, 4 for a D-twist,
//! one of which scales by `P.y`.

use ark_ff::{
    fields::{
        fp12_2over3over2::{Fp12, Fp12Config},
        fp2::Fp2Config,
        fp6_3over2::Fp6Config,
        Field, Fp2,
    },
    AdditiveGroup, CyclotomicMultSubgroup,
};
use ark_std::vec::Vec;
use num_traits::{One, Zero};

/// A G2 twist is either multiplicative or divisive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwistType {
    M,
    D,
}

pub(crate) type Fp2Of<C> = Fp2<<<C as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config>;
pub(crate) type FpOf<C> = <<<C as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config as Fp2Config>::Fp;
pub(crate) type Line<C> = (Fp2Of<C>, Fp2Of<C>, Fp2Of<C>);

/// The coefficient `ell` scales by `P.y`: index 2 for an M-twist, index 0 for a D-twist.
#[inline(always)]
pub(crate) const fn py_coeff<C: Fp12Config>(twist: TwistType, c: &Line<C>) -> &Fp2Of<C> {
    match twist {
        TwistType::M => &c.2,
        TwistType::D => &c.0,
    }
}

/// The two coefficients of a normalized line other than its unit `P.y` coefficient, in slot
/// order.
#[inline(always)]
pub(crate) const fn fixed_line<C: Fp12Config>(
    twist: TwistType,
    c: &Line<C>,
) -> (Fp2Of<C>, Fp2Of<C>) {
    match twist {
        TwistType::M => (c.0, c.1),
        TwistType::D => (c.1, c.2),
    }
}

/// Divides every line by its `P.y` coefficient, which leaves that coefficient 1. A line whose
/// `P.y` coefficient is zero stays raw. One batched `Fp2` inversion and two `Fp2` products per
/// line.
pub(crate) fn normalize_lines<C: Fp12Config>(twist: TwistType, lines: &mut [Line<C>]) {
    let mut inv: Vec<Fp2Of<C>> = lines.iter().map(|c| *py_coeff::<C>(twist, c)).collect();
    ark_ff::batch_inversion(&mut inv);
    for (c, inv) in lines.iter_mut().zip(inv) {
        if inv.is_zero() {
            continue;
        }
        match twist {
            TwistType::M => {
                c.0 *= inv;
                c.1 *= inv;
                c.2 = Fp2::one();
            },
            TwistType::D => {
                c.0 = Fp2::one();
                c.1 *= inv;
                c.2 *= inv;
            },
        }
    }
}

/// The lines of `-Q` from those of `Q`: a raw line negates its `P.y` coefficient, a normalized
/// line its other two.
pub(crate) fn negate_lines<C: Fp12Config>(twist: TwistType, lines: &mut [Line<C>]) {
    for c in lines {
        match (twist, py_coeff::<C>(twist, c).is_one()) {
            (TwistType::M, false) => {
                c.2.neg_in_place();
            },
            (TwistType::D, false) => {
                c.0.neg_in_place();
            },
            (TwistType::M, true) => {
                c.0.neg_in_place();
                c.1.neg_in_place();
            },
            (TwistType::D, true) => {
                c.1.neg_in_place();
                c.2.neg_in_place();
            },
        }
    }
}

/// `(1/P.y, P.x/P.y)` for each `(P.x, P.y, normalized)` whose `Q` has normalized lines, from one
/// batched inversion, and `None` for the others and for `P.y = 0`.
pub(crate) fn line_scales<F: Field>(
    points: impl Iterator<Item = (F, F, bool)>,
) -> Vec<Option<(F, F)>> {
    let points: Vec<_> = points.collect();
    let mut yinv: Vec<F> = points
        .iter()
        .map(|&(_, y, normalized)| if normalized { y } else { F::zero() })
        .collect();
    if yinv.iter().any(|y| !y.is_zero()) {
        ark_ff::batch_inversion(&mut yinv);
    }
    points
        .iter()
        .zip(yinv)
        .map(|(&(x, _, _), yinv)| (!yinv.is_zero()).then(|| (yinv, x * yinv)))
        .collect()
}

/// A raw line scaled by `P`, its coefficients at their sparse slots.
#[inline(always)]
pub(crate) fn scale_line<C: Fp12Config>(
    twist: TwistType,
    c: &Line<C>,
    px: &FpOf<C>,
    py: &FpOf<C>,
) -> Line<C> {
    let (mut c0, mut c1, mut c2) = *c;
    c1.mul_assign_by_fp(px);
    match twist {
        TwistType::M => c2.mul_assign_by_fp(py),
        TwistType::D => c0.mul_assign_by_fp(py),
    }
    (c0, c1, c2)
}

/// Multiplies `f` by one scaled line.
#[inline(always)]
pub(crate) fn mul_line<C: Fp12Config>(twist: TwistType, f: &mut Fp12<C>, a: &Line<C>) {
    match twist {
        TwistType::M => f.mul_by_014(&a.0, &a.1, &a.2),
        TwistType::D => f.mul_by_034(&a.0, &a.1, &a.2),
    }
}

/// Multiplies `f` by two scaled lines. `Fp12::mul_by_014_pair` / `Fp12::mul_by_034_pair` pick
/// between two sparse products and one line-by-line product followed by a semi-sparse one.
/// Pairing lines across pairs follows MIRACL core
/// [`ate2`](https://github.com/miracl/core/blob/a6df6733c1ad1ad0918306abd0c3983b4cd4a58c/rust/pair.rs#L522-L541).
#[inline(always)]
fn mul_line_pair<C: Fp12Config>(twist: TwistType, f: &mut Fp12<C>, a: &Line<C>, b: &Line<C>) {
    match twist {
        TwistType::M => f.mul_by_014_pair(&a.0, &a.1, &a.2, &b.0, &b.1, &b.2),
        TwistType::D => f.mul_by_034_pair(&a.0, &a.1, &a.2, &b.0, &b.1, &b.2),
    }
}

/// Multiplies `f` by one normalized line reconstructed at `P` from
/// `scale = (1/P.y, P.x/P.y)`, its `P.y` slot then 1.
#[inline(always)]
pub(crate) fn mul_fixed_line<C: Fp12Config>(
    twist: TwistType,
    f: &mut Fp12<C>,
    line: &(Fp2Of<C>, Fp2Of<C>),
    yinv: &FpOf<C>,
    pxyinv: &FpOf<C>,
) {
    let (mut a, mut b) = *line;
    match twist {
        TwistType::M => {
            a.mul_assign_by_fp(yinv);
            b.mul_assign_by_fp(pxyinv);
            f.mul_by_014_c4_one(&a, &b);
        },
        TwistType::D => {
            a.mul_assign_by_fp(pxyinv);
            b.mul_assign_by_fp(yinv);
            f.mul_by_034_c0_one(&a, &b);
        },
    }
}

/// Multiplies `f` by one line at `P`. A normalized line takes [`mul_fixed_line`] when `scale` is
/// known. Any other line is scaled and paired with a held-back line in `pending`, or held back.
#[inline(always)]
pub(crate) fn feed_line<C: Fp12Config>(
    twist: TwistType,
    f: &mut Fp12<C>,
    pending: &mut Option<Line<C>>,
    coeffs: &Line<C>,
    px: &FpOf<C>,
    py: &FpOf<C>,
    scale: &Option<(FpOf<C>, FpOf<C>)>,
) {
    match scale {
        Some((yinv, pxyinv)) if py_coeff::<C>(twist, coeffs).is_one() => {
            mul_fixed_line(twist, f, &fixed_line::<C>(twist, coeffs), yinv, pxyinv)
        },
        _ => {
            let line = scale_line::<C>(twist, coeffs, px, py);
            match pending.take() {
                Some(prev) => mul_line_pair(twist, f, &prev, &line),
                None => *pending = Some(line),
            }
        },
    }
}

/// The easy part of the final exponentiation, `f^((p^6 - 1)(p^2 + 1))`, or `None` for `f = 0`.
/// Beuchat et al., <https://eprint.iacr.org/2010/354>, page 9.
pub(crate) fn final_exp_easy_part<C: Fp12Config>(f: Fp12<C>) -> Option<Fp12<C>> {
    let mut f1 = f;
    f1.cyclotomic_inverse_in_place(); // f^(p^6)
    f.inverse().map(|f2| {
        let mut r = f1 * &f2; // f^(p^6 - 1)
        let f2 = r;
        r.frobenius_map_in_place(2);
        r *= &f2; // f^((p^6 - 1)(p^2 + 1))
        r
    })
}

/// Whether nonzero `f` is in the cyclotomic subgroup, of order `\Phi_12(p) = p^4 - p^2 + 1`:
/// `f^(p^2) == f^(p^4) * f`.
pub(crate) fn is_cyclotomic<C: Fp12Config>(f: &Fp12<C>) -> bool {
    if f.is_zero() {
        return false;
    }
    let mut a = *f;
    a.frobenius_map_in_place(2);
    let mut b = a;
    b.frobenius_map_in_place(2);
    b *= f;
    a == b
}
