use ark_ff::{
    fields::{Field, Fp2},
    AdditiveGroup,
};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{io::Read, ops::Neg, vec::*};
use educe::Educe;
use num_traits::{One, Zero};

use crate::{
    bn::{BnConfig, TwistType},
    pairing::g2_doubling_y,
    short_weierstrass::{Affine, Projective},
    AffineRepr, CurveGroup,
};

pub type G2Affine<P> = Affine<<P as BnConfig>::G2Config>;
pub type G2Projective<P> = Projective<<P as BnConfig>::G2Config>;

/// `Q` preprocessed into the line functions of the optimal ate Miller loop, three `Fp2`
/// coefficients per line, from the homogeneous projective doubling and mixed addition of
/// Aranha, Barreto, Longa, Ricardini, [The Realm of the Pairings](https://eprint.iacr.org/2013/722),
/// section 4.3. A line is only defined up to a factor in `Fp2`. The final exponentiation
/// `(p^12 - 1) / r` is a multiple of `p^6 - 1`, hence of `p^2 - 1`, so it maps every nonzero
/// element of `Fp2` to 1. The doubling keeps its point scaled by 4 and
/// [`G2Prepared::normalize_lines`] rescales each line on this basis, so raw `MillerLoopOutput`
/// values depend on the implementation and only final-exponentiated values are canonical.
#[derive(Educe, CanonicalSerialize)]
#[educe(Clone, Debug, PartialEq, Eq)]
pub struct G2Prepared<P: BnConfig> {
    /// Stores the coefficients of the line evaluations as calculated in
    /// <https://eprint.iacr.org/2013/722.pdf>
    pub ell_coeffs: Vec<EllCoeff<P>>,
    pub infinity: bool,
}

impl<P: BnConfig> Valid for G2Prepared<P> {
    /// The line count the Miller loop consumes, none at infinity.
    fn check(&self) -> Result<(), SerializationError> {
        let expected = if self.infinity { 0 } else { num_ell_coeffs::<P>() };
        if self.ell_coeffs.len() != expected {
            return Err(SerializationError::InvalidData);
        }
        self.ell_coeffs.check()?;
        Ok(())
    }
}

impl<P: BnConfig> CanonicalDeserialize for G2Prepared<P> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let ell_coeffs = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let infinity = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let prepared = Self { ell_coeffs, infinity };
        if validate == Validate::Yes {
            prepared.check()?;
        }
        Ok(prepared)
    }
}

pub type EllCoeff<P> = (
    Fp2<<P as BnConfig>::Fp2Config>,
    Fp2<<P as BnConfig>::Fp2Config>,
    Fp2<<P as BnConfig>::Fp2Config>,
);

#[derive(Educe)]
#[educe(Clone, Copy, Debug)]
pub struct G2HomProjective<P: BnConfig> {
    x: Fp2<P::Fp2Config>,
    y: Fp2<P::Fp2Config>,
    z: Fp2<P::Fp2Config>,
}

impl<P: BnConfig> G2HomProjective<P> {
    pub fn double_in_place(&mut self) -> EllCoeff<P> {
        // Homogeneous projective doubling with its tangent line (Costello, Lange, Naehrig,
        // https://eprint.iacr.org/2009/615, section 5), in the form of Aranha, Karabina,
        // Longa, Gebotys, Lopez, https://eprint.iacr.org/2010/526, section 4, execution (3).
        // With A = XY/2, B = Y^2, C = Z^2, E = 3b'C, F = 3E, G = (B + F)/2 and
        // H = (Y + Z)^2 - B - C, it gives 2T = (A(B - F), G^2 - 3E^2, BH) and the line
        // (E - B, 3X^2, -H). Here 2T is kept scaled by 4, which drops the two halvings and
        // the per-preparation inverse of 2. A projective point is unchanged by a nonzero
        // scalar, and the line coefficients are homogeneous of degree 2 in (X, Y, Z), so
        // every later line picks up an `Fp` factor, which the final exponentiation removes.
        let a = self.x * &self.y; // X*Y
        let b = self.y.square(); // Y^2
        let c = self.z.square(); // Z^2
        let e = P::mul_by_3b_twist(c); // 3*b'*Z^2
        let f = e.double() + &e; // 9*b'*Z^2
        let h = (self.y + &self.z).square() - &(b + &c); // 2*Y*Z
        let i = e - &b;
        let j = self.x.square();
        let g = b + &f; // Y^2 + 9*b'*Z^2
        self.x = (a * &(b - &f)).double(); // 2*X*Y*(Y^2 - 9*b'*Z^2)
        self.y = g2_doubling_y(&g, &e); // (Y^2 + 9*b'*Z^2)^2 - 12*(3*b'*Z^2)^2
        self.z = (b * &h).double().double(); // 4*Y^2*(2*Y*Z)
        match P::TWIST_TYPE {
            TwistType::M => (i, j.double() + &j, -h),
            TwistType::D => (-h, j.double() + &j, i),
        }
    }

    pub fn add_in_place(&mut self, q: &G2Affine<P>) -> EllCoeff<P> {
        // Mixed addition T + Q with its chord line, 2013/722 section 4.3 (the formulas after
        // equation (13)): theta = Y - qy Z, lambda = X - qx Z, and the line
        // (theta qx - lambda qy, -theta, lambda). The coefficients are homogeneous of degree 1
        // in (X, Y, Z), so the scaled point of `double_in_place` again costs an `Fp` factor.
        let theta = self.y - &(q.y * &self.z);
        let lambda = self.x - &(q.x * &self.z);
        let c = theta.square();
        let d = lambda.square();
        let e = lambda * &d;
        let f = self.z * &c;
        let g = self.x * &d;
        let h = e + &f - &g.double();
        self.x = lambda * &h;
        self.y = theta * &(g - &h) - &(e * &self.y);
        self.z *= &e;
        let j = theta * &q.x - &(lambda * &q.y);

        match P::TWIST_TYPE {
            TwistType::M => (j, -theta, lambda),
            TwistType::D => (lambda, -theta, j),
        }
    }

    /// The chord line of [`Self::add_in_place`] without the point update, for the second
    /// Frobenius addition, whose sum is never read. After gnark-crypto
    /// [`lineCompute`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/pairing.go#L368-L386).
    fn chord_line(&self, q: &G2Affine<P>) -> EllCoeff<P> {
        let theta = self.y - &(q.y * &self.z);
        let lambda = self.x - &(q.x * &self.z);
        let j = theta * &q.x - &(lambda * &q.y);
        match P::TWIST_TYPE {
            TwistType::M => (j, -theta, lambda),
            TwistType::D => (lambda, -theta, j),
        }
    }
}

impl<P: BnConfig> Default for G2Prepared<P> {
    fn default() -> Self {
        Self::from(G2Affine::<P>::generator())
    }
}

impl<P: BnConfig> From<G2Affine<P>> for G2Prepared<P> {
    fn from(q: G2Affine<P>) -> Self {
        if q.is_zero() {
            Self {
                ell_coeffs: Vec::new(),
                infinity: true,
            }
        } else {
            let mut ell_coeffs = Vec::with_capacity(num_ell_coeffs::<P>());
            let mut r = G2HomProjective::<P> {
                x: q.x,
                y: q.y,
                z: Fp2::one(),
            };

            let neg_q = -q;

            for bit in P::ATE_LOOP_COUNT.iter().rev().skip(1) {
                ell_coeffs.push(r.double_in_place());

                match bit {
                    1 => ell_coeffs.push(r.add_in_place(&q)),
                    -1 => ell_coeffs.push(r.add_in_place(&neg_q)),
                    _ => {},
                }
            }

            let q1 = mul_by_char::<P>(q);
            let mut q2 = mul_by_char::<P>(q1);

            if P::X_IS_NEGATIVE {
                r.y = -r.y;
            }

            q2.y = -q2.y;

            ell_coeffs.push(r.add_in_place(&q1));
            ell_coeffs.push(r.chord_line(&q2));

            debug_assert_eq!(ell_coeffs.len(), num_ell_coeffs::<P>());
            Self {
                ell_coeffs,
                infinity: false,
            }
        }
    }
}

impl<P: BnConfig> From<G2Projective<P>> for G2Prepared<P> {
    fn from(q: G2Projective<P>) -> Self {
        q.into_affine().into()
    }
}

impl<'a, P: BnConfig> From<&'a G2Affine<P>> for G2Prepared<P> {
    fn from(other: &'a G2Affine<P>) -> Self {
        (*other).into()
    }
}

impl<'a, P: BnConfig> From<&'a G2Projective<P>> for G2Prepared<P> {
    fn from(q: &'a G2Projective<P>) -> Self {
        q.into_affine().into()
    }
}

impl<P: BnConfig> G2Prepared<P> {
    pub const fn is_zero(&self) -> bool {
        self.infinity
    }

    /// Divides every line by its `P.y` coefficient, which leaves that coefficient 1. The Miller
    /// loop then scales the line by `1/P.y` and multiplies by it with
    /// `Fp12::mul_by_014_c4_one` / `Fp12::mul_by_034_c0_one`. Normalized and raw prepared points
    /// share one Miller loop, e.g. a Groth16 verifying key's `gamma` and `delta` with the proof's
    /// `B`. A line whose `P.y` coefficient is zero, which only a `Q` outside the prime-order
    /// subgroup produces, stays raw. Costs one batched `Fp2` inversion and two `Fp2` products per
    /// line. Per-point normalized lines in the ordinary Miller loop follow Zakura
    /// [`normalize_lines`](https://github.com/zakura-core/common/blob/348011982f03eb0bbca703b1c91cd2d52dc97c27/crates/bls12_381/src/pairings.rs#L640-L664),
    /// which normalizes the constant coefficient instead.
    pub fn normalize_lines(&mut self) {
        let mut inv: Vec<Fp2<P::Fp2Config>> =
            self.ell_coeffs.iter().map(|c| *py_coeff::<P>(c)).collect();
        ark_ff::batch_inversion(&mut inv);
        for (c, inv) in self.ell_coeffs.iter_mut().zip(inv) {
            if inv.is_zero() {
                continue;
            }
            match P::TWIST_TYPE {
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
}

/// The coefficient `ell` scales by `P.y`: index 2 for an M-twist, index 0 for a D-twist.
pub(crate) const fn py_coeff<P: BnConfig>(c: &EllCoeff<P>) -> &Fp2<P::Fp2Config> {
    match P::TWIST_TYPE {
        TwistType::M => &c.2,
        TwistType::D => &c.0,
    }
}

/// The two coefficients of a normalized line other than its unit `P.y` coefficient, in slot
/// order.
pub(crate) const fn fixed_line<P: BnConfig>(c: &EllCoeff<P>) -> (Fp2<P::Fp2Config>, Fp2<P::Fp2Config>) {
    match P::TWIST_TYPE {
        TwistType::M => (c.0, c.1),
        TwistType::D => (c.1, c.2),
    }
}

fn mul_by_char<P: BnConfig>(r: G2Affine<P>) -> G2Affine<P> {
    // multiply by field characteristic

    let mut s = r;
    s.x.frobenius_map_in_place(1);
    s.x *= &P::TWIST_MUL_BY_Q_X;
    s.y.frobenius_map_in_place(1);
    s.y *= &P::TWIST_MUL_BY_Q_Y;

    s
}

/// Number of line coefficients a `G2Prepared` holds: one per doubling, one per
/// nonzero ate-loop digit below the top, and two for the Frobenius steps.
const fn num_ell_coeffs<P: BnConfig>() -> usize {
    let ate = P::ATE_LOOP_COUNT;
    let doublings = ate.len() - 1;
    let mut adds = 0usize;
    let mut i = 0;
    while i + 1 < ate.len() {
        if ate[i] != 0 {
            adds += 1;
        }
        i += 1;
    }
    doublings + adds + 2
}

impl<P: BnConfig> Neg for G2Prepared<P> {
    type Output = Self;

    /// The prepared form of `-Q`: negate, in each line, the coefficient that
    /// `ell` scales by `P.y` (index 2 for an M-twist, index 0 for a D-twist),
    /// which is the same as evaluating every line at `-P`. Preparing `-Q` directly
    /// gives these lines. Doubling the representative `(X, -Y, Z)` yields
    /// `(-X3, Y3, -Z3)`, the same point as `(X3, -Y3, Z3)`, and negates only `H`.
    /// Adding `(qx, -qy)` to `(-X, Y, -Z)` keeps `theta` and `theta qx - lambda qy`
    /// and negates only `lambda`. The Frobenius additions follow from
    /// `psi(-Q) = -psi(Q)`. A normalized line negates its other two coefficients instead,
    /// the same line up to a factor `-1`, so it stays normalized. Since
    /// `e(P, -Q) = e(-P, Q)`, negating the affine `P` is cheaper when it is available.
    fn neg(mut self) -> Self {
        for coeff in &mut self.ell_coeffs {
            let normalized = py_coeff::<P>(coeff).is_one();
            match (P::TWIST_TYPE, normalized) {
                (TwistType::M, false) => {
                    coeff.2.neg_in_place();
                },
                (TwistType::D, false) => {
                    coeff.0.neg_in_place();
                },
                (TwistType::M, true) => {
                    coeff.0.neg_in_place();
                    coeff.1.neg_in_place();
                },
                (TwistType::D, true) => {
                    coeff.1.neg_in_place();
                    coeff.2.neg_in_place();
                },
            }
        }
        self
    }
}
