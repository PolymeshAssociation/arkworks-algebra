use ark_ff::{AdditiveGroup, BitIteratorBE, Field, Fp2};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{io::Read, ops::Neg, vec::*, One};
use educe::Educe;

use crate::{
    bls12::{Bls12Config, TwistType},
    models::fp12_lines as lines,
    pairing::g2_doubling_y,
    short_weierstrass::{Affine, Projective},
    AffineRepr, CurveGroup,
};

pub type G2Affine<P> = Affine<<P as Bls12Config>::G2Config>;
pub type G2Projective<P> = Projective<<P as Bls12Config>::G2Config>;

/// `Q` preprocessed into the line functions of the Miller loop, three `Fp2` coefficients per
/// line, from the homogeneous projective doubling and mixed addition of Aranha, Barreto, Longa,
/// Ricardini, [The Realm of the Pairings](https://eprint.iacr.org/2013/722), section 4.3. A line
/// is only defined up to a factor in `Fp2`. The final exponentiation `(p^12 - 1) / r` is a
/// multiple of `p^6 - 1`, hence of `p^2 - 1`, so it maps every nonzero element of `Fp2` to 1. The
/// doubling keeps its point scaled by 4, and [`G2Prepared::normalize_lines`] and
/// [`G2PreparedFixed`] rescale each line on this basis, so raw `MillerLoopOutput` values depend on
/// the implementation and only final-exponentiated values are canonical.
#[derive(Educe, CanonicalSerialize)]
#[educe(Clone, Debug, PartialEq, Eq)]
pub struct G2Prepared<P: Bls12Config> {
    /// Stores the coefficients of the line evaluations as calculated in
    /// <https://eprint.iacr.org/2013/722.pdf>
    pub ell_coeffs: Vec<EllCoeff<P>>,
    pub infinity: bool,
}

impl<P: Bls12Config> Valid for G2Prepared<P> {
    /// The line count the Miller loop consumes, none at infinity.
    fn check(&self) -> Result<(), SerializationError> {
        let expected = if self.infinity {
            0
        } else {
            num_ell_coeffs::<P>()
        };
        if self.ell_coeffs.len() != expected {
            return Err(SerializationError::InvalidData);
        }
        self.ell_coeffs.check()?;
        Ok(())
    }
}

impl<P: Bls12Config> CanonicalDeserialize for G2Prepared<P> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let ell_coeffs =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let infinity =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let prepared = Self {
            ell_coeffs,
            infinity,
        };
        if validate == Validate::Yes {
            prepared.check()?;
        }
        Ok(prepared)
    }
}

pub type EllCoeff<P> = (
    Fp2<<P as Bls12Config>::Fp2Config>,
    Fp2<<P as Bls12Config>::Fp2Config>,
    Fp2<<P as Bls12Config>::Fp2Config>,
);

#[derive(Educe)]
#[educe(Clone, Copy, Debug)]
pub struct G2HomProjective<P: Bls12Config> {
    x: Fp2<P::Fp2Config>,
    y: Fp2<P::Fp2Config>,
    z: Fp2<P::Fp2Config>,
}

impl<P: Bls12Config> Default for G2Prepared<P> {
    fn default() -> Self {
        Self::from(G2Affine::<P>::generator())
    }
}

impl<P: Bls12Config> From<G2Affine<P>> for G2Prepared<P> {
    fn from(q: G2Affine<P>) -> Self {
        let zero = Self {
            ell_coeffs: Vec::new(),
            infinity: true,
        };
        q.xy().map_or(zero, |(q_x, q_y)| {
            let mut ell_coeffs = Vec::with_capacity(num_ell_coeffs::<P>());
            let mut r = G2HomProjective::<P> {
                x: q_x,
                y: q_y,
                z: Fp2::one(),
            };

            let mut iter = BitIteratorBE::without_leading_zeros(P::X)
                .skip(1)
                .peekable();
            while let Some(i) = iter.next() {
                if iter.peek().is_none() && !i {
                    // Last iteration with no trailing addition: the doubled
                    // point is discarded, so only the line is needed.
                    ell_coeffs.push(r.tangent_line());
                } else {
                    ell_coeffs.push(r.double_in_place());
                    if i {
                        ell_coeffs.push(r.add_in_place(&q));
                    }
                }
            }

            debug_assert_eq!(ell_coeffs.len(), num_ell_coeffs::<P>());
            Self {
                ell_coeffs,
                infinity: false,
            }
        })
    }
}

impl<P: Bls12Config> From<G2Projective<P>> for G2Prepared<P> {
    fn from(q: G2Projective<P>) -> Self {
        q.into_affine().into()
    }
}

impl<'a, P: Bls12Config> From<&'a G2Affine<P>> for G2Prepared<P> {
    fn from(other: &'a G2Affine<P>) -> Self {
        (*other).into()
    }
}

impl<'a, P: Bls12Config> From<&'a G2Projective<P>> for G2Prepared<P> {
    fn from(q: &'a G2Projective<P>) -> Self {
        q.into_affine().into()
    }
}

impl<P: Bls12Config> G2Prepared<P> {
    pub const fn is_zero(&self) -> bool {
        self.infinity
    }

    /// Divides every line by its `P.y` coefficient, which leaves that coefficient 1, as
    /// [`G2PreparedFixed`] does. The Miller loop then scales the line by `1/P.y` and multiplies by
    /// it with `Fp12::mul_by_014_c4_one` / `Fp12::mul_by_034_c0_one`. Normalized and raw prepared
    /// points share one Miller loop, e.g. a Groth16 verifying key's `gamma` and `delta` with the
    /// proof's `B`. A line whose `P.y` coefficient is zero, which only a `Q` outside the
    /// prime-order subgroup produces, stays raw. Costs one batched `Fp2` inversion and two `Fp2`
    /// products per line. Per-point normalized lines in the ordinary Miller loop follow Zakura
    /// [`normalize_lines`](https://github.com/zakura-core/common/blob/348011982f03eb0bbca703b1c91cd2d52dc97c27/crates/bls12_381/src/pairings.rs#L640-L664),
    /// which normalizes the constant coefficient instead.
    pub fn normalize_lines(&mut self) {
        lines::normalize_lines::<P::Fp12Config>(P::TWIST_TYPE, &mut self.ell_coeffs);
    }
}

impl<P: Bls12Config> G2HomProjective<P> {
    fn double_in_place(&mut self) -> EllCoeff<P> {
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

    /// The doubling line without the point update, for a last iteration whose doubled point is
    /// never read, i.e. when the last bit of `X` is 0. The BN loop is followed by the two
    /// Frobenius additions, which read the point, so it has no such step. After gnark-crypto
    /// [`tangentLine`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/pairing.go#L316-L337).
    fn tangent_line(&self) -> EllCoeff<P> {
        let b = self.y.square();
        let c = self.z.square();
        let e = P::mul_by_3b_twist(c);
        let h = (self.y + &self.z).square() - &(b + &c);
        let i = e - &b;
        let j = self.x.square();
        match P::TWIST_TYPE {
            TwistType::M => (i, j.double() + &j, -h),
            TwistType::D => (-h, j.double() + &j, i),
        }
    }

    fn add_in_place(&mut self, q: &G2Affine<P>) -> EllCoeff<P> {
        let (qx, qy) = q.xy().unwrap();
        // Mixed addition T + Q with its chord line, 2013/722 section 4.3 (the formulas after
        // equation (13)): theta = Y - qy Z, lambda = X - qx Z, and the line
        // (theta qx - lambda qy, -theta, lambda). The coefficients are homogeneous of degree 1
        // in (X, Y, Z), so the scaled point of `double_in_place` again costs an `Fp` factor.
        let theta = self.y - &(qy * &self.z);
        let lambda = self.x - &(qx * &self.z);
        let c = theta.square();
        let d = lambda.square();
        let e = lambda * &d;
        let f = self.z * &c;
        let g = self.x * &d;
        let h = e + &f - &g.double();
        self.x = lambda * &h;
        self.y = theta * &(g - &h) - &(e * &self.y);
        self.z *= &e;
        let j = theta * &qx - &(lambda * &qy);

        match P::TWIST_TYPE {
            TwistType::M => (j, -theta, lambda),
            TwistType::D => (lambda, -theta, j),
        }
    }
}

/// Number of line coefficients a `G2Prepared` holds: one per doubling plus one
/// per set bit of `X` after the most significant.
const fn num_ell_coeffs<P: Bls12Config>() -> usize {
    let x = P::X;
    let mut bits = 64 * x.len();
    let mut i = x.len();
    while i > 0 {
        let limb = x[i - 1];
        if limb == 0 {
            bits -= 64;
            i -= 1;
        } else {
            bits -= limb.leading_zeros() as usize;
            break;
        }
    }
    let mut ones = 0usize;
    let mut j = 0;
    while j < x.len() {
        ones += x[j].count_ones() as usize;
        j += 1;
    }
    (bits - 1) + (ones - 1)
}

/// A `G2Prepared` with every line reduced to two `Fp2` coefficients, normalized
/// so the coefficient `ell` scales by `P.y` is one after a per-`P` rescale. One
/// third smaller, and the Miller loop uses a cheaper sparse multiplication
/// (`Fp12::mul_by_014_c4_one` / `Fp12::mul_by_034_c0_one`).
/// Built once for a fixed `Q` (e.g. a Groth16 verifying-key point) and reused. Each stored line
/// is the raw line divided by its `P.y` coefficient (an `Fp2` constant of `Q`), and the Miller loop
/// divides the evaluated line by `P.y` (an `Fp` constant of `P`), which makes that slot 1. Both
/// factors lie in `Fp2`, which the final exponentiation maps to 1 (see [`G2Prepared`]). Building
/// costs one batched `Fp2` inversion and two `Fp2` products per line, repaid from about the
/// second Miller loop with the same `Q`.
///
/// Follows gnark-crypto's precomputed fixed-Q lines
/// ([`PrecomputeLines`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/pairing.go#L660-L732) and its fixed-Q Miller loop),
/// whose normalized two-coefficient lines are consumed by
/// [`MulBy01`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12_pairing.go#L70-L89) /
/// [`MulBy34`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-377/internal/fptower/e12_pairing.go#L67-L85).
#[derive(Educe, CanonicalSerialize)]
#[educe(Clone, Debug, PartialEq, Eq)]
pub struct G2PreparedFixed<P: Bls12Config> {
    pub lines: Vec<FixedLine<P>>,
    pub infinity: bool,
}

/// A normalized line's two remaining `Fp2` coefficients.
pub type FixedLine<P> = (
    Fp2<<P as Bls12Config>::Fp2Config>,
    Fp2<<P as Bls12Config>::Fp2Config>,
);

impl<P: Bls12Config> Valid for G2PreparedFixed<P> {
    /// The line count the Miller loop consumes, none at infinity.
    fn check(&self) -> Result<(), SerializationError> {
        let expected = if self.infinity {
            0
        } else {
            num_ell_coeffs::<P>()
        };
        if self.lines.len() != expected {
            return Err(SerializationError::InvalidData);
        }
        self.lines.check()?;
        Ok(())
    }
}

impl<P: Bls12Config> CanonicalDeserialize for G2PreparedFixed<P> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let lines = CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let infinity =
            CanonicalDeserialize::deserialize_with_mode(&mut reader, compress, validate)?;
        let prepared = Self { lines, infinity };
        if validate == Validate::Yes {
            prepared.check()?;
        }
        Ok(prepared)
    }
}

/// A line whose `P.y` coefficient is zero, which [`G2PreparedFixed`] cannot hold. Only a `Q`
/// outside the prime-order subgroup produces one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroPyCoefficient;

impl<P: Bls12Config> TryFrom<G2Prepared<P>> for G2PreparedFixed<P> {
    type Error = ZeroPyCoefficient;

    fn try_from(mut prep: G2Prepared<P>) -> Result<Self, Self::Error> {
        if prep.infinity {
            return Ok(Self {
                lines: Vec::new(),
                infinity: true,
            });
        }
        prep.normalize_lines();
        let twist = P::TWIST_TYPE;
        if !prep
            .ell_coeffs
            .iter()
            .all(|c| lines::py_coeff::<P::Fp12Config>(twist, c).is_one())
        {
            return Err(ZeroPyCoefficient);
        }
        Ok(Self {
            lines: prep
                .ell_coeffs
                .iter()
                .map(|c| lines::fixed_line::<P::Fp12Config>(twist, c))
                .collect(),
            infinity: false,
        })
    }
}

impl<P: Bls12Config> TryFrom<G2Affine<P>> for G2PreparedFixed<P> {
    type Error = ZeroPyCoefficient;

    fn try_from(q: G2Affine<P>) -> Result<Self, Self::Error> {
        G2Prepared::from(q).try_into()
    }
}

impl<P: Bls12Config> Default for G2PreparedFixed<P> {
    fn default() -> Self {
        Self::try_from(G2Affine::<P>::generator()).expect("the generator is in the subgroup")
    }
}

impl<P: Bls12Config> Neg for G2Prepared<P> {
    type Output = Self;

    /// The prepared form of `-Q`: negate, in each line, the coefficient that
    /// `ell` scales by `P.y` (index 2 for an M-twist, index 0 for a D-twist),
    /// which is the same as evaluating every line at `-P`. Preparing `-Q` directly
    /// gives these lines. Doubling the representative `(X, -Y, Z)` yields
    /// `(-X3, Y3, -Z3)`, the same point as `(X3, -Y3, Z3)`, and negates only `H`.
    /// Adding `(qx, -qy)` to `(-X, Y, -Z)` keeps `theta` and `theta qx - lambda qy`
    /// and negates only `lambda`. A normalized line negates its other two coefficients
    /// instead, the same line up to a factor `-1`, so it stays normalized. Since
    /// `e(P, -Q) = e(-P, Q)`, negating the affine `P` is cheaper when it is available.
    fn neg(mut self) -> Self {
        lines::negate_lines::<P::Fp12Config>(P::TWIST_TYPE, &mut self.ell_coeffs);
        self
    }
}
