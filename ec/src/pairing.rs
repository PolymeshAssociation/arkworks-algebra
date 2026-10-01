use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, Field, Fp2, Fp2Config, One, PrimeField};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    borrow::Borrow,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    io::{Read, Write},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rand::{
        distributions::{Distribution, Standard},
        Rng,
    },
    vec::*,
    UniformRand, Zero,
};
use educe::Educe;
use zeroize::Zeroize;

use crate::{AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};

/// Collection of types (mainly fields and curves) that together describe
/// how to compute a pairing over a pairing-friendly curve.
pub trait Pairing: Sized + 'static + Copy + Debug + Sync + Send + Eq {
    /// This is the base field of the G1 group and base prime field of G2.
    type BaseField: PrimeField;

    /// This is the scalar field of the G1/G2 groups.
    type ScalarField: PrimeField;

    /// An element in G1.
    type G1: CurveGroup<
            BaseField = Self::BaseField,
            ScalarField = Self::ScalarField,
            Affine = Self::G1Affine,
        > + From<Self::G1Affine>
        + Into<Self::G1Affine>
        // needed due to https://github.com/rust-lang/rust/issues/69640
        + MulAssign<Self::ScalarField>;

    type G1Affine: AffineRepr<Group = Self::G1, BaseField = Self::BaseField, ScalarField = Self::ScalarField>
        + From<Self::G1>
        + Into<Self::G1>
        + Into<Self::G1Prepared>;

    /// A G1 element that has been preprocessed for use in a pairing.
    type G1Prepared: Default
        + Clone
        + Send
        + Sync
        + Debug
        + CanonicalSerialize
        + CanonicalDeserialize
        + From<Self::G1>
        + From<Self::G1Affine>;

    /// An element of G2.
    type G2: CurveGroup<
            ScalarField = Self::ScalarField,
            Affine = Self::G2Affine,
            BaseField: Field<BasePrimeField = Self::BaseField>,
        > + From<Self::G2Affine>
        + Into<Self::G2Affine>
        // needed due to https://github.com/rust-lang/rust/issues/69640
        + MulAssign<Self::ScalarField>;

    /// The affine representation of an element in G2.
    type G2Affine: AffineRepr<
            Group = Self::G2,
            ScalarField = Self::ScalarField,
            BaseField: Field<BasePrimeField = Self::BaseField>,
        > + From<Self::G2>
        + Into<Self::G2>
        + Into<Self::G2Prepared>;

    /// A G2 element that has been preprocessed for use in a pairing.
    type G2Prepared: Default
        + Clone
        + Send
        + Sync
        + Debug
        + CanonicalSerialize
        + CanonicalDeserialize
        + From<Self::G2>
        + From<Self::G2Affine>;

    /// The extension field that hosts the target group of the pairing.
    type TargetField: CyclotomicMultSubgroup;

    /// Computes the product of Miller loops for some number of (G1, G2) pairs.
    fn multi_miller_loop(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> MillerLoopOutput<Self>;

    /// Computes the Miller loop over `a` and `b`.
    fn miller_loop(
        a: impl Into<Self::G1Prepared>,
        b: impl Into<Self::G2Prepared>,
    ) -> MillerLoopOutput<Self> {
        Self::multi_miller_loop([a], [b])
    }

    /// Performs final exponentiation of the result of a `Self::multi_miller_loop`.
    #[must_use]
    fn final_exponentiation(mlo: MillerLoopOutput<Self>) -> Option<PairingOutput<Self>>;

    /// Returns whether `f` lies in the target group GT, i.e. the order-r subgroup
    /// of `Self::TargetField`. The default exponentiates (`f^r == 1`), which costs
    /// about a pairing; pairing curves override it with a cheaper Frobenius-based
    /// test (Scott, <https://eprint.iacr.org/2021/1130>).
    fn is_in_gt(f: &Self::TargetField) -> bool {
        f.pow(Self::ScalarField::characteristic()).is_one()
    }

    /// Returns `f^scalar` for `f` in GT. The default is a windowed cyclotomic
    /// exponentiation; curves may override with a Frobenius-based GLS multi-exp.
    /// `f` must be in GT for an overriding implementation to be correct, since the
    /// Frobenius acts as the scalar `p mod r` only on the order-`r` subgroup.
    fn gt_exp(f: &Self::TargetField, scalar: &[u64]) -> Self::TargetField {
        f.cyclotomic_exp(scalar)
    }

    /// Computes a "product" of pairings.
    fn multi_pairing(
        a: impl IntoIterator<Item = impl Into<Self::G1Prepared>>,
        b: impl IntoIterator<Item = impl Into<Self::G2Prepared>>,
    ) -> PairingOutput<Self> {
        Self::final_exponentiation(Self::multi_miller_loop(a, b)).unwrap()
    }

    /// Performs multiple pairing operations
    fn pairing(
        p: impl Into<Self::G1Prepared>,
        q: impl Into<Self::G2Prepared>,
    ) -> PairingOutput<Self> {
        Self::multi_pairing([p], [q])
    }
}

/// Represents the target group of a pairing. This struct is a
/// wrapper around the field that the target group is embedded in.
#[derive(Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct PairingOutput<P: Pairing>(pub P::TargetField);

impl<P: Pairing> Default for PairingOutput<P> {
    fn default() -> Self {
        // Default value is AdditiveGroup::ZERO (i.e., P::TargetField::one())
        Self::ZERO
    }
}

impl<P: Pairing> CanonicalSerialize for PairingOutput<P> {
    #[inline]
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    #[inline]
    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<P: Pairing> PairingOutput<P> {
    /// Whether this element is in the target group GT, via [`Pairing::is_in_gt`]
    /// (the fast test where a curve provides one).
    pub fn is_in_group(&self) -> bool {
        P::is_in_gt(&self.0)
    }

    /// Reference GT-membership test by exponentiation, `f^r == 1`. Correct but as
    /// costly as a pairing. [`Self::is_in_group`] is the one [`Valid::check`] uses;
    /// this is kept as the oracle the fast test is checked against.
    pub fn is_in_group_naive(&self) -> bool {
        self.0.pow(P::ScalarField::characteristic()).is_one()
    }
}

impl<P: Pairing> Valid for PairingOutput<P> {
    fn check(&self) -> Result<(), SerializationError> {
        if P::is_in_gt(&self.0) {
            Ok(())
        } else {
            Err(SerializationError::InvalidData)
        }
    }
}

impl<P: Pairing> CanonicalDeserialize for PairingOutput<P> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let f = P::TargetField::deserialize_with_mode(reader, compress, validate).map(Self)?;
        if validate == Validate::Yes {
            f.check()?;
        }
        Ok(f)
    }
}

impl<P: Pairing> Display for PairingOutput<P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl<P: Pairing> Zero for PairingOutput<P> {
    /// The identity element, or "zero", of the group is the identity element of the multiplicative group of the underlying field, i.e., `P::TargetField::one()`.
    fn zero() -> Self {
        Self(P::TargetField::one())
    }

    fn is_zero(&self) -> bool {
        self.0.is_one()
    }
}

impl<'a, P: Pairing> Add<&'a Self> for PairingOutput<P> {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &'a Self) -> Self {
        self += other;
        self
    }
}

impl<'a, P: Pairing> AddAssign<&'a Self> for PairingOutput<P> {
    fn add_assign(&mut self, other: &'a Self) {
        self.0 *= other.0;
    }
}

impl<'a, P: Pairing> SubAssign<&'a Self> for PairingOutput<P> {
    fn sub_assign(&mut self, other: &'a Self) {
        self.0 *= other.0.cyclotomic_inverse().unwrap();
    }
}

impl<'a, P: Pairing> Sub<&'a Self> for PairingOutput<P> {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &'a Self) -> Self {
        self -= other;
        self
    }
}

ark_ff::impl_additive_ops_from_ref!(PairingOutput, Pairing);

impl<P: Pairing, T: Borrow<P::ScalarField>> MulAssign<T> for PairingOutput<P> {
    fn mul_assign(&mut self, other: T) {
        *self = self.mul_bigint(other.borrow().into_bigint());
    }
}

impl<P: Pairing, T: Borrow<P::ScalarField>> Mul<T> for PairingOutput<P> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        self.mul_bigint(other.borrow().into_bigint())
    }
}

impl<P: Pairing> Zeroize for PairingOutput<P> {
    fn zeroize(&mut self) {
        self.0.zeroize()
    }
}

impl<P: Pairing> Neg for PairingOutput<P> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(self.0.cyclotomic_inverse().unwrap())
    }
}

impl<P: Pairing> Distribution<PairingOutput<P>> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PairingOutput<P> {
        // Sample a random G1 element
        let g1 = P::G1::rand(rng);
        // Sample a random G2 element
        let g2 = P::G2::rand(rng);
        P::pairing(g1, g2)
    }
}

impl<P: Pairing> AdditiveGroup for PairingOutput<P> {
    type Scalar = P::ScalarField;

    const ZERO: Self = Self(P::TargetField::ONE);

    fn double_in_place(&mut self) -> &mut Self {
        self.0.cyclotomic_square_in_place();
        self
    }
}

impl<P: Pairing> PrimeGroup for PairingOutput<P> {
    type ScalarField = P::ScalarField;

    fn generator() -> Self {
        // TODO: hardcode these values.
        // Sample a random G1 element
        let g1 = P::G1::generator();
        // Sample a random G2 element
        let g2 = P::G2::generator();
        P::pairing(g1.into(), g2.into())
    }

    fn mul_bigint(&self, other: impl AsRef<[u64]>) -> Self {
        Self(P::gt_exp(&self.0, other.as_ref()))
    }

    /// [`Self::mul_bigint`] on the big-endian bits, packed into little-endian limbs from the end.
    fn mul_bits_be(&self, other: impl Iterator<Item = bool>) -> Self {
        let bits = other.collect::<Vec<_>>();
        let limbs = bits
            .rchunks(64)
            .map(|chunk| chunk.iter().fold(0u64, |r, bit| (r << 1) | u64::from(*bit)))
            .collect::<Vec<_>>();
        self.mul_bigint(limbs)
    }
}

impl<P: Pairing> crate::ScalarMul for PairingOutput<P> {
    type MulBase = Self;
    const NEGATION_IS_CHEAP: bool = P::TargetField::INVERSE_IS_FAST;

    fn batch_convert_to_mul_base(bases: &[Self]) -> Vec<Self::MulBase> {
        bases.to_vec()
    }
}

impl<P: Pairing> VariableBaseMSM for PairingOutput<P> {
    type Bucket = Self;
    const ZERO_BUCKET: Self::Bucket = Self::ZERO;
}

/// Represents the output of the Miller loop of the pairing.
#[derive(Educe)]
#[educe(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[must_use]
pub struct MillerLoopOutput<P: Pairing>(pub P::TargetField);

impl<P: Pairing> Mul<P::ScalarField> for MillerLoopOutput<P> {
    type Output = Self;

    fn mul(self, other: P::ScalarField) -> Self {
        Self(self.0.pow(other.into_bigint()))
    }
}

/// `prod_i bases[i]^{digits[i]}` for GT elements: a 16-entry subset-product table
/// and one interleaved square-and-multiply over the widest digit. A negative digit
/// conjugates its base (cheap in GT). The heart of the Frobenius GLS/GLV
/// exponentiation of GT elements. Costs 15 multiplications for the table, then one
/// cyclotomic squaring and at most one multiplication per bit of the widest digit.
///
/// Straus simultaneous multi-exponentiation (Hankerson, Menezes, Vanstone, Guide
/// to Elliptic Curve Cryptography (2004), Algorithm 3.48), with the four-digit
/// decomposition supplied by the callers' Galbraith-Scott GLS (`Bn::gt_exp`,
/// `Bls12::gt_exp`).
pub fn gt_multiexp<F: CyclotomicMultSubgroup>(
    mut bases: [F; 4],
    digits: [(bool, u64); 4],
) -> F {
    for i in 0..4 {
        if digits[i].0 {
            bases[i].cyclotomic_inverse_in_place();
        }
    }
    let mags = [digits[0].1, digits[1].1, digits[2].1, digits[3].1];
    let mut table = [F::one(); 16];
    for mask in 1..16usize {
        let i = mask.trailing_zeros() as usize;
        table[mask] = table[mask & (mask - 1)];
        table[mask] *= &bases[i];
    }
    let nbits = mags.iter().map(|&m| 64 - m.leading_zeros()).max().unwrap_or(0);
    let mut acc = F::one();
    let mut started = false;
    for bit in (0..nbits).rev() {
        if started {
            acc.cyclotomic_square_in_place();
        }
        let mut mask = 0usize;
        for i in 0..4 {
            if (mags[i] >> bit) & 1 == 1 {
                mask |= 1 << i;
            }
        }
        if mask != 0 {
            if started {
                acc *= &table[mask];
            } else {
                acc = table[mask];
                started = true;
            }
        }
    }
    acc
}

/// Preprocesses a G1 element for use in a pairing.
pub fn prepare_g1<E: Pairing>(g: impl Into<E::G1Affine>) -> E::G1Prepared {
    E::G1Prepared::from(g.into())
}

/// Preprocesses a G2 element for use in a pairing.
pub fn prepare_g2<E: Pairing>(g: impl Into<E::G2Affine>) -> E::G2Prepared {
    E::G2Prepared::from(g.into())
}

/// `G^2 - 12 E^2` over `Fp2`, the `Y` of the BN and BLS12 G2 doubling. With the nonresidue `-1`
/// each coordinate is one `sum_of_products`, `(g0 + g1)(g0 - g1) + (e0 + e1) 12 (e1 - e0)` and
/// `g0 2 g1 - e0 24 e1`, with one reduction each instead of one per multiplication. After
/// RELIC [`pp_dbl_k12_projc_lazyr`](https://github.com/relic-toolkit/relic/blob/7778fce00ad056563435e5b3e8083ea2cf46b2e5/src/pp/relic_pp_dbl_k12.c#L291-L297).
pub(crate) fn g2_doubling_y<C: Fp2Config>(g: &Fp2<C>, e: &Fp2<C>) -> Fp2<C> {
    let twelve = |x: C::Fp| {
        let four = x.double().double();
        four.double() + four
    };
    if C::NONRESIDUE != C::Fp::NEG_ONE {
        let e_square = e.square();
        return g.square() - Fp2::new(twelve(e_square.c0), twelve(e_square.c1));
    }
    let m = twelve(e.c1 - e.c0);
    let n = -twelve(e.c1).double();
    Fp2::new(
        C::Fp::sum_of_products(&[g.c0 + g.c1, e.c0 + e.c1], &[g.c0 - g.c1, m]),
        C::Fp::sum_of_products(&[g.c0, e.c0], &[g.c1.double(), n]),
    )
}

#[cfg(test)]
mod tests {
    use super::g2_doubling_y;
    use ark_ff::{AdditiveGroup, Field, One};
    use ark_std::{test_rng, UniformRand};
    use ark_test_curves::bls12_381::{Fq, Fq2};

    #[test]
    fn g2_doubling_y_matches_squares() {
        let direct = |g: Fq2, e: Fq2| g.square() - e.square() * Fq2::from(12u64);
        let mut rng = test_rng();
        let max = -Fq::one();
        let edges = [
            Fq2::ZERO,
            Fq2::one(),
            Fq2::new(max, max),
            Fq2::new(max, Fq::ZERO),
        ];
        for g in edges {
            for e in edges {
                assert_eq!(g2_doubling_y(&g, &e), direct(g, e));
            }
        }
        for _ in 0..1000 {
            let (g, e) = (Fq2::rand(&mut rng), Fq2::rand(&mut rng));
            assert_eq!(g2_doubling_y(&g, &e), direct(g, e));
        }
    }
}
