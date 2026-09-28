use super::cubic_extension::{CubicExtConfig, CubicExtField};
use crate::{
    fields::{CyclotomicMultSubgroup, Fp2, Fp2Config, SqrtPrecomputation},
    BigInteger, Field, PrimeField,
};
use core::marker::PhantomData;

/// Whether the Montgomery `sum_of_products` over `C::Fp` reduces six products at once, which it
/// does for a modulus with at least 3 spare bits.
#[inline(always)]
pub(crate) fn reduces_six_products<C: Fp2Config>() -> bool {
    let bits = 64 * <C::Fp as PrimeField>::BigInt::NUM_LIMBS as u32;
    bits - <C::Fp as PrimeField>::MODULUS_BIT_SIZE >= 3
}

/// `y` paired with `NONRESIDUE * y.c1`, the right-hand operand of [`sum_of_fp2_products`] and
/// [`sum_of_two_fp2_products`].
#[inline(always)]
pub(crate) fn fp2_operand<C: Fp2Config>(y: &Fp2<C>) -> (Fp2<C>, C::Fp) {
    let mut nr_c1 = y.c1;
    C::mul_fp_by_nonresidue_in_place(&mut nr_c1);
    (*y, nr_c1)
}

/// `\sum_i{x_i * y_i}` over three `Fp2` products, each `Fp` coordinate one `sum_of_products` of
/// six terms with a single reduction.
#[inline(always)]
pub(crate) fn sum_of_fp2_products<C: Fp2Config>(
    x: [&Fp2<C>; 3],
    y: [&(Fp2<C>, C::Fp); 3],
) -> Fp2<C> {
    let xs = [x[0].c0, x[0].c1, x[1].c0, x[1].c1, x[2].c0, x[2].c1];
    Fp2::new(
        C::Fp::sum_of_products(
            &xs,
            &[y[0].0.c0, y[0].1, y[1].0.c0, y[1].1, y[2].0.c0, y[2].1],
        ),
        C::Fp::sum_of_products(
            &xs,
            &[y[0].0.c1, y[0].0.c0, y[1].0.c1, y[1].0.c0, y[2].0.c1, y[2].0.c0],
        ),
    )
}

/// [`sum_of_fp2_products`] over two `Fp2` products.
#[inline(always)]
pub(crate) fn sum_of_two_fp2_products<C: Fp2Config>(
    x: [&Fp2<C>; 2],
    y: [&(Fp2<C>, C::Fp); 2],
) -> Fp2<C> {
    let xs = [x[0].c0, x[0].c1, x[1].c0, x[1].c1];
    Fp2::new(
        C::Fp::sum_of_products(&xs, &[y[0].0.c0, y[0].1, y[1].0.c0, y[1].1]),
        C::Fp::sum_of_products(&xs, &[y[0].0.c1, y[0].0.c0, y[1].0.c1, y[1].0.c0]),
    )
}

pub trait Fp6Config: 'static + Send + Sync + Copy {
    type Fp2Config: Fp2Config;

    const NONRESIDUE: Fp2<Self::Fp2Config>;

    /// Determines the algorithm for computing square roots.
    const SQRT_PRECOMP: Option<SqrtPrecomputation<Fp6<Self>>> = None;

    /// Coefficients for the Frobenius automorphism.
    const FROBENIUS_COEFF_FP6_C1: &[Fp2<Self::Fp2Config>];
    const FROBENIUS_COEFF_FP6_C2: &[Fp2<Self::Fp2Config>];

    #[inline(always)]
    fn mul_fp2_by_nonresidue_in_place(fe: &mut Fp2<Self::Fp2Config>) -> &mut Fp2<Self::Fp2Config> {
        *fe *= &Self::NONRESIDUE;
        fe
    }
    #[inline(always)]
    fn mul_fp2_by_nonresidue(mut fe: Fp2<Self::Fp2Config>) -> Fp2<Self::Fp2Config> {
        Self::mul_fp2_by_nonresidue_in_place(&mut fe);
        fe
    }
}

pub struct Fp6ConfigWrapper<P: Fp6Config>(PhantomData<P>);

impl<P: Fp6Config> CubicExtConfig for Fp6ConfigWrapper<P> {
    type BasePrimeField = <P::Fp2Config as Fp2Config>::Fp;
    type BaseField = Fp2<P::Fp2Config>;
    type FrobCoeff = Fp2<P::Fp2Config>;

    const SQRT_PRECOMP: Option<SqrtPrecomputation<CubicExtField<Self>>> = P::SQRT_PRECOMP;

    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 6;

    const NONRESIDUE: Self::BaseField = P::NONRESIDUE;

    const FROBENIUS_COEFF_C1: &[Self::FrobCoeff] = P::FROBENIUS_COEFF_FP6_C1;
    const FROBENIUS_COEFF_C2: &[Self::FrobCoeff] = P::FROBENIUS_COEFF_FP6_C2;

    #[inline(always)]
    fn mul_base_field_by_nonresidue_in_place(fe: &mut Self::BaseField) -> &mut Self::BaseField {
        P::mul_fp2_by_nonresidue_in_place(fe)
    }

    fn mul_base_field_by_frob_coeff(
        c1: &mut Self::BaseField,
        c2: &mut Self::BaseField,
        power: usize,
    ) {
        c1.mul_assign_by_frob_coeff(
            &Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD],
        );
        c2.mul_assign_by_frob_coeff(
            &Self::FROBENIUS_COEFF_C2[power % Self::DEGREE_OVER_BASE_PRIME_FIELD],
        );
    }

    /// Schoolbook over `Fp2` with each output `Fp` coordinate one `sum_of_products`, after
    /// zkcrypto [`Fp6::mul_interleaved`](https://github.com/zkcrypto/bls12_381/blob/5d22dd74c2a14fb9f3d3b85ae2c39d0c669ddd99/src/fp6.rs#L200-L274). Karatsuba when
    /// `sum_of_products` cannot take six products at once.
    #[inline(always)]
    fn mul_in_place(a: &mut Fp6<P>, b: &Fp6<P>) {
        if !reduces_six_products::<P::Fp2Config>() {
            return a.mul_assign_karatsuba(b);
        }
        let (l0, l1, l2) = (fp2_operand(&b.c0), fp2_operand(&b.c1), fp2_operand(&b.c2));
        let l1x = fp2_operand(&P::mul_fp2_by_nonresidue(b.c1));
        let l2x = fp2_operand(&P::mul_fp2_by_nonresidue(b.c2));
        let x = [&a.c0, &a.c1, &a.c2];
        *a = Fp6::new(
            sum_of_fp2_products(x, [&l0, &l2x, &l1x]),
            sum_of_fp2_products(x, [&l1, &l0, &l2x]),
            sum_of_fp2_products(x, [&l2, &l1, &l0]),
        );
    }
}

pub type Fp6<P> = CubicExtField<Fp6ConfigWrapper<P>>;

impl<P: Fp6Config> Fp6<P> {
    pub fn mul_assign_by_fp2(&mut self, other: Fp2<P::Fp2Config>) {
        self.c0 *= &other;
        self.c1 *= &other;
        self.c2 *= &other;
    }

    pub fn mul_by_fp(&mut self, element: &<P::Fp2Config as Fp2Config>::Fp) {
        self.c0.mul_assign_by_fp(element);
        self.c1.mul_assign_by_fp(element);
        self.c2.mul_assign_by_fp(element);
    }

    pub fn mul_by_fp2(&mut self, element: &Fp2<P::Fp2Config>) {
        self.c0 *= element;
        self.c1 *= element;
        self.c2 *= element;
    }

    pub fn mul_by_1(&mut self, c1: &Fp2<P::Fp2Config>) {
        let mut b_b = self.c1;
        b_b *= c1;

        let mut t1 = *c1;
        {
            let mut tmp = self.c1;
            tmp += &self.c2;

            t1 *= &tmp;
            t1 -= &b_b;
            P::mul_fp2_by_nonresidue_in_place(&mut t1);
        }

        let mut t2 = *c1;
        {
            let mut tmp = self.c0;
            tmp += &self.c1;

            t2 *= &tmp;
            t2 -= &b_b;
        }

        self.c0 = t1;
        self.c1 = t2;
        self.c2 = b_b;
    }

    pub fn mul_by_01(&mut self, c0: &Fp2<P::Fp2Config>, c1: &Fp2<P::Fp2Config>) {
        let mut a_a = self.c0;
        let mut b_b = self.c1;
        a_a *= c0;
        b_b *= c1;

        let mut t1 = *c1;
        {
            let mut tmp = self.c1;
            tmp += &self.c2;

            t1 *= &tmp;
            t1 -= &b_b;
            P::mul_fp2_by_nonresidue_in_place(&mut t1);
            t1 += &a_a;
        }

        let mut t3 = *c0;
        {
            let mut tmp = self.c0;
            tmp += &self.c2;

            t3 *= &tmp;
            t3 -= &a_a;
            t3 += &b_b;
        }

        let mut t2 = *c0;
        t2 += c1;
        {
            let mut tmp = self.c0;
            tmp += &self.c1;

            t2 *= &tmp;
            t2 -= &a_a;
            t2 -= &b_b;
        }

        self.c0 = t1;
        self.c1 = t2;
        self.c2 = t3;
    }

    /// Multiply by the sparse element `c1 * v + c2 * v^2`, the `(0, c1, c2)` sibling of
    /// [`Self::mul_by_01`]; used by `Fp12::mul_by_01245`. With `v^3 = xi`,
    /// `(a0 + a1 v + a2 v^2)(c1 v + c2 v^2)` is
    /// `xi (a1 c2 + a2 c1) + (a0 c1 + xi a2 c2) v + (a0 c2 + a1 c1) v^2`. Each output `Fp`
    /// coordinate is one `sum_of_products` of four terms when the base field allows six,
    /// otherwise five `Fp2` products with the cross term `a1 c2 + a2 c1` taken by Karatsuba.
    pub fn mul_by_12(&mut self, c1: &Fp2<P::Fp2Config>, c2: &Fp2<P::Fp2Config>) {
        if reduces_six_products::<P::Fp2Config>() {
            let (l1, l2) = (fp2_operand(c1), fp2_operand(c2));
            let l1x = fp2_operand(&P::mul_fp2_by_nonresidue(*c1));
            let l2x = fp2_operand(&P::mul_fp2_by_nonresidue(*c2));
            let (a0, a1, a2) = (self.c0, self.c1, self.c2);
            self.c0 = sum_of_two_fp2_products([&a1, &a2], [&l2x, &l1x]);
            self.c1 = sum_of_two_fp2_products([&a0, &a2], [&l1, &l2x]);
            self.c2 = sum_of_two_fp2_products([&a0, &a1], [&l2, &l1]);
            return;
        }
        let a0 = self.c0;
        let a1 = self.c1;
        let a2 = self.c2;
        let t1 = a1 * c1;
        let t2 = a2 * c2;
        let a0c1 = a0 * c1;
        let a0c2 = a0 * c2;
        // (a1 + a2)(c1 + c2) - t1 - t2 = a1*c2 + a2*c1
        let cross = (a1 + &a2) * &(*c1 + c2) - &t1 - &t2;
        let mut r0 = cross;
        P::mul_fp2_by_nonresidue_in_place(&mut r0);
        let mut xi_t2 = t2;
        P::mul_fp2_by_nonresidue_in_place(&mut xi_t2);
        self.c0 = r0;
        self.c1 = a0c1 + &xi_t2;
        self.c2 = a0c2 + &t1;
    }
}

// We just use the default algorithms; there don't seem to be any faster ones.
impl<P: Fp6Config> CyclotomicMultSubgroup for Fp6<P> {}
