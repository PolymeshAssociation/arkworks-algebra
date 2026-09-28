use super::quadratic_extension::{QuadExtConfig, QuadExtField};
use crate::{
    fields::{
        fp6_3over2::{
            fp2_operand, reduces_six_products, sum_of_fp2_products, sum_of_two_fp2_products, Fp6,
            Fp6Config,
        },
        Field, Fp2, Fp2Config as Fp2ConfigTrait,
    },
    AdditiveGroup, CyclotomicMultSubgroup, Zero,
};
use core::{marker::PhantomData, ops::Not};

type Fp2Config<P> = <<P as Fp12Config>::Fp6Config as Fp6Config>::Fp2Config;

pub trait Fp12Config: 'static + Send + Sync + Copy {
    type Fp6Config: Fp6Config;

    /// This *must* equal (0, 1, 0);
    /// see [[DESD06, Section 6.1]](https://eprint.iacr.org/2006/471.pdf).
    const NONRESIDUE: Fp6<Self::Fp6Config>;

    /// Coefficients for the Frobenius automorphism.
    const FROBENIUS_COEFF_FP12_C1: &[Fp2<Fp2Config<Self>>];

    /// Multiply by quadratic nonresidue v.
    #[inline(always)]
    fn mul_fp6_by_nonresidue_in_place(fe: &mut Fp6<Self::Fp6Config>) -> &mut Fp6<Self::Fp6Config> {
        // see [[DESD06, Section 6.1]](https://eprint.iacr.org/2006/471.pdf).
        let old_c1 = fe.c1;
        fe.c1 = fe.c0;
        fe.c0 = fe.c2;
        Self::Fp6Config::mul_fp2_by_nonresidue_in_place(&mut fe.c0);
        fe.c2 = old_c1;
        fe
    }
}

pub struct Fp12ConfigWrapper<P: Fp12Config>(PhantomData<P>);

impl<P: Fp12Config> QuadExtConfig for Fp12ConfigWrapper<P> {
    type BasePrimeField = <Fp2Config<P> as Fp2ConfigTrait>::Fp;
    type BaseField = Fp6<P::Fp6Config>;
    type FrobCoeff = Fp2<Fp2Config<P>>;

    const DEGREE_OVER_BASE_PRIME_FIELD: usize = 12;

    const NONRESIDUE: Self::BaseField = P::NONRESIDUE;

    const FROBENIUS_COEFF_C1: &[Self::FrobCoeff] = P::FROBENIUS_COEFF_FP12_C1;

    #[inline(always)]
    fn mul_base_field_by_nonresidue_in_place(fe: &mut Self::BaseField) -> &mut Self::BaseField {
        P::mul_fp6_by_nonresidue_in_place(fe)
    }

    fn mul_base_field_by_frob_coeff(fe: &mut Self::BaseField, power: usize) {
        let coeff = &Self::FROBENIUS_COEFF_C1[power % Self::DEGREE_OVER_BASE_PRIME_FIELD];
        fe.c0.mul_assign_by_frob_coeff(coeff);
        fe.c1.mul_assign_by_frob_coeff(coeff);
        fe.c2.mul_assign_by_frob_coeff(coeff);
    }
}

pub type Fp12<P> = QuadExtField<Fp12ConfigWrapper<P>>;

impl<P: Fp12Config> Fp12<P> {
    pub fn mul_by_fp(&mut self, element: &<Self as Field>::BasePrimeField) {
        self.c0.mul_by_fp(element);
        self.c1.mul_by_fp(element);
    }

    /// Multiply by the D-twist line element `c0 + c3 w + c4 v w`. Schoolbook over `Fp2` with
    /// each output `Fp` coordinate one `sum_of_products` of six terms when the base field
    /// allows it (after Zakura [#489](https://github.com/zakura-core/common/pull/489)),
    /// Karatsuba otherwise.
    pub fn mul_by_034(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c3: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
    ) {
        if !reduces_six_products::<Fp2Config<P>>() {
            return self.mul_by_034_karatsuba(c0, c3, c4);
        }
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let (l0, l3, l4) = (fp2_operand(c0), fp2_operand(c3), fp2_operand(c4));
        let (l3x, l4x) = (fp2_operand(&nr(*c3)), fp2_operand(&nr(*c4)));
        let (a, b) = (self.c0, self.c1);
        self.c0 = Fp6::new(
            sum_of_fp2_products([&a.c0, &b.c1, &b.c2], [&l0, &l4x, &l3x]),
            sum_of_fp2_products([&a.c1, &b.c0, &b.c2], [&l0, &l3, &l4x]),
            sum_of_fp2_products([&a.c2, &b.c0, &b.c1], [&l0, &l4, &l3]),
        );
        self.c1 = Fp6::new(
            sum_of_fp2_products([&a.c0, &a.c2, &b.c0], [&l3, &l4x, &l0]),
            sum_of_fp2_products([&a.c0, &a.c1, &b.c1], [&l4, &l3, &l0]),
            sum_of_fp2_products([&a.c1, &a.c2, &b.c2], [&l4, &l3, &l0]),
        );
    }

    fn mul_by_034_karatsuba(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c3: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
    ) {
        let a0 = self.c0.c0 * c0;
        let a1 = self.c0.c1 * c0;
        let a2 = self.c0.c2 * c0;
        let a = Fp6::new(a0, a1, a2);
        let mut b = self.c1;
        b.mul_by_01(c3, c4);

        let c0 = *c0 + c3;
        let c1 = c4;
        let mut e = self.c0 + &self.c1;
        e.mul_by_01(&c0, c1);
        self.c1 = e - &(a + &b);
        self.c0 = b;
        P::mul_fp6_by_nonresidue_in_place(&mut self.c0);
        self.c0 += &a;
    }

    /// Multiply by the M-twist line element `c0 + c1 v + c4 v w`. Schoolbook over `Fp2` with
    /// each output `Fp` coordinate one `sum_of_products` of six terms when the base field
    /// allows it (after Zakura [#489](https://github.com/zakura-core/common/pull/489)),
    /// Karatsuba otherwise.
    pub fn mul_by_014(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c1: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
    ) {
        if !reduces_six_products::<Fp2Config<P>>() {
            return self.mul_by_014_karatsuba(c0, c1, c4);
        }
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let (l0, l1, l4) = (fp2_operand(c0), fp2_operand(c1), fp2_operand(c4));
        let (l1x, l4x) = (fp2_operand(&nr(*c1)), fp2_operand(&nr(*c4)));
        let (a, b) = (self.c0, self.c1);
        self.c0 = Fp6::new(
            sum_of_fp2_products([&a.c0, &a.c2, &b.c1], [&l0, &l1x, &l4x]),
            sum_of_fp2_products([&a.c0, &a.c1, &b.c2], [&l1, &l0, &l4x]),
            sum_of_fp2_products([&a.c1, &a.c2, &b.c0], [&l1, &l0, &l4]),
        );
        self.c1 = Fp6::new(
            sum_of_fp2_products([&a.c2, &b.c0, &b.c2], [&l4x, &l0, &l1x]),
            sum_of_fp2_products([&a.c0, &b.c0, &b.c1], [&l4, &l1, &l0]),
            sum_of_fp2_products([&a.c1, &b.c1, &b.c2], [&l4, &l1, &l0]),
        );
    }

    fn mul_by_014_karatsuba(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c1: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
    ) {
        let mut aa = self.c0;
        aa.mul_by_01(c0, c1);
        let mut bb = self.c1;
        bb.mul_by_1(c4);
        let mut o = *c1;
        o += c4;
        self.c1 += &self.c0;
        self.c1.mul_by_01(c0, &o);
        self.c1 -= &aa;
        self.c1 -= &bb;
        self.c0 = bb;
        P::mul_fp6_by_nonresidue_in_place(&mut self.c0);
        self.c0 += &aa;
    }

    /// Product of two M-twist line elements (each nonzero at positions 0, 1, 4),
    /// returning coefficients at positions 0, 1, 2, 4, 5 for [`Self::mul_by_01245`].
    /// Positions 0 to 5 are the coefficients of `1, v, v^2, w, vw, v^2 w`, i.e.
    /// `(c0.c0, c0.c1, c0.c2, c1.c0, c1.c1, c1.c2)`, with `w^2 = v` and `v^3 = xi`.
    /// Then `(vw)^2 = xi` puts `c4 d4` at position 0 and `v (vw) = v^2 w` puts the
    /// cross term of positions 1 and 4 at position 5. Six `Fp2` products, three of
    /// them Karatsuba cross terms.
    ///
    /// Ported from gnark-crypto
    /// [`Mul014By014`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12_pairing.go#L92-L117) /
    /// [`MulBy01245`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12_pairing.go#L140-L159).
    /// The sparse-multiplication framework is Aranha, Karabina, Longa, Gebotys,
    /// Lopez, <https://eprint.iacr.org/2010/526> section 4.
    pub fn mul_014_by_014(
        c0: &Fp2<Fp2Config<P>>,
        c1: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
        d0: &Fp2<Fp2Config<P>>,
        d1: &Fp2<Fp2Config<P>>,
        d4: &Fp2<Fp2Config<P>>,
    ) -> [Fp2<Fp2Config<P>>; 5] {
        let x0 = *c0 * d0;
        let x1 = *c1 * d1;
        let x4 = *c4 * d4;
        let x01 = (*c0 + c1) * &(*d0 + d1) - &x0 - &x1;
        let x04 = (*c0 + c4) * &(*d0 + d4) - &x0 - &x4;
        let x14 = (*c1 + c4) * &(*d1 + d4) - &x1 - &x4;
        let z00 = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue(x4) + &x0;
        [z00, x01, x1, x04, x14]
    }

    /// Multiply by two M-twist line elements `c` and `d`. Two [`Self::mul_by_014`] when
    /// `sum_of_products` takes six products at once, [`Self::mul_014_by_014`] and
    /// [`Self::mul_by_01245`] otherwise.
    pub fn mul_by_014_pair(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c1: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
        d0: &Fp2<Fp2Config<P>>,
        d1: &Fp2<Fp2Config<P>>,
        d4: &Fp2<Fp2Config<P>>,
    ) {
        if reduces_six_products::<Fp2Config<P>>() {
            self.mul_by_014(c0, c1, c4);
            self.mul_by_014(d0, d1, d4);
        } else {
            self.mul_by_01245(&Self::mul_014_by_014(c0, c1, c4, d0, d1, d4));
        }
    }

    /// Multiply by the sparse element with `c0 = (x0, x1, x2)` and
    /// `c1 = (0, x3, x4)`, the shape [`Self::mul_014_by_014`] returns. Karatsuba over
    /// `w` with two full `Fp6` products and one `Fp6::mul_by_12`, 17 `Fp2` products.
    pub fn mul_by_01245(&mut self, x: &[Fp2<Fp2Config<P>>; 5]) {
        let g0 = Fp6::new(x[0], x[1], x[2]);
        let sum_g = Fp6::new(x[0], x[1] + &x[3], x[2] + &x[4]);
        let mut b = self.c0;
        b *= &g0;
        let mut c = self.c1;
        c.mul_by_12(&x[3], &x[4]);
        let mut a = self.c0 + &self.c1;
        a *= &sum_g;
        self.c1 = a - &b - &c;
        let mut c0 = c;
        P::mul_fp6_by_nonresidue_in_place(&mut c0);
        self.c0 = c0 + &b;
    }

    /// Product of two D-twist line elements (each nonzero at positions 0, 3, 4),
    /// returning coefficients at positions 0, 1, 2, 3, 4 for [`Self::mul_by_01234`].
    /// With the positions of [`Self::mul_014_by_014`], `w^2 = v` puts `c3 d3` at
    /// position 1, `(vw)^2 = xi` puts `c4 d4` at position 0, and `w (vw) = v^2` puts
    /// the cross term of positions 3 and 4 at position 2. Six `Fp2` products.
    ///
    /// Ported from gnark-crypto
    /// [`Mul034By034`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/internal/fptower/e12_pairing.go#L133-L158) /
    /// [`MulBy01234`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/internal/fptower/e12_pairing.go#L181-L200).
    pub fn mul_034_by_034(
        c0: &Fp2<Fp2Config<P>>,
        c3: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
        d0: &Fp2<Fp2Config<P>>,
        d3: &Fp2<Fp2Config<P>>,
        d4: &Fp2<Fp2Config<P>>,
    ) -> [Fp2<Fp2Config<P>>; 5] {
        let x0 = *c0 * d0;
        let x3 = *c3 * d3;
        let x4 = *c4 * d4;
        let x03 = (*c0 + c3) * &(*d0 + d3) - &x0 - &x3;
        let x04 = (*c0 + c4) * &(*d0 + d4) - &x0 - &x4;
        let x34 = (*c3 + c4) * &(*d3 + d4) - &x3 - &x4;
        let z00 = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue(x4) + &x0;
        [z00, x3, x34, x03, x04]
    }

    /// [`Self::mul_by_014`] specialized to `c4 = 1` (an M-twist fixed-Q line
    /// normalized so its `P.y` slot is one), with the `c4` products replaced by additions.
    /// The Karatsuba form mirrors gnark-crypto
    /// [`MulBy01`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12_pairing.go#L70-L89).
    pub fn mul_by_014_c4_one(&mut self, c0: &Fp2<Fp2Config<P>>, c1: &Fp2<Fp2Config<P>>) {
        if !reduces_six_products::<Fp2Config<P>>() {
            return self.mul_by_014_c4_one_karatsuba(c0, c1);
        }
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let (l0, l1, l1x) = (fp2_operand(c0), fp2_operand(c1), fp2_operand(&nr(*c1)));
        let (a, b) = (self.c0, self.c1);
        self.c0 = Fp6::new(
            sum_of_two_fp2_products([&a.c0, &a.c2], [&l0, &l1x]) + nr(b.c1),
            sum_of_two_fp2_products([&a.c0, &a.c1], [&l1, &l0]) + nr(b.c2),
            sum_of_two_fp2_products([&a.c1, &a.c2], [&l1, &l0]) + b.c0,
        );
        self.c1 = Fp6::new(
            sum_of_two_fp2_products([&b.c0, &b.c2], [&l0, &l1x]) + nr(a.c2),
            sum_of_two_fp2_products([&b.c0, &b.c1], [&l1, &l0]) + a.c0,
            sum_of_two_fp2_products([&b.c1, &b.c2], [&l1, &l0]) + a.c1,
        );
    }

    fn mul_by_014_c4_one_karatsuba(&mut self, c0: &Fp2<Fp2Config<P>>, c1: &Fp2<Fp2Config<P>>) {
        let mut aa = self.c0;
        aa.mul_by_01(c0, c1);
        // bb = self.c1 * (0, 1, 0) = self.c1 * v.
        let mut bb = self.c1;
        P::mul_fp6_by_nonresidue_in_place(&mut bb);
        let mut o = *c1;
        o += <Fp2<Fp2Config<P>> as num_traits::One>::one();
        self.c1 += &self.c0;
        self.c1.mul_by_01(c0, &o);
        self.c1 -= &aa;
        self.c1 -= &bb;
        self.c0 = bb;
        P::mul_fp6_by_nonresidue_in_place(&mut self.c0);
        self.c0 += &aa;
    }

    /// [`Self::mul_by_034`] specialized to `c0 = 1` (a D-twist fixed-Q line
    /// normalized so its `P.y` slot is one), with the `c0` products replaced by additions.
    /// The Karatsuba form mirrors gnark-crypto
    /// [`MulBy34`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/internal/fptower/e12_pairing.go#L112-L130).
    pub fn mul_by_034_c0_one(&mut self, c3: &Fp2<Fp2Config<P>>, c4: &Fp2<Fp2Config<P>>) {
        if !reduces_six_products::<Fp2Config<P>>() {
            return self.mul_by_034_c0_one_karatsuba(c3, c4);
        }
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let (l3, l4) = (fp2_operand(c3), fp2_operand(c4));
        let (l3x, l4x) = (fp2_operand(&nr(*c3)), fp2_operand(&nr(*c4)));
        let (a, b) = (self.c0, self.c1);
        self.c0 = Fp6::new(
            sum_of_two_fp2_products([&b.c1, &b.c2], [&l4x, &l3x]) + a.c0,
            sum_of_two_fp2_products([&b.c0, &b.c2], [&l3, &l4x]) + a.c1,
            sum_of_two_fp2_products([&b.c0, &b.c1], [&l4, &l3]) + a.c2,
        );
        self.c1 = Fp6::new(
            sum_of_two_fp2_products([&a.c0, &a.c2], [&l3, &l4x]) + b.c0,
            sum_of_two_fp2_products([&a.c0, &a.c1], [&l4, &l3]) + b.c1,
            sum_of_two_fp2_products([&a.c1, &a.c2], [&l4, &l3]) + b.c2,
        );
    }

    fn mul_by_034_c0_one_karatsuba(&mut self, c3: &Fp2<Fp2Config<P>>, c4: &Fp2<Fp2Config<P>>) {
        let a = self.c0; // self.c0 * 1
        let mut b = self.c1;
        b.mul_by_01(c3, c4);
        let c0 = *c3 + <Fp2<Fp2Config<P>> as num_traits::One>::one();
        let mut e = self.c0 + &self.c1;
        e.mul_by_01(&c0, c4);
        self.c1 = e - &(a + &b);
        self.c0 = b;
        P::mul_fp6_by_nonresidue_in_place(&mut self.c0);
        self.c0 += &a;
    }

    /// Multiply by two D-twist line elements `c` and `d`. Two [`Self::mul_by_034`] when
    /// `sum_of_products` takes six products at once, [`Self::mul_034_by_034`] and
    /// [`Self::mul_by_01234`] otherwise.
    pub fn mul_by_034_pair(
        &mut self,
        c0: &Fp2<Fp2Config<P>>,
        c3: &Fp2<Fp2Config<P>>,
        c4: &Fp2<Fp2Config<P>>,
        d0: &Fp2<Fp2Config<P>>,
        d3: &Fp2<Fp2Config<P>>,
        d4: &Fp2<Fp2Config<P>>,
    ) {
        if reduces_six_products::<Fp2Config<P>>() {
            self.mul_by_034(c0, c3, c4);
            self.mul_by_034(d0, d3, d4);
        } else {
            self.mul_by_01234(&Self::mul_034_by_034(c0, c3, c4, d0, d3, d4));
        }
    }

    /// Multiply by the sparse element with `c0 = (x0, x1, x2)` and
    /// `c1 = (x3, x4, 0)`, the shape [`Self::mul_034_by_034`] returns. Karatsuba over
    /// `w` with two full `Fp6` products and one `Fp6::mul_by_01`, 17 `Fp2` products.
    pub fn mul_by_01234(&mut self, x: &[Fp2<Fp2Config<P>>; 5]) {
        let g0 = Fp6::new(x[0], x[1], x[2]);
        let sum_g = Fp6::new(x[0] + &x[3], x[1] + &x[4], x[2]);
        let mut b = self.c0;
        b *= &g0;
        let mut c = self.c1;
        c.mul_by_01(&x[3], &x[4]);
        let mut a = self.c0 + &self.c1;
        a *= &sum_g;
        self.c1 = a - &b - &c;
        let mut c0 = c;
        P::mul_fp6_by_nonresidue_in_place(&mut c0);
        self.c0 = c0 + &b;
    }
}

/// Karabina's compressed form of an element of the cyclotomic subgroup: the coordinates
/// `(g1, g2, g3, g5) = (c0.c1, c0.c2, c1.c0, c1.c2)`, with `g0 = c0.c0` and `g4 = c1.c1`
/// recovered by [`Self::decompress_pair`]. Squarings chain without decompressing.
/// Karabina, "Squaring in cyclotomic subgroups", <https://eprint.iacr.org/2010/542>,
/// Theorem 3.2; after gnark-crypto
/// [`CyclotomicSquareCompressed`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12.go#L142-L213),
/// [`DecompressKarabina`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12.go#L226-L281) and
/// [`BatchDecompressKarabina`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12.go#L296-L367).
#[derive(educe::Educe)]
#[educe(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressedCyclotomic<P: Fp12Config> {
    pub g1: Fp2<Fp2Config<P>>,
    pub g2: Fp2<Fp2Config<P>>,
    pub g3: Fp2<Fp2Config<P>>,
    pub g5: Fp2<Fp2Config<P>>,
}

impl<P: Fp12Config> Fp12<P> {
    /// The [`CompressedCyclotomic`] form of `self`, which must be in the cyclotomic subgroup.
    pub fn compress_cyclotomic(&self) -> CompressedCyclotomic<P> {
        CompressedCyclotomic {
            g1: self.c0.c1,
            g2: self.c0.c2,
            g3: self.c1.c0,
            g5: self.c1.c2,
        }
    }
}

impl<P: Fp12Config> CompressedCyclotomic<P> {
    /// Squares in place with two `Fp4` squarings of 2 `Fp2` multiplications each:
    /// `(A, A') = (g3 + g2 y)^2`, `(B, B') = (g1 + g5 y)^2`, then `g1 = 3A - 2 g1`,
    /// `g2 = 3B - 2 g2`, `g3 = 3 xi B' + 2 g3`, `g5 = 3A' + 2 g5`.
    pub fn square_in_place(&mut self) {
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        // (a0 + a1 y)^2 = (a0^2 + xi a1^2) + 2 a0 a1 y, with y^2 = xi.
        let fp4_square = |a0: Fp2<Fp2Config<P>>, a1: Fp2<Fp2Config<P>>| {
            let t = a0 * a1;
            ((a0 + a1) * (nr(a1) + a0) - t - nr(t), t.double())
        };
        let (a, a1) = fp4_square(self.g3, self.g2);
        let (b, b1) = fp4_square(self.g1, self.g5);
        let b1 = nr(b1);
        self.g1 = a.double() + a - self.g1.double();
        self.g2 = b.double() + b - self.g2.double();
        self.g3 = b1.double() + b1 + self.g3.double();
        self.g5 = a1.double() + a1 + self.g5.double();
    }

    /// Decompresses two elements with one `Fp2` inversion:
    /// `g4 = (xi g5^2 + 3 g1^2 - 2 g2) / (4 g3)`, `g0 = xi (2 g4^2 + g3 g5 - 3 g1 g2) + 1`.
    /// Returns `None` when either `g3` is zero.
    pub fn decompress_pair(a: &Self, b: &Self) -> Option<(Fp12<P>, Fp12<P>)> {
        let den_a = a.g3.double().double();
        let den_b = b.g3.double().double();
        let inv = (den_a * den_b).inverse()?;
        Some((
            a.decompress_with_g4(a.g4_numerator() * (den_b * inv)),
            b.decompress_with_g4(b.g4_numerator() * (den_a * inv)),
        ))
    }

    fn g4_numerator(&self) -> Fp2<Fp2Config<P>> {
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let g1_sq = self.g1.square();
        nr(self.g5.square()) + g1_sq.double() + g1_sq - self.g2.double()
    }

    fn decompress_with_g4(&self, g4: Fp2<Fp2Config<P>>) -> Fp12<P> {
        let nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;
        let g1_g2 = self.g1 * self.g2;
        let g0 = nr(g4.square().double() + self.g3 * self.g5 - g1_g2.double() - g1_g2)
            + <Fp2<Fp2Config<P>> as num_traits::One>::one();
        Fp12::new(
            Fp6::new(g0, self.g1, self.g2),
            Fp6::new(self.g3, g4, self.g5),
        )
    }
}

pub const fn characteristic_square_mod_6_is_one(characteristic: &[u64]) -> bool {
    // char mod 6 = (a_0 + 2**64 * a_1 + ...) mod 6
    //            = a_0 mod 6 + (2**64 * a_1 mod 6) + (...) mod 6
    //            = a_0 mod 6 + (4 * a_1 mod 6) + (4 * ...) mod 6
    let mut char_mod_6 = 0u64;
    crate::const_for!((i in 0..(characteristic.len())) {
        char_mod_6 += if i == 0 {
            characteristic[i] % 6
        } else {
            (4 * (characteristic[i] % 6)) % 6
        };
    });
    (char_mod_6 * char_mod_6) % 6 == 1
}

impl<P: Fp12Config> CyclotomicMultSubgroup for Fp12<P> {
    const INVERSE_IS_FAST: bool = true;

    fn cyclotomic_inverse_in_place(&mut self) -> Option<&mut Self> {
        self.is_zero().not().then(|| self.conjugate_in_place())
    }

    fn cyclotomic_square_in_place(&mut self) -> &mut Self {
        // Faster Squaring in the Cyclotomic Subgroup of Sixth Degree Extensions
        // - Robert Granger and Michael Scott
        //
        if characteristic_square_mod_6_is_one(Self::characteristic()) {
            let fp2_nr = <P::Fp6Config as Fp6Config>::mul_fp2_by_nonresidue;

            let r0 = &self.c0.c0;
            let r4 = &self.c0.c1;
            let r3 = &self.c0.c2;
            let r2 = &self.c1.c0;
            let r1 = &self.c1.c1;
            let r5 = &self.c1.c2;

            // t0 + t1*y = (z0 + z1*y)^2 = a^2
            let mut tmp = *r0 * r1;
            let t0 = (*r0 + r1) * &(fp2_nr(*r1) + r0) - &tmp - &fp2_nr(tmp);
            let t1 = tmp.double();

            // t2 + t3*y = (z2 + z3*y)^2 = b^2
            tmp = *r2 * r3;
            let t2 = (*r2 + r3) * &(fp2_nr(*r3) + r2) - &tmp - &fp2_nr(tmp);
            let t3 = tmp.double();

            // t4 + t5*y = (z4 + z5*y)^2 = c^2
            tmp = *r4 * r5;
            let t4 = (*r4 + r5) * &(fp2_nr(*r5) + r4) - &tmp - &fp2_nr(tmp);
            let t5 = tmp.double();

            let z0 = &mut self.c0.c0;
            let z4 = &mut self.c0.c1;
            let z3 = &mut self.c0.c2;
            let z2 = &mut self.c1.c0;
            let z1 = &mut self.c1.c1;
            let z5 = &mut self.c1.c2;

            // for A

            // z0 = 3 * t0 - 2 * z0
            *z0 = t0 - &*z0;
            z0.double_in_place();
            *z0 += &t0;

            // z1 = 3 * t1 + 2 * z1
            *z1 = t1 + &*z1;
            z1.double_in_place();
            *z1 += &t1;

            // for B

            // z2 = 3 * (xi * t5) + 2 * z2
            tmp = fp2_nr(t5);
            *z2 += tmp;
            z2.double_in_place();
            *z2 += &tmp;

            // z3 = 3 * t4 - 2 * z3
            *z3 = t4 - &*z3;
            z3.double_in_place();
            *z3 += &t4;

            // for C

            // z4 = 3 * t2 - 2 * z4
            *z4 = t2 - &*z4;
            z4.double_in_place();
            *z4 += &t2;

            // z5 = 3 * t3 + 2 * z5
            *z5 += t3;
            z5.double_in_place();
            *z5 += &t3;
            self
        } else {
            self.square_in_place()
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_characteristic_square_mod_6_is_one() {
        use super::*;
        assert!(!characteristic_square_mod_6_is_one(&[36]));
        assert!(characteristic_square_mod_6_is_one(&[37]));
        assert!(!characteristic_square_mod_6_is_one(&[38]));
        assert!(!characteristic_square_mod_6_is_one(&[39]));
        assert!(!characteristic_square_mod_6_is_one(&[40]));
        assert!(characteristic_square_mod_6_is_one(&[41]));

        assert!(!characteristic_square_mod_6_is_one(&[36, 36]));
        assert!(!characteristic_square_mod_6_is_one(&[36, 37]));
        assert!(!characteristic_square_mod_6_is_one(&[36, 38]));
        assert!(!characteristic_square_mod_6_is_one(&[36, 39]));
        assert!(!characteristic_square_mod_6_is_one(&[36, 40]));
        assert!(!characteristic_square_mod_6_is_one(&[36, 41]));

        assert!(!characteristic_square_mod_6_is_one(&[36, 41]));
        assert!(!characteristic_square_mod_6_is_one(&[37, 41]));
        assert!(!characteristic_square_mod_6_is_one(&[38, 41]));
        assert!(characteristic_square_mod_6_is_one(&[39, 41]));
        assert!(!characteristic_square_mod_6_is_one(&[40, 41]));
        assert!(characteristic_square_mod_6_is_one(&[41, 41]));
        assert!(characteristic_square_mod_6_is_one(&[1, u64::MAX]));
    }
}

#[cfg(test)]
mod sparse_line_product_tests {
    use ark_std::{test_rng, UniformRand};
    use ark_test_curves::bls12_381::{Fq12, Fq2};

    // Both identities are purely algebraic in Fp12, independent of any curve's
    // actual twist, so a single Fp12 instance exercises both families.
    #[test]
    fn mul_014_by_014_matches_sequential() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let f = Fq12::rand(&mut rng);
            let l = [Fq2::rand(&mut rng), Fq2::rand(&mut rng), Fq2::rand(&mut rng)];
            let m = [Fq2::rand(&mut rng), Fq2::rand(&mut rng), Fq2::rand(&mut rng)];
            let mut seq = f;
            seq.mul_by_014(&l[0], &l[1], &l[2]);
            seq.mul_by_014(&m[0], &m[1], &m[2]);
            let prod = Fq12::mul_014_by_014(&l[0], &l[1], &l[2], &m[0], &m[1], &m[2]);
            let mut batched = f;
            batched.mul_by_01245(&prod);
            assert_eq!(seq, batched);
        }
    }

    #[test]
    fn mul_by_c_one_matches_generic() {
        use ark_std::One;
        let mut rng = test_rng();
        for _ in 0..100 {
            let f = Fq12::rand(&mut rng);
            let c0 = Fq2::rand(&mut rng);
            let c1 = Fq2::rand(&mut rng);
            let mut a = f;
            a.mul_by_014_c4_one(&c0, &c1);
            let mut b = f;
            b.mul_by_014(&c0, &c1, &Fq2::one());
            assert_eq!(a, b);

            let c3 = Fq2::rand(&mut rng);
            let c4 = Fq2::rand(&mut rng);
            let mut a = f;
            a.mul_by_034_c0_one(&c3, &c4);
            let mut b = f;
            b.mul_by_034(&Fq2::one(), &c3, &c4);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn sparse_products_match_full_multiplication() {
        use ark_std::One;
        use ark_test_curves::{ark_ff::AdditiveGroup, bls12_381::{Fq, Fq6 as Fp6}};
        let mut rng = test_rng();
        let max = Fq2::new(-Fq::one(), -Fq::one());
        let max12 = Fq12::new(Fp6::new(max, max, max), Fp6::new(max, max, max));
        let mut cases: ark_std::vec::Vec<_> = (0..100)
            .map(|_| {
                let f = Fq12::rand(&mut rng);
                (f, [Fq2::rand(&mut rng), Fq2::rand(&mut rng), Fq2::rand(&mut rng)])
            })
            .collect();
        cases.push((max12, [max; 3]));
        for (f, l) in cases {
            let z = Fq2::ZERO;
            let mut a = f;
            a.mul_by_014(&l[0], &l[1], &l[2]);
            assert_eq!(a, f * Fq12::new(Fp6::new(l[0], l[1], z), Fp6::new(z, l[2], z)));
            let mut a = f;
            a.mul_by_034(&l[0], &l[1], &l[2]);
            assert_eq!(a, f * Fq12::new(Fp6::new(l[0], z, z), Fp6::new(l[1], l[2], z)));
        }
    }

    #[test]
    fn fp6_products_match_karatsuba() {
        use ark_std::{One, Zero};
        use ark_test_curves::bls12_381::{Fq, Fq6};
        let mut rng = test_rng();
        let max = Fq2::new(-Fq::one(), -Fq::one());
        let mut cases: ark_std::vec::Vec<_> =
            (0..200).map(|_| (Fq6::rand(&mut rng), Fq6::rand(&mut rng))).collect();
        cases.push((Fq6::new(max, max, max), Fq6::new(max, max, max)));
        for (a, b) in cases {
            let mut k = a;
            k.mul_assign_karatsuba(&b);
            assert_eq!(a * b, k);
            let mut s = a;
            s.mul_by_12(&b.c1, &b.c2);
            let mut k = a;
            k.mul_assign_karatsuba(&Fq6::new(Fq2::zero(), b.c1, b.c2));
            assert_eq!(s, k);
        }
    }

    #[test]
    fn sum_of_products_seven_terms() {
        use ark_std::One;
        use ark_test_curves::Field;
        use ark_test_curves::bls12_381::Fq;
        let mut rng = test_rng();
        let max = -Fq::one();
        let naive = |a: &[Fq; 7], b: &[Fq; 7]| a.iter().zip(b).map(|(x, y)| *x * y).sum::<Fq>();
        assert_eq!(Fq::sum_of_products(&[max; 7], &[max; 7]), naive(&[max; 7], &[max; 7]));
        for _ in 0..1000 {
            let a: [Fq; 7] = core::array::from_fn(|_| Fq::rand(&mut rng));
            let b: [Fq; 7] = core::array::from_fn(|_| Fq::rand(&mut rng));
            assert_eq!(Fq::sum_of_products(&a, &b), naive(&a, &b));
        }
    }

    #[test]
    fn mul_034_by_034_matches_sequential() {
        let mut rng = test_rng();
        for _ in 0..100 {
            let f = Fq12::rand(&mut rng);
            let l = [Fq2::rand(&mut rng), Fq2::rand(&mut rng), Fq2::rand(&mut rng)];
            let m = [Fq2::rand(&mut rng), Fq2::rand(&mut rng), Fq2::rand(&mut rng)];
            let mut seq = f;
            seq.mul_by_034(&l[0], &l[1], &l[2]);
            seq.mul_by_034(&m[0], &m[1], &m[2]);
            let prod = Fq12::mul_034_by_034(&l[0], &l[1], &l[2], &m[0], &m[1], &m[2]);
            let mut batched = f;
            batched.mul_by_01234(&prod);
            assert_eq!(seq, batched);
        }
    }
}
