use crate::Field;
use ark_std::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A field element in loose representation: for prime fields of the Pasta shape, on targets other
/// than wasm32 and x86-64 with its assembly kernels, an integer below twice the modulus instead of
/// below the modulus, so multiplication and squaring skip their closing conditional subtraction.
/// Every other field and target keeps its canonical representation and the operations below are the
/// ordinary ones. Comparison needs the canonical representative, so `Loose` has no `PartialEq`;
/// [`Self::reduce`] returns it.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Loose<F: Field>(F);

impl<F: Field> Loose<F> {
    /// The loose zero.
    pub const ZERO: Self = Self(F::ZERO);

    /// The canonical field element.
    #[inline(always)]
    pub fn reduce(&self) -> F {
        self.0.loose_reduce()
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.0.loose_is_zero()
    }

    #[inline(always)]
    pub fn square_in_place(&mut self) -> &mut Self {
        self.0.loose_square_in_place();
        self
    }

    #[inline(always)]
    pub fn double_in_place(&mut self) -> &mut Self {
        self.0.loose_double_in_place();
        self
    }

    #[inline(always)]
    pub fn neg_in_place(&mut self) -> &mut Self {
        self.0.loose_neg_in_place();
        self
    }

    #[inline(always)]
    pub fn square(&self) -> Self {
        let mut r = *self;
        r.square_in_place();
        r
    }

    #[inline(always)]
    pub fn double(&self) -> Self {
        let mut r = *self;
        r.double_in_place();
        r
    }

    /// `a[0] * b[0] + a[1] * b[1]` for canonical `a`.
    #[inline(always)]
    pub fn sum_of_two_products(a: &[F; 2], b: &[Self; 2]) -> Self {
        Self(F::loose_sum_of_two_products(a, &[b[0].0, b[1].0]))
    }
}

impl<F: Field> Default for Loose<F> {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field> From<F> for Loose<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F: Field> fmt::Debug for Loose<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.reduce(), f)
    }
}

impl<F: Field> Neg for Loose<F> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self {
        self.neg_in_place();
        self
    }
}

macro_rules! loose_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $hook:ident) => {
        impl<F: Field> $assign_trait<&Loose<F>> for Loose<F> {
            #[inline(always)]
            fn $assign_method(&mut self, other: &Self) {
                self.0.$hook(&other.0);
            }
        }

        impl<F: Field> $assign_trait for Loose<F> {
            #[inline(always)]
            fn $assign_method(&mut self, other: Self) {
                self.0.$hook(&other.0);
            }
        }

        impl<F: Field> $trait<&Loose<F>> for Loose<F> {
            type Output = Self;

            #[inline(always)]
            fn $method(mut self, other: &Self) -> Self {
                self.0.$hook(&other.0);
                self
            }
        }

        impl<F: Field> $trait for Loose<F> {
            type Output = Self;

            #[inline(always)]
            fn $method(mut self, other: Self) -> Self {
                self.0.$hook(&other.0);
                self
            }
        }
    };
}

loose_binop!(Add, add, AddAssign, add_assign, loose_add_assign);
loose_binop!(Sub, sub, SubAssign, sub_assign, loose_sub_assign);
loose_binop!(Mul, mul, MulAssign, mul_assign, loose_mul_assign);
