/// Fields that have a cyclotomic multiplicative subgroup, and which can
/// leverage efficient inversion and squaring algorithms for elements in this subgroup.
///
/// If a field has multiplicative order p^d - 1, the cyclotomic subgroups refer to subgroups of order φ_n(p),
/// for any n < d, where φ_n is the [n-th cyclotomic polynomial](https://en.wikipedia.org/wiki/Cyclotomic_polynomial).
///
/// ## Note
///
/// Note that this trait is unrelated to the `Group` trait from the `ark_ec` crate. That trait
/// denotes an *additive* group, while this trait denotes a *multiplicative* group.
pub trait CyclotomicMultSubgroup: crate::Field {
    /// Is the inverse fast to compute? For example, in quadratic extensions, the inverse
    /// can be computed at the cost of negating one coordinate, which is much faster than
    /// standard inversion.
    /// By default this is `false`, but should be set to `true` for quadratic extensions.
    const INVERSE_IS_FAST: bool = false;

    /// Compute a square in the cyclotomic subgroup. By default this is computed using [`Field::square`](crate::Field::square), but for
    /// degree 12 extensions, this can be computed faster than normal squaring.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_square(&self) -> Self {
        let mut result = *self;
        *result.cyclotomic_square_in_place()
    }

    /// Square `self` in place. By default this is computed using
    /// [`Field::square_in_place`](crate::Field::square_in_place), but for degree 12 extensions,
    /// this can be computed faster than normal squaring.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_square_in_place(&mut self) -> &mut Self {
        self.square_in_place()
    }

    /// Compute the inverse of `self`. See [`Self::INVERSE_IS_FAST`] for details.
    /// Returns [`None`] if `self.is_zero()`, and [`Some`] otherwise.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_inverse(&self) -> Option<Self> {
        let mut result = *self;
        result.cyclotomic_inverse_in_place().copied()
    }

    /// Compute the inverse of `self`. See [`Self::INVERSE_IS_FAST`] for details.
    /// Returns [`None`] if `self.is_zero()`, and [`Some`] otherwise.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_inverse_in_place(&mut self) -> Option<&mut Self> {
        self.inverse_in_place()
    }

    /// Compute a cyclotomic exponentiation of `self` with respect to `e`.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_exp(&self, e: impl AsRef<[u64]>) -> Self {
        let mut result = *self;
        result.cyclotomic_exp_in_place(e);
        result
    }

    /// Set `self` to be the result of exponentiating `self` by `e`,
    /// using efficient cyclotomic algorithms.
    ///
    /// # Warning
    ///
    /// This method should be invoked only when `self` is in the cyclotomic subgroup.
    fn cyclotomic_exp_in_place(&mut self, e: impl AsRef<[u64]>) {
        if self.is_zero() {
            return;
        }

        if Self::INVERSE_IS_FAST {
            // Fast inverses let us use a width-`W` signed windowed method, serving
            // negative digits with the (cheap) inverse.
            wnaf_exp(self, e.as_ref());
        } else {
            exp_loop(
                self,
                crate::bits::BitIteratorBE::without_leading_zeros(e.as_ref()).map(|e| e as i8),
            )
        };
    }
}

/// Width-`W` windowed cyclotomic exponentiation for fields with a fast inverse.
/// Precomputes the `2^(W-2)` odd powers `f, f^3, ..., f^(2^(W-1) - 1)` and serves negative
/// wNAF digits by conjugating the corresponding power. Nonzero digits average `1/(W+1)`
/// of the exponent length, against `1/3` for NAF, which matters for dense exponents such as
/// the BN254 `x`.
///
/// Window-NAF exponentiation (Hankerson, Menezes, Vanstone, Guide to Elliptic
/// Curve Cryptography (2004), Algorithm 3.36). A negative wNAF digit is free
/// because inversion in the cyclotomic subgroup is a conjugation.
fn wnaf_exp<F: CyclotomicMultSubgroup>(f: &mut F, e: &[u64]) {
    const W: usize = 5;
    let wnaf = crate::biginteger::arithmetic::find_wnaf(e, W);
    if wnaf.is_empty() {
        *f = F::one();
        return;
    }

    // f2 = f^2; table[k] = f^(2k + 1).
    let mut f2 = *f;
    f2.cyclotomic_square_in_place();
    let table_len = 1usize << (W - 2);
    let mut table = ark_std::vec::Vec::with_capacity(table_len);
    table.push(*f);
    for k in 1..table_len {
        let mut t = table[k - 1];
        t *= &f2;
        table.push(t);
    }

    let mut res = F::one();
    let mut found = false;
    for &d in wnaf.iter().rev() {
        if found {
            res.cyclotomic_square_in_place();
        }
        if d != 0 {
            let idx = ((d.unsigned_abs() as usize) - 1) / 2;
            if !found {
                res = table[idx];
                if d < 0 {
                    res.cyclotomic_inverse_in_place();
                }
                found = true;
            } else if d > 0 {
                res *= &table[idx];
            } else {
                let mut inv = table[idx];
                inv.cyclotomic_inverse_in_place();
                res *= &inv;
            }
        }
    }
    *f = res;
}

/// Helper function to calculate the double-and-add loop for exponentiation.
fn exp_loop<F: CyclotomicMultSubgroup, I: Iterator<Item = i8>>(f: &mut F, e: I) {
    // If the inverse is fast and we're using naf, we compute the inverse of the base.
    // Otherwise we do nothing with the variable, so we default it to one.
    let self_inverse = if F::INVERSE_IS_FAST {
        f.cyclotomic_inverse().unwrap() // The inverse must exist because self is not zero.
    } else {
        F::one()
    };
    let mut res = F::one();
    let mut found_nonzero = false;
    for value in e {
        if found_nonzero {
            res.cyclotomic_square_in_place();
        }

        if value != 0 {
            found_nonzero = true;

            if value > 0 {
                res *= &*f;
            } else if F::INVERSE_IS_FAST {
                // only use naf if inversion is fast.
                res *= &self_inverse;
            }
        }
    }
    *f = res;
}
