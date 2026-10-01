//! Four-dimensional GLS scalar multiplication on the G2 of a BN or BLS12 curve. On G2 the
//! untwist-Frobenius-twist endomorphism `psi` acts as `[p mod r]`, and `psi^4 - psi^2 + 1 = 0`
//! there because `r` divides `\Phi_{12}(p) = p^4 - p^2 + 1`. So a scalar `k` splits into four
//! digits of about `r^{1/4}`, 64 bits for BN254 and BLS12-381, with
//! `k = \sum_i k_i p^i (mod r)`, and `[k]P = \sum_i [k_i] psi^i(P)`. That halves the doublings of
//! the two-dimensional GLV split (about 64 instead of 128). The additions stay cheap because
//! `psi` maps an affine point to an affine point with one Frobenius and two `Fp2` products, so
//! the odd-multiple tables of `psi(P)`, `psi^2(P)` and `psi^3(P)` come from the one table of `P`.
//! It beats two-dimensional GLV at every scalar width from 16 bits up. Full-width scalars go from
//! 135 to 104 us on BLS12-381 G2 and from 72 to 55 us on BN254 G2, 128-bit scalars from 131 to
//! 79 us and from 71 to 54 us, and 16-bit scalars from 31 to 27 us and from 15 to 14 us.
//! Galbraith, Scott, [Exponentiation in pairing-friendly groups using
//! homomorphisms](https://eprint.iacr.org/2008/117), Lemma 1 in section 5 (`psi` on G2),
//! examples 3 and 5 in section 6 (the BLS12 and BN splits), and section 7 (interleaved w-NAF).

use crate::{
    scalar_mul::{double_and_add, double_and_add_affine, glv::scalar_below_modulus},
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    AdditiveGroup, AffineRepr, CurveGroup,
};
use ark_std::{vec::Vec, Zero};

/// Width of the NAF of each digit. The table of `P` holds `2^{W-2}` odd multiples, and each
/// digit has about `64 / (W + 1)` nonzero terms. On full-width scalars `W = 5` ties `W = 4` on
/// BLS12-381 (103 us) and beats it on BN254 (57 us against 61 us). `W = 6` loses on both
/// (119 us and 65 us) to its 16-point table.
const W: u32 = 5;

/// `[k]P` by [`gls4_mul`] for `k < r`, with the digits of `k` from `digits`, and by
/// `double_and_add` for `k >= r` or when `digits` returns `None`. `double_and_add` is exact on
/// every curve point. The split applies `psi` wherever a digit other than `k_0` is nonzero and is
/// exact there only where `psi` acts as `[p mod r]`, i.e. on the order-`r` subgroup. On BN254 and
/// BLS12-381 every `k < 2^63` splits as `(k, 0, 0, 0)`, so those scalars are exact on every curve
/// point too.
pub fn gls4_mul_bigint<P: SWCurveConfig>(
    p: &Projective<P>,
    k: &[u64],
    digits: impl FnOnce(&[u64]) -> Option<[(bool, u64); 4]>,
    psi: impl Fn(&Affine<P>) -> Affine<P>,
) -> Projective<P> {
    match scalar_below_modulus::<P::ScalarField>(k).and_then(|_| digits(k)) {
        Some(d) => gls4_mul(p, &d, psi),
        None => double_and_add(p, k),
    }
}

/// [`gls4_mul_bigint`] for an affine base, whose `k >= r` fallback is `double_and_add_affine`
/// with mixed additions.
pub fn gls4_mul_affine_bigint<P: SWCurveConfig>(
    p: &Affine<P>,
    k: &[u64],
    digits: impl FnOnce(&[u64]) -> Option<[(bool, u64); 4]>,
    psi: impl Fn(&Affine<P>) -> Affine<P>,
) -> Projective<P> {
    match scalar_below_modulus::<P::ScalarField>(k).and_then(|_| digits(k)) {
        Some(d) => gls4_mul(&p.into_group(), &d, psi),
        None => double_and_add_affine(p, k),
    }
}

/// `\sum_i [k_i] psi^i(P)` for four signed digits `k_i = (negative, magnitude)`, by interleaved
/// width-`W` NAFs over affine tables of odd multiples with mixed additions.
pub fn gls4_mul<P: SWCurveConfig>(
    p: &Projective<P>,
    digits: &[(bool, u64); 4],
    psi: impl Fn(&Affine<P>) -> Affine<P>,
) -> Projective<P> {
    // Tables up to the last nonzero digit only: a 128-bit scalar on BLS12-381 has two.
    let Some(top) = digits.iter().rposition(|&(_, k)| k != 0) else {
        return Projective::zero();
    };
    if p.is_zero() {
        return Projective::zero();
    }
    // `[1, 3, ..., 2^{W-1} - 1] P`, then the same multiples of `psi^i(P)`.
    let size = 1 << (W - 2);
    let two_p = p.double();
    let mut odd = Vec::with_capacity(size);
    odd.push(*p);
    for j in 1..size {
        odd.push(odd[j - 1] + two_p);
    }
    let mut tables: [Vec<Affine<P>>; 4] = Default::default();
    tables[0] = Projective::normalize_batch(&odd);
    for i in 1..=top {
        tables[i] = tables[i - 1].iter().map(&psi).collect();
    }

    let nafs = digits.map(|(_, k)| wnaf_u64(k));
    let len = nafs.iter().map(|(_, len)| *len).max().unwrap_or(0);
    let mut acc = Projective::<P>::zero();
    for j in (0..len).rev() {
        acc.double_in_place();
        for i in 0..=top {
            let d = nafs[i].0[j];
            if d != 0 {
                let t = &tables[i][usize::from(d.unsigned_abs() / 2)];
                if (d < 0) != digits[i].0 {
                    acc -= t;
                } else {
                    acc += t;
                }
            }
        }
    }
    acc
}

/// Width-`W` NAF digits of `k`, least significant first, and their count. Each nonzero digit is
/// odd with absolute value below `2^{W-1}`, and there are at most 65 digits.
fn wnaf_u64(k: u64) -> ([i8; 65], usize) {
    let mut digits = [0i8; 65];
    let mut k = u128::from(k);
    let mut len = 0;
    while k != 0 {
        if k & 1 == 1 {
            let low = (k & ((1 << W) - 1)) as i16;
            let d = if low >= 1 << (W - 1) {
                low - (1 << W)
            } else {
                low
            };
            digits[len] = d as i8;
            k = (k as i128 - i128::from(d)) as u128;
        }
        k >>= 1;
        len += 1;
    }
    (digits, len)
}

#[cfg(test)]
mod tests {
    use super::wnaf_u64;

    #[test]
    fn wnaf_u64_reconstructs() {
        for k in [
            0u64,
            1,
            2,
            15,
            16,
            31,
            1 << 63,
            u64::MAX,
            u64::MAX - 1,
            0xd201000000010000,
        ] {
            let (digits, len) = wnaf_u64(k);
            let mut sum = 0i128;
            for j in (0..len).rev() {
                sum = 2 * sum + i128::from(digits[j]);
                assert!(digits[j] == 0 || digits[j] % 2 != 0);
                assert!(digits[j].unsigned_abs() < 1 << 4);
            }
            assert_eq!(sum, i128::from(k), "k = {k:#x}");
            assert!(digits[..len].windows(super::W as usize).all(|w| w
                .iter()
                .filter(|&&d| d != 0)
                .count()
                <= 1));
        }
    }
}
