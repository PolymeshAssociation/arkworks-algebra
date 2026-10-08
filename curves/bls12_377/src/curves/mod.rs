use ark_ec::{
    bls12,
    bls12::{Bls12, Bls12Config, TwistType},
};
use ark_ff::{AdditiveGroup, CyclotomicMultSubgroup, MontFp};

use crate::*;

pub mod g1;
pub mod g2;

mod g1_swu_iso;
mod g2_swu_iso;

#[cfg(test)]
mod tests;

pub struct Config;

impl Bls12Config for Config {
    const X: &'static [u64] = &[0x8508c00000000001];
    /// `x` is positive.
    const X_IS_NEGATIVE: bool = false;
    const TWIST_TYPE: TwistType = TwistType::D;
    type Fp = Fq;
    type Fp2Config = Fq2Config;
    type Fp6Config = Fq6Config;
    type Fp12Config = Fq12Config;
    type G1Config = g1::Config;
    type G2Config = g2::Config;

    /// `3 b' c` for `b' = 1/u = -u/5`: `(3 c1, (-3/5) c0)`, one `Fq` multiplication.
    /// After gnark-crypto
    /// [`MulBybTwistCurveCoeff`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-377/internal/fptower/e2_bls377.go#L95-L103).
    fn mul_by_3b_twist(c: Fq2) -> Fq2 {
        const NEG_THREE_OVER_FIVE: Fq = MontFp!("206931540810375275208522186955914826829114810203931728431907410133376374678672658219975110511658688099552257166541");
        Fq2::new(c.c1.double() + c.c1, c.c0 * NEG_THREE_OVER_FIVE)
    }

    /// `f^x` for `x = 0x8508c00000000001 = 136227 * 2^46 + 1`: a shortest addition
    /// chain for 136227, then 46 cyclotomic squarings and one multiplication. 63
    /// squarings and 5 multiplications. gnark-crypto
    /// [`Expt`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-377/internal/fptower/e12_pairing.go#L16-L43).
    fn exp_by_x(f: Fq12) -> Fq12 {
        fn square_n(r: &mut Fq12, n: usize) {
            for _ in 0..n {
                r.cyclotomic_square_in_place();
            }
        }
        let mut r = f;
        square_n(&mut r, 5);
        r *= &f; // 0x21
        let x33 = r;
        square_n(&mut r, 7);
        r *= &x33; // 0x10a1
        square_n(&mut r, 4);
        r *= &f; // 0x10a11
        square_n(&mut r, 1);
        r *= &f; // 0x21423 = 136227
        square_n(&mut r, 46);
        r *= &f;
        r
    }
}

pub type Bls12_377 = Bls12<Config>;

pub type G1Affine = bls12::G1Affine<Config>;
pub type G1Projective = bls12::G1Projective<Config>;
pub type G2Affine = bls12::G2Affine<Config>;
pub type G2Projective = bls12::G2Projective<Config>;

pub use g1::{G1TEAffine, G1TEProjective};
