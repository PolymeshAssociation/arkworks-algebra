use ark_ec::bls12::{Bls12, Bls12Config, TwistType};
use ark_ff::{
    fields::fp12_2over3over2::CompressedCyclotomic, AdditiveGroup, CyclotomicMultSubgroup,
};

use crate::{Fq, Fq12, Fq12Config, Fq2, Fq2Config, Fq6Config};

pub mod g1;
pub mod g2;
pub(crate) mod util;

mod g1_swu_iso;
mod g2_swu_iso;

#[cfg(test)]
mod tests;

pub use self::{
    g1::{G1Affine, G1Projective},
    g2::{G2Affine, G2Projective},
};

pub type Bls12_381 = Bls12<Config>;

pub struct Config;

impl Bls12Config for Config {
    const X: &'static [u64] = &[0xd201000000010000];
    const X_IS_NEGATIVE: bool = true;
    const TWIST_TYPE: TwistType = TwistType::M;
    type Fp = Fq;
    type Fp2Config = Fq2Config;
    type Fp6Config = Fq6Config;
    type Fp12Config = Fq12Config;
    type G1Config = self::g1::Config;
    type G2Config = self::g2::Config;

    /// `b' = 4*(1 + u)`, so `3*b'*c = 12*(1 + u)*c`, and `(1 + u)*c = (c0 - c1,
    /// c0 + c1)` because `u^2 = -1`. The `(1 + u)*c` step is gnark-crypto's
    /// [`MulBybTwistCurveCoeff`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e2_bls381.go#L85-L94).
    fn mul_by_3b_twist(c: Fq2) -> Fq2 {
        let t = Fq2::new(c.c0 - c.c1, c.c0 + c.c1);
        let four = t.double().double();
        four.double() + four
    }

    /// `f^x` for `|x| = 0xd201000000010000`, `x < 0`, through
    /// `f^(|x|/2)` with `|x|/2 = 2^62 + 2^61 + 2^59 + 2^56 + 2^47 + 2^15`, after
    /// gnark-crypto [`ExptHalf`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bls12-381/internal/fptower/e12_pairing.go#L17-L36).
    /// The runs of 15 and 32 squarings are compressed ([`CompressedCyclotomic`]) and the
    /// two checkpoints share one `Fq2` inversion; the remaining 16 squarings are
    /// Granger-Scott. Falls back to `exp_by_x_chain` when a checkpoint has `g3 = 0`.
    fn exp_by_x(f: Fq12) -> Fq12 {
        let mut c = f.compress_cyclotomic();
        for _ in 0..15 {
            c.square_in_place();
        }
        let first = c;
        for _ in 0..32 {
            c.square_in_place();
        }
        let Some((mut r, mut t)) = CompressedCyclotomic::decompress_pair(&first, &c) else {
            return exp_by_x_chain(f);
        };
        r *= &t; // 2^15 + 2^47
        for squarings in [9, 3, 2, 1] {
            for _ in 0..squarings {
                t.cyclotomic_square_in_place();
            }
            r *= &t; // + 2^56, 2^59, 2^61, 2^62 = |x|/2
        }
        r.cyclotomic_inverse_in_place(); // x is negative
        r.cyclotomic_square_in_place();
        r
    }
}

/// `f^x` by a fixed addition chain after blst's `raise_to_z` / `raise_to_z_div_by_2`
/// (<https://github.com/supranational/blst/blob/v0.3.17/src/pairing.c#L355-L366>): 63 cyclotomic squarings,
/// 5 multiplications, one conjugation. The chain reconstructs `|x|` as
/// `0x2 -> 0xc -> 0x68 -> 0xd200 -> 0xd20100000000 -> 0xd201000000010000`.
fn exp_by_x_chain(f: Fq12) -> Fq12 {
    fn mul_then_square(r: &mut Fq12, f: &Fq12, squarings: usize) {
        *r *= f;
        for _ in 0..squarings {
            r.cyclotomic_square_in_place();
        }
    }
    let mut r = f.cyclotomic_square(); // 0x2
    mul_then_square(&mut r, &f, 2); // 0xc
    mul_then_square(&mut r, &f, 3); // 0x68
    mul_then_square(&mut r, &f, 9); // 0xd200
    mul_then_square(&mut r, &f, 32); // 0xd20100000000
    mul_then_square(&mut r, &f, 16); // 0xd201000000010000 = |x|
    r.cyclotomic_inverse_in_place(); // x is negative
    r
}
