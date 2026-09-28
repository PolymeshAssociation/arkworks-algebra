use ark_ec::{
    bn,
    bn::{Bn, BnConfig, GtGlsParams, TwistType},
};
use ark_ff::{CyclotomicMultSubgroup, MontFp};

use crate::*;

pub mod g1;
pub mod g2;

/// Addition-subtraction chain for `x = 4965661367192848881 = 0x44e992b44a6909f1`, from
/// gnark-crypto [`Expt`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/internal/fptower/e12_pairing.go#L17-L88)
/// and [`mulBySeed`](https://github.com/Consensys/gnark-crypto/blob/v0.21.0/ecc/bn254/g2.go#L751-L835).
/// Start from `a^8` with the table `a, a^3, a^5, a^7`; each step `(n, d)` squares `n`
/// times and multiplies by `a^d`, inverted when `d < 0`. 60 squarings, 17 multiplications.
pub(crate) const SEED_CHAIN: [(u8, i8); 13] = [
    (3, 5),
    (5, -3),
    (4, 3),
    (5, 5),
    (4, -5),
    (4, -3),
    (4, 1),
    (5, 5),
    (5, 7),
    (4, -7),
    (7, 5),
    (5, -1),
    (4, 1),
];

#[cfg(test)]
mod tests;

pub struct Config;

impl BnConfig for Config {
    const X: &'static [u64] = &[4965661367192848881];
    /// `x` is positive.
    const X_IS_NEGATIVE: bool = false;
    const ATE_LOOP_COUNT: &'static [i8] = &[
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, 1, 1,
    ];

    const TWIST_MUL_BY_Q_X: Fq2 = Fq2::new(
        MontFp!("21575463638280843010398324269430826099269044274347216827212613867836435027261"),
        MontFp!("10307601595873709700152284273816112264069230130616436755625194854815875713954"),
    );
    const TWIST_MUL_BY_Q_Y: Fq2 = Fq2::new(
        MontFp!("2821565182194536844548159561693502659359617185244120367078079554186484126554"),
        MontFp!("3505843767911556378687030309984248845540243509899259641013678093033130930403"),
    );
    const TWIST_TYPE: TwistType = TwistType::D;

    /// GT 4-dimensional GLS (Galbraith-Scott, <https://eprint.iacr.org/2008/117>):
    /// LLL-reduced basis of the lattice
    /// `{ v : v0 + v1 p + v2 p^2 + v3 p^3 == 0 (mod r) }` (rows, entries ~x), and its
    /// first adjugate row (numerator of `beta_j = round(k * adj0[j] / r)`). In Sage, with
    /// `lam = 6*x^2` (`p mod r`), `matrix(ZZ, [[r,0,0,0], [-lam,1,0,0], [-lam^2 % r,0,1,0],
    /// [-lam^3 % r,0,0,1]]).LLL()` gives these rows with the middle two swapped,
    /// `det(basis) = r`, and `adj0` is `basis.adjugate().row(0)` for the row order below.
    /// The digit bound `(1/2) sum_j |basis[j][i]|` is below `2^64` in every column.
    const GT_GLS: Option<GtGlsParams> = Some(GtGlsParams {
        basis: [
            [9931322734385697762, 4965661367192848882, -4965661367192848881, 4965661367192848881],
            [4965661367192848882, 4965661367192848881, 4965661367192848881, -9931322734385697762],
            [-4965661367192848881, 4965661367192848881, -4965661367192848881, -9931322734385697763],
            [9931322734385697763, -4965661367192848881, -4965661367192848882, -4965661367192848881],
        ],
        adj0: [
            (true, [0x113c366715dedaf5, 0xd7adf45cf590c4c8, 0x1df623ef8af183e3, 0x0]),
            (true, [0x620aaa6f726909f1, 0x46fb76a5e4491ec5, 0x1df623ef8af183e4, 0x0]),
            (false, [0xd8378506dd96f60e, 0x46fb76a5e4491ec4, 0x1df623ef8af183e4, 0x0]),
            (true, [0x934df252932dec1d, 0x46fb76a5e4491ec4, 0x1df623ef8af183e4, 0x0]),
        ],
    });
    type Fp = Fq;
    type Fp2Config = Fq2Config;
    type Fp6Config = Fq6Config;
    type Fp12Config = Fq12Config;
    type G1Config = g1::Config;
    type G2Config = g2::Config;

    /// `f^x` by [`SEED_CHAIN`], negative digits through the conjugate.
    fn exp_by_x(f: Fq12) -> Fq12 {
        let f2 = f.cyclotomic_square();
        let f3 = f2 * f;
        let f5 = f2 * f3;
        let f7 = f2 * f5;
        let table = [f, f3, f5, f7];
        let mut r = f7 * f;
        for (squarings, digit) in SEED_CHAIN {
            for _ in 0..squarings {
                r.cyclotomic_square_in_place();
            }
            let mut t = table[usize::from(digit.unsigned_abs() / 2)];
            if digit < 0 {
                t.cyclotomic_inverse_in_place();
            }
            r *= t;
        }
        r
    }
}

pub type Bn254 = Bn<Config>;

pub type G1Affine = bn::G1Affine<Config>;
pub type G1Projective = bn::G1Projective<Config>;
pub type G2Affine = bn::G2Affine<Config>;
pub type G2Projective = bn::G2Projective<Config>;
