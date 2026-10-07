use ark_ec::{
    scalar_mul::double_and_add_affine,
    short_weierstrass::{Affine, SWCurveConfig},
    AffineRepr, CurveGroup, PrimeGroup,
};
use ark_ff::{Field, PrimeField, UniformRand, Zero};
use ark_std::{rand::Rng, test_rng, vec::Vec};
use num_bigint::BigUint;

fn to_biguint(limbs: &[u64]) -> BigUint {
    limbs
        .iter()
        .rev()
        .fold(BigUint::zero(), |acc, &l| (acc << 64u32) + l)
}

fn random_curve_point<P: SWCurveConfig, R: Rng>(rng: &mut R) -> Affine<P> {
    loop {
        let x = P::BaseField::rand(rng);
        if let Some(p) = Affine::<P>::get_point_from_x_unchecked(x, rng.gen()) {
            return p;
        }
    }
}

/// Cross-checks `P::is_in_correct_subgroup_assuming_on_curve` against the naive
/// `[r]P == 0` on subgroup points, random curve points, points of the cofactor group,
/// points `T` of each prime order in `cofactor_primes` (primes dividing the cofactor),
/// and sums `S + T` with `S` in the subgroup.
pub fn test_subgroup_check<P: SWCurveConfig>(cofactor_primes: &[u64], samples: usize) {
    let mut rng = test_rng();
    let r = P::ScalarField::characteristic();
    let naive = |p: &Affine<P>| double_and_add_affine(p, r).is_zero();
    let fast = |p: &Affine<P>| P::is_in_correct_subgroup_assuming_on_curve(p);
    let order = to_biguint(P::COFACTOR) * to_biguint(r);

    for _ in 0..samples {
        let s = (Affine::<P>::generator() * P::ScalarField::rand(&mut rng)).into_affine();
        assert!(fast(&s), "subgroup point rejected");

        let p = random_curve_point::<P, _>(&mut rng);
        assert_eq!(fast(&p), naive(&p), "random curve point");

        let t = double_and_add_affine(&p, r).into_affine();
        if !t.is_zero() {
            assert!(!fast(&t), "cofactor-group point accepted");
            assert!(
                !fast(&(s + t).into_affine()),
                "subgroup + cofactor point accepted"
            );
        }
    }

    for &l in cofactor_primes {
        assert!(
            (&order % l).is_zero(),
            "{l} does not divide the group order"
        );
        // `[m]R` has `l`-power order; multiplying by `l` until the next step is zero
        // gives a point of order exactly `l`.
        let mut m = order.clone();
        while (&m % l).is_zero() {
            m /= l;
        }
        let m: Vec<u64> = m.to_u64_digits();
        let mut found = 0;
        for _ in 0..(20 * samples) {
            if found == samples {
                break;
            }
            let mut t =
                double_and_add_affine(&random_curve_point::<P, _>(&mut rng), &m).into_affine();
            if t.is_zero() {
                continue;
            }
            found += 1;
            let s = (Affine::<P>::generator() * P::ScalarField::rand(&mut rng)).into_affine();
            loop {
                assert!(!fast(&t), "point of order a power of {l} accepted");
                assert!(
                    !fast(&(s + t).into_affine()),
                    "subgroup + {l}-power point accepted"
                );
                let next = double_and_add_affine(&t, [l]).into_affine();
                if next.is_zero() {
                    break;
                }
                t = next;
            }
        }
        assert!(found > 0, "no point of order {l} found");
    }
}

/// Checks affine and projective `mul_bigint` against `double_and_add_affine`.
/// - On random curve points, which need not lie in the order-`r` subgroup: `r`, `r - 1`, the
///   cofactor, `2^b - 1` and random integers below `2^64`, `2^128` and `2^b`, for `b` the scalar
///   field's bit size. `mul_bigint` takes an integer, which subgroup checks and cofactor clearing
///   apply off the subgroup, so it must be exact at every width. Endomorphism methods belong
///   behind `*`.
/// - On subgroup points: random scalars of every bit width, the width boundaries `2^w - 1` and
///   `2^w`, `0`, `1`, `r - 1`, and the integers `r`, `r + 1`, `2r`, the cofactor and one wider
///   than the scalar field. Also the identity, and `*` for random scalar field elements.
pub fn test_scalar_mul_matches_double_and_add<P: SWCurveConfig>(samples: usize) {
    let mut rng = test_rng();
    let r = P::ScalarField::characteristic();
    let r_big = to_biguint(r);
    let limbs = |x: &BigUint| x.to_u64_digits();
    let one = BigUint::from(1u64);
    let check = |p: &Affine<P>, k: &[u64], what: &str| {
        let expected = double_and_add_affine(p, k);
        assert_eq!(p.mul_bigint(k), expected, "affine, {what}, k = {k:?}");
        assert_eq!(
            p.into_group().mul_bigint(k),
            expected,
            "projective, {what}, k = {k:?}"
        );
    };
    let random_below = |rng: &mut _, bits: u32| -> BigUint {
        let k: [u64; 8] = Rng::gen(rng);
        to_biguint(&k) % (&one << bits)
    };

    let mut fixed: Vec<BigUint> = [0u64, 1, 2, 3].iter().map(|&k| BigUint::from(k)).collect();
    for w in [8u32, 32, 63, 64, 65, 127, 128, 129, 192] {
        fixed.push((&one << w) - 1u64);
        fixed.push(&one << w);
    }
    fixed.push(&r_big - 1u64);
    fixed.extend([
        r_big.clone(),
        &r_big + 1u64,
        &r_big * 2u64,
        to_biguint(P::COFACTOR),
    ]);
    fixed.push((&one << (64 * r.len() as u32)) + 5u64);

    let modulus_bits = r_big.bits() as u32;
    for _ in 0..samples {
        let p = random_curve_point::<P, _>(&mut rng);
        for k in [
            r_big.clone(),
            &r_big - 1u64,
            to_biguint(P::COFACTOR),
            (&one << modulus_bits) - 1u64,
        ] {
            check(&p, &limbs(&k), "curve point");
        }
        for w in [64, 128, modulus_bits] {
            check(&p, &limbs(&random_below(&mut rng, w)), "curve point");
        }

        let s = (Affine::<P>::generator() * P::ScalarField::rand(&mut rng)).into_affine();
        for k in &fixed {
            check(&s, &limbs(k), "subgroup point");
        }
        for w in (1..modulus_bits)
            .step_by(7)
            .chain([63, 64, 127, 128, modulus_bits - 1])
        {
            let k = random_below(&mut rng, w - 1) | (&one << (w - 1));
            check(&s, &limbs(&k), "subgroup point");
        }
        let k = P::ScalarField::rand(&mut rng);
        let expected = double_and_add_affine(&s, k.into_bigint());
        assert_eq!(s * k, expected, "affine, k = {k}");
        assert_eq!(s.into_group() * k, expected, "projective, k = {k}");
    }
    for k in &fixed {
        check(&Affine::<P>::zero(), &limbs(k), "identity");
    }
}
