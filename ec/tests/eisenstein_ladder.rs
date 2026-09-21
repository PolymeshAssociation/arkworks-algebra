//! Correctness of the synchronized batch-affine GLV ladder against the per-point ladder and an
//! independent double-and-add reference.

use ark_ec::{
    scalar_mul::glv::{
        eisenstein::{glv_mul_same_scalar, Decomposed, Table},
        GLVConfig,
    },
    short_weierstrass::{Affine, Projective},
    AdditiveGroup, AffineRepr, CurveGroup,
};
use ark_ff::{BigInteger, Field, PrimeField, UniformRand, Zero};
use ark_std::{rand::Rng, test_rng, vec::Vec};
use core::any::type_name;

macro_rules! for_each_curve {
    ($body:ident) => {{
        $body::<ark_test_curves::bls12_381::g1::Config>();
        $body::<ark_pallas::PallasConfig>();
        $body::<ark_vesta::VestaConfig>();
        $body::<ark_secp256k1::Config>();
        $body::<ark_secq256k1::Config>();
    }};
}

/// `k * p` by MSB-first double-and-add, independent of GLV.
fn naive_mul<P: GLVConfig>(p: Projective<P>, k: P::ScalarField) -> Projective<P> {
    let mut acc = Projective::<P>::zero();
    for bit in k.into_bigint().to_bits_be() {
        acc.double_in_place();
        if bit {
            acc += p;
        }
    }
    acc
}

/// The points agree, table by table, with the per-point ladder and the reference. Each point is
/// multiplied by `k`
fn check_if_tables_work<P: GLVConfig>(points: &[Projective<P>], k: P::ScalarField) {
    let curve = type_name::<P>();
    let n = points.len();
    let affine = points.iter().map(|p| p.into_affine()).collect::<Vec<_>>();
    let expected = points.iter().map(|p| naive_mul(*p, k)).collect::<Vec<_>>();

    let got = glv_mul_same_scalar::<P>(&affine, k);
    assert_eq!(
        got, expected,
        "{curve}: glv_mul_same_scalar mismatch (n={n})"
    );

    if let Some(d) = Decomposed::<P>::new(k) {
        let tables = Table::<P>::batch(points);
        let batch = Table::mul_decomposed_batch(&tables, &d);
        assert_eq!(batch.len(), n);
        for (i, (table, b)) in tables.iter().zip(&batch).enumerate() {
            assert_eq!(
                *b,
                table.mul_decomposed(&d),
                "{curve}: table {i} batch vs per-point (n={n})"
            );
            assert_eq!(*b, expected[i], "{curve}: table {i} batch vs naive (n={n})");
        }
    }
}

/// A random scalar whose GLV halves fit the recoding, so the table-level batch runs the ladder
/// rather than the decomposition-miss fallback.
fn decomposing_scalar<P: GLVConfig, R: Rng + ?Sized>(rng: &mut R) -> P::ScalarField {
    loop {
        let k = P::ScalarField::rand(rng);
        if Decomposed::<P>::new(k).is_some() {
            return k;
        }
    }
}

#[test]
fn batch_matches_per_point() {

    fn check<P: GLVConfig>() {
        let mut rng = test_rng();
        for &n in &[1usize, 8, 31, 32, 33, 100] {
            let k = decomposing_scalar::<P, _>(&mut rng);
            let mut points =
                (0..n).map(|_| Projective::<P>::rand(&mut rng)).collect::<Vec<_>>();
            check_if_tables_work(&points, k);

            // Zero point works.
            points[0] = Projective::<P>::zero();
            check_if_tables_work(&points, k);

            // Zero scalar works.
            check_if_tables_work(&points, P::ScalarField::ZERO);
        }
    }

    for_each_curve!(check);
}

// Standalone port of the one-inversion affine `2P + Q` formula, for validating the
// combined-denominator algebra directly, including the `Q = -P` case the previous two-step formula
// could not handle. The batch kernel `double_add_finish_batch` applies the same algebra staged.

/// Combined denominator `(u - x)^2 (2x + u) - (v - y)^2` and the reusable `(u - x)^2`.
fn double_add_denominator<F: Field>(x: F, y: F, u: F, v: F) -> (F, F) {
    let h = u - x;
    let r = v - y;
    let h_squared = h.square();
    (h_squared * (x.double() + u) - r.square(), h_squared)
}

/// `2P + Q` from the inverse of [`double_add_denominator`].
fn double_add_finish<F: Field>(
    x: F,
    y: F,
    u: F,
    v: F,
    h_squared: F,
    denominator_inverse: F,
) -> (F, F) {
    let h = u - x;
    let r = v - y;
    let a = y * denominator_inverse;
    let b = a * h;
    let c = b * h_squared;
    let lambda = c - r;
    let x2 = u + (b * lambda).double().double();
    let y2 = -y - (F::ONE + (a * lambda).double().double()) * (lambda + c);
    (x2, y2)
}

/// The direct one-inversion `2P + Q` formula matches native group arithmetic where its denominator
/// is nonzero, returns `P` for `Q = -P`, and reports a zero denominator for `Q = P` and `Q = -2P`.
fn direct_double_add_for<P: GLVConfig>() {
    let curve = type_name::<P>();
    let g = Affine::<P>::generator().into_group();
    let coords = |p: Projective<P>, q: Projective<P>| {
        let a = Projective::<P>::normalize_batch(&[p, q]);
        (a[0].x, a[0].y, a[1].x, a[1].y)
    };

    let mut checked = 0;
    for p_s in 1..=16u64 {
        let p = g * P::ScalarField::from(p_s);
        for q_s in 1..=16u64 {
            let q = g * P::ScalarField::from(q_s);
            let (x, y, u, v) = coords(p, q);
            let (den, h2) = double_add_denominator(x, y, u, v);
            let Some(inv) = den.inverse() else { continue };
            let (ox, oy) = double_add_finish(x, y, u, v, h2, inv);
            assert_eq!(
                Affine::<P>::new_unchecked(ox, oy).into_group(),
                p.double() + q,
                "{curve}: 2*[{p_s}]G + [{q_s}]G"
            );
            checked += 1;
        }
    }
    assert!(checked > 200, "{curve}: only {checked} pairs checked");

    // Q = -P: 2P + (-P) = P, handled without an intermediate identity.
    let p = g * P::ScalarField::from(17u64);
    let (x, y, u, v) = coords(p, -p);
    let (den, h2) = double_add_denominator(x, y, u, v);
    let (ox, oy) = double_add_finish(x, y, u, v, h2, den.inverse().unwrap());
    assert_eq!(
        Affine::<P>::new_unchecked(ox, oy).into_group(),
        p,
        "{curve}: Q = -P"
    );

    // Q = P and Q = -2P: the combined denominator vanishes (schedules with these are gated out).
    let (x, y, u, v) = coords(p, p);
    assert!(
        double_add_denominator(x, y, u, v).0.is_zero(),
        "{curve}: Q = P"
    );
    let (x, y, u, v) = coords(p, -p.double());
    assert!(
        double_add_denominator(x, y, u, v).0.is_zero(),
        "{curve}: Q = -2P"
    );
}

#[test]
fn direct_double_add() {
    for_each_curve!(direct_double_add_for);
}
