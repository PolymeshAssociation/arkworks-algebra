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

/// A nonzero lattice vector `(a, b)` with `\pm a \pm b\lambda = 0 \bmod r` recodes to a nonzero
/// digit string for the scalar zero. Its schedule fails the batch-affine ladder's safety check,
/// so the batch takes the per-point ladder and every product is the identity.
fn lattice_vector_schedule_for<P: GLVConfig>() {
    let curve = type_name::<P>();
    let coeffs = P::SCALAR_DECOMP_COEFFS;
    let rows = [(coeffs[0], coeffs[1]), (coeffs[2], coeffs[3])];
    let mut found = None;
    'search: for ((_, a), (_, b)) in rows {
        let (a, b) = (P::ScalarField::from_bigint(a).unwrap(), P::ScalarField::from_bigint(b).unwrap());
        for sa in [true, false] {
            for sb in [true, false] {
                let va = if sa { a } else { -a };
                let vb = if sb { b } else { -b };
                if (va + vb * P::LAMBDA).is_zero() {
                    if let Some(d) = Decomposed::<P>::new_given_halves(sa, a, sb, b) {
                        found = Some(d);
                        break 'search;
                    }
                }
            }
        }
    }
    let d = found.unwrap_or_else(|| panic!("{curve}: no lattice row fits the recoding"));
    assert!(d.len() > 2, "{curve}: lattice vector recodes to {} digits", d.len());

    let mut rng = test_rng();
    let points = (0..40).map(|_| Projective::<P>::rand(&mut rng)).collect::<Vec<_>>();
    let tables = Table::<P>::batch(&points);
    let batch = Table::mul_decomposed_batch(&tables, &d);
    for (i, (table, b)) in tables.iter().zip(&batch).enumerate() {
        assert!(b.is_zero(), "{curve}: table {i} batch is not the identity");
        assert_eq!(*b, table.mul_decomposed(&d), "{curve}: table {i} batch vs per-point");
    }
}

#[test]
fn lattice_vector_schedule_takes_per_point_ladder() {
    for_each_curve!(lattice_vector_schedule_for);
}

/// BLS12-381 G1 has the order-3 points `(0, \pm 2)`, fixed by the endomorphism, so the affine
/// table chain meets an equal-x pair in its first layer and an identity operand in its second.
/// Both fall back to the projective build, and the tables match it entry by entry.
#[test]
fn table_batch_falls_back_on_exceptional_points() {
    type P = ark_test_curves::bls12_381::g1::Config;
    type Fq = <P as ark_ec::CurveConfig>::BaseField;
    let mut rng = test_rng();
    let two = Fq::from(2u64);
    let torsion = [Affine::<P>::new_unchecked(Fq::zero(), two), Affine::<P>::new_unchecked(Fq::zero(), -two)];
    assert!(torsion.iter().all(|t| t.is_on_curve()));
    let mut points = (0..6).map(|_| Projective::<P>::rand(&mut rng)).collect::<Vec<_>>();
    points.insert(2, torsion[0].into_group());
    points.insert(5, Projective::<P>::zero());
    points.push(torsion[1].into_group());

    let affine = points.iter().map(|p| p.into_affine()).collect::<Vec<_>>();
    let expected = Table::<P>::batch_projective(&points);
    for (label, tables) in [
        ("batch", Table::<P>::batch(&points)),
        ("batch_from_affine", Table::<P>::batch_from_affine(&affine)),
    ] {
        for (i, (t, e)) in tables.iter().zip(&expected).enumerate() {
            assert_eq!(t.is_identity(), e.is_identity(), "{label}: table {i} identity");
            if e.is_identity() {
                continue;
            }
            for code in 1..=48u8 {
                assert_eq!(t.digit_point(code), e.digit_point(code), "{label}: table {i}, digit {code}");
            }
        }
    }

    // GLV is exact off the subgroup for scalars below `2^128`.
    let k = <P as ark_ec::CurveConfig>::ScalarField::from(u128::rand(&mut rng));
    let d = Decomposed::<P>::new(k).unwrap();
    for (p, t) in points.iter().zip(Table::<P>::batch(&points)) {
        assert_eq!(t.mul_decomposed(&d), naive_mul(*p, k));
    }
}
