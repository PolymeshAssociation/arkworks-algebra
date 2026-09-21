use ark_ec::{
    scalar_mul::variable_base::{ChunkedPippenger, HashMapPippenger, VariableBaseMSM},
    ScalarMul,
};
use ark_ff::{AdditiveGroup, PrimeField, UniformRand};
use ark_std::{rand::seq::SliceRandom, vec, vec::*};

fn naive_var_base_msm<G: ScalarMul>(bases: &[G::MulBase], scalars: &[G::ScalarField]) -> G {
    let mut acc = G::zero();

    for (base, scalar) in bases.iter().zip(scalars.iter()) {
        acc += *base * scalar;
    }
    acc
}

pub fn test_var_base_msm<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let naive = naive_var_base_msm::<G>(g.as_slice(), v.as_slice());
    let fast = G::msm(g.as_slice(), v.as_slice()).unwrap();

    assert_eq!(naive, fast);
}

type F<G> = <G as ark_ec::PrimeGroup>::ScalarField;

pub fn test_var_base_msm_mixed_scalars<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();
    let mut v = Vec::<F<G>>::with_capacity(SAMPLES * 11);
    // Positive and negative u1s
    v.extend((0..SAMPLES).map(|_| F::<G>::from(bool::rand(&mut rng))));
    v.extend((0..SAMPLES).map(|_| -F::<G>::from(bool::rand(&mut rng))));

    // Positive and negative u8s
    v.extend((0..SAMPLES).map(|_| F::<G>::from(u8::rand(&mut rng))));
    v.extend((0..SAMPLES).map(|_| -F::<G>::from(u8::rand(&mut rng))));

    // Positive and negative u16s
    v.extend((0..SAMPLES).map(|_| F::<G>::from(u16::rand(&mut rng))));
    v.extend((0..SAMPLES).map(|_| -F::<G>::from(u16::rand(&mut rng))));

    // Positive and negative u32s
    v.extend((0..SAMPLES).map(|_| F::<G>::from(u32::rand(&mut rng))));
    v.extend((0..SAMPLES).map(|_| -F::<G>::from(u32::rand(&mut rng))));

    // Positive and negative u64s
    v.extend((0..SAMPLES).map(|_| F::<G>::from(u64::rand(&mut rng))));
    v.extend((0..SAMPLES).map(|_| -F::<G>::from(u64::rand(&mut rng))));

    // Random scalars
    v.extend((0..SAMPLES).map(|_| F::<G>::from(G::ScalarField::rand(&mut rng))));
    v.shuffle(&mut rng);

    let g = (0..v.len()).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let naive = naive_var_base_msm::<G>(g.as_slice(), v.as_slice());
    let fast = G::msm(g.as_slice(), v.as_slice()).unwrap();

    assert_eq!(naive, fast);
}

pub fn test_var_base_msm_specialized<G: VariableBaseMSM>() {
    const SAMPLES: usize = (1 << 10) * 5;

    let rng = &mut ark_std::test_rng();
    let g = (0..SAMPLES).map(|_| G::rand(rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let v = (0..SAMPLES).map(|_| bool::rand(rng)).collect::<Vec<_>>();
    let v_fe = v.iter().map(|&b| F::<G>::from(b)).collect::<Vec<_>>();
    let naive = naive_var_base_msm::<G>(g.as_slice(), v_fe.as_slice());
    let fast = G::msm_u1(g.as_slice(), v.as_slice());
    assert_eq!(naive, fast);

    let v = (0..SAMPLES).map(|_| u8::rand(rng)).collect::<Vec<_>>();
    let v_fe = v.iter().map(|&b| F::<G>::from(b)).collect::<Vec<_>>();
    let naive = naive_var_base_msm::<G>(g.as_slice(), v_fe.as_slice());
    let fast = G::msm_u8(g.as_slice(), v.as_slice());
    assert_eq!(naive, fast);

    let v = (0..SAMPLES).map(|_| u16::rand(rng)).collect::<Vec<_>>();
    let v_fe = v.iter().map(|&b| F::<G>::from(b)).collect::<Vec<_>>();
    let naive = naive_var_base_msm::<G>(g.as_slice(), v_fe.as_slice());
    let fast = G::msm_u16(g.as_slice(), v.as_slice());
    assert_eq!(naive, fast);

    let v = (0..SAMPLES).map(|_| u32::rand(rng)).collect::<Vec<_>>();
    let v_fe = v.iter().map(|&b| F::<G>::from(b)).collect::<Vec<_>>();
    let naive = naive_var_base_msm::<G>(g.as_slice(), v_fe.as_slice());
    let fast = G::msm_u32(g.as_slice(), v.as_slice());
    assert_eq!(naive, fast);

    let v = (0..SAMPLES).map(|_| u64::rand(rng)).collect::<Vec<_>>();
    let v_fe = v.iter().map(|&b| F::<G>::from(b)).collect::<Vec<_>>();
    let naive = naive_var_base_msm::<G>(g.as_slice(), v_fe.as_slice());
    let fast = G::msm_u64(g.as_slice(), v.as_slice());
    assert_eq!(naive, fast);
}

pub fn test_chunked_pippenger<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let v = (0..SAMPLES)
        .map(|_| G::ScalarField::rand(&mut rng).into_bigint())
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let arkworks = G::msm_bigint(g.as_slice(), v.as_slice());

    let mut p = ChunkedPippenger::<G>::new(1 << 20);
    for (s, g) in v.iter().zip(g) {
        p.add(g, s);
    }
    let mine = p.finalize();
    assert_eq!(arkworks, mine);
}

pub fn test_hashmap_pippenger<G: VariableBaseMSM>() {
    const SAMPLES: usize = 1 << 10;

    let mut rng = ark_std::test_rng();

    let mut v_scal = Vec::new();
    let v = (0..SAMPLES)
        .map(|_| {
            let x = G::ScalarField::rand(&mut rng);
            v_scal.push(x);
            x.into_bigint()
        })
        .collect::<Vec<_>>();
    let g = (0..SAMPLES).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
    let g = G::batch_convert_to_mul_base(&g);

    let arkworks = G::msm_bigint(g.as_slice(), v.as_slice());

    let mut p = HashMapPippenger::<G>::new(1 << 20);
    for (s, g) in v_scal.iter().zip(g) {
        p.add(g, s);
    }
    let mine = p.finalize();
    assert_eq!(arkworks, mine);
}

/// The routed small-MSM path must agree with the bucket algorithm it bypasses, at every size
/// around the routing thresholds and on the inputs that stress the ladders: zero scalars,
/// `1`, `r - 1`, small scalars, repeated bases and identity bases.
pub fn test_var_base_msm_small<G: VariableBaseMSM>() {
    let rng = &mut ark_std::test_rng();
    let one = F::<G>::from(1u64);
    let neg_one = -one;

    for n in [0usize, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 200] {
        let random = (0..n).map(|_| G::rand(rng)).collect::<Vec<_>>();
        let mut repeated = random.clone();
        if n > 1 {
            let first = repeated[0];
            repeated.iter_mut().for_each(|g| *g = first);
        }
        let mut with_identity = random.clone();
        for (i, g) in with_identity.iter_mut().enumerate() {
            if i % 3 == 0 {
                *g = G::zero();
            }
        }

        for bases in [random, repeated, with_identity] {
            let bases = G::batch_convert_to_mul_base(&bases);
            let scalar_sets = [
                (0..n).map(|_| F::<G>::rand(rng)).collect::<Vec<_>>(),
                vec![F::<G>::ZERO; n],
                vec![one; n],
                vec![neg_one; n],
                (0..n).map(|i| F::<G>::from(i as u64)).collect::<Vec<_>>(),
                (0..n)
                    .map(|i| if i % 2 == 0 { F::<G>::ZERO } else { neg_one })
                    .collect::<Vec<_>>(),
            ];
            for scalars in scalar_sets {
                let bigints = scalars
                    .iter()
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let expected = G::msm_bigint(&bases, &bigints);
                assert_eq!(
                    G::msm_unchecked(&bases, &scalars),
                    expected,
                    "n = {n}"
                );
                assert_eq!(naive_var_base_msm::<G>(&bases, &scalars), expected, "n = {n}");
            }
        }
    }
}

/// The batch-affine bucket MSM must agree with the `xyzz` path it replaces, including on the
/// inputs that force same-x pairs inside a bucket at every level of the reduction tree.
pub fn test_batch_affine_msm<P: ark_ec::short_weierstrass::SWCurveConfig>() {
    use ark_ec::{
        scalar_mul::{sw_pippenger::msm_batch_affine_bigint, variable_base::msm_bigint_wnaf},
        short_weierstrass::{Affine, Projective},
        AffineRepr, CurveGroup,
    };
    let rng = &mut ark_std::test_rng();
    let one = P::ScalarField::from(1u64);
    let neg_one = -one;

    for n in [1usize, 2, 31, 32, 1000, 1 << 12] {
        let random: Vec<Affine<P>> = (0..n)
            .map(|_| Projective::<P>::rand(rng).into_affine())
            .collect();
        let base = random[0];

        // Every base equal: one bucket fills with copies, so the first level is all doublings
        // and the ones after it are doublings of the same intermediate.
        let all_equal = vec![base; n];
        // Alternating `P` and `-P`: pairs cancel to the identity inside a bucket.
        let mut alternating = vec![base; n];
        for (i, b) in alternating.iter_mut().enumerate() {
            if i % 2 == 1 {
                *b = -base;
            }
        }
        // Every third base the identity, which the sort must drop.
        let mut with_identity = random.clone();
        for (i, b) in with_identity.iter_mut().enumerate() {
            if i % 3 == 0 {
                *b = Affine::<P>::zero();
            }
        }
        // Four distinct bases repeated, so buckets hold long runs of equal points.
        let few: Vec<Affine<P>> = (0..n).map(|i| random[i % 4.min(n)]).collect();

        for bases in [random.clone(), all_equal, alternating, with_identity, few] {
            let scalar_sets = [
                (0..n).map(|_| P::ScalarField::rand(rng)).collect::<Vec<_>>(),
                vec![one; n],
                vec![neg_one; n],
                // Few distinct digits, so buckets are deep and unevenly filled.
                (0..n)
                    .map(|i| P::ScalarField::from((i % 3) as u64))
                    .collect::<Vec<_>>(),
            ];
            for scalars in scalar_sets {
                let bigints = scalars
                    .iter()
                    .map(|s| s.into_bigint())
                    .collect::<Vec<_>>();
                let expected: Projective<P> = msm_bigint_wnaf(&bases, &bigints);
                assert_eq!(msm_batch_affine_bigint::<P>(&bases, &bigints), expected, "n = {n}");
            }
        }
    }
}
