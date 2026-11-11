/// An ignored `generic_ops` test timing mixed addition, full addition, doubling and MSMs of
/// `2^5`, `2^8` and `2^12` points for a short Weierstrass projective group, one `ROW <label> <ns>`
/// line per operation, each the median of 15 runs.
#[macro_export]
macro_rules! bench_sw_ops {
    ($group:ty) => {
        #[test]
        #[ignore = "timing; run explicitly with --release --nocapture"]
        fn generic_ops() {
            use ::std::{hint::black_box, time::Instant};
            use ark_ec::{AdditiveGroup, CurveGroup, PrimeGroup, VariableBaseMSM};
            use ark_ff::UniformRand;

            type Affine = <$group as CurveGroup>::Affine;
            type Scalar = <$group as PrimeGroup>::ScalarField;

            fn row(label: &str, per: usize, mut f: impl FnMut()) {
                f();
                let mut t: ::std::vec::Vec<f64> = (0..15)
                    .map(|_| {
                        let s = Instant::now();
                        f();
                        s.elapsed().as_secs_f64() * 1e9 / per as f64
                    })
                    .collect();
                t.sort_by(f64::total_cmp);
                ::std::println!("ROW {label} {:.2}", t[t.len() / 2]);
            }

            let rng = &mut ark_std::test_rng();
            let proj: ::std::vec::Vec<$group> = (0..1024).map(|_| <$group>::rand(rng)).collect();
            let aff: ::std::vec::Vec<Affine> = <$group>::normalize_batch(&proj);
            let inner = 100_000;
            row("madd_ns", inner, || {
                let mut acc = proj[0];
                for i in 0..inner {
                    acc += &aff[i & 1023];
                }
                let _ = black_box(acc);
            });
            row("add_ns", inner, || {
                let mut acc = proj[0];
                for i in 0..inner {
                    acc += &proj[i & 1023];
                }
                let _ = black_box(acc);
            });
            row("double_ns", inner, || {
                let mut acc = proj[0];
                for _ in 0..inner {
                    acc.double_in_place();
                }
                let _ = black_box(acc);
            });
            let s: ::std::vec::Vec<Scalar> = (0..1 << 12).map(|_| Scalar::rand(rng)).collect();
            let b: ::std::vec::Vec<Affine> = (0..1 << 12).map(|i| aff[i % 1024]).collect();
            for log_n in [5usize, 8, 12] {
                let n = 1 << log_n;
                let rounds = (1 << 13) / n;
                row(
                    &::std::format!("msm_2^{log_n}_ns_per_point"),
                    rounds * n,
                    || {
                        for _ in 0..rounds {
                            let _ = black_box(<$group>::msm_unchecked(&b[..n], &s[..n]));
                        }
                    },
                );
            }
        }
    };
}
