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

/// An ignored `mulx_dispatch` test timing `Fq` and `Fr` multiplication and squaring, G1 addition,
/// doubling and a `2^12` MSM, and a pairing, each the median of 15 runs. Comparing a build that
/// reaches the x86-64 MULX/ADX kernel through the run-time CPUID check with one that folds the
/// check away gives the check's cost.
#[macro_export]
macro_rules! bench_mulx_dispatch {
    ($pairing:ty, $fq:ty) => {
        #[test]
        #[ignore = "timing comparison; run explicitly with --release --nocapture"]
        fn mulx_dispatch() {
            use ::std::{hint::black_box, time::Instant, vec::Vec};
            use ark_ec::{pairing::Pairing, AdditiveGroup, CurveGroup, VariableBaseMSM};
            use ark_ff::{Field, UniformRand};

            type Fr = <$pairing as Pairing>::ScalarField;
            type G1 = <$pairing as Pairing>::G1;
            type G2 = <$pairing as Pairing>::G2;

            const SAMPLES: usize = 1024;
            const INNER: usize = 50_000;
            const REPS: usize = 15;

            /// Median ns per call over `REPS` repetitions of `n` calls of `f`.
            fn time(n: usize, mut f: impl FnMut(usize)) -> f64 {
                let mut t: Vec<f64> = (0..REPS)
                    .map(|_| {
                        let s = Instant::now();
                        for i in 0..n {
                            f(i);
                        }
                        s.elapsed().as_nanos() as f64 / n as f64
                    })
                    .collect();
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            }

            /// Throughput over independent operations and latency over one dependent chain.
            fn field_rows<F: Field + UniformRand>(name: &str) {
                let mut rng = ark_std::test_rng();
                let xs: Vec<F> = (0..SAMPLES).map(|_| F::rand(&mut rng)).collect();
                let ys: Vec<F> = (0..SAMPLES).map(|_| F::rand(&mut rng)).collect();
                let ops: [(&str, fn(F, F) -> F); 2] =
                    [("mul", |x, y| x * y), ("square", |x, _| x.square())];
                for (op_name, op) in ops {
                    let thr = time(INNER, |i| {
                        black_box(op(xs[i % SAMPLES], ys[i % SAMPLES]));
                    });
                    let mut acc = xs[0];
                    let lat = time(INNER, |i| acc = op(acc, ys[i % SAMPLES]));
                    black_box(acc);
                    ::std::println!("  {name} {op_name:<7} thr {thr:>9.2} ns   lat {lat:>9.2} ns");
                }
            }

            ::std::println!(
                "target_arch {} | asm feature {} | MULX kernel {} | BMI2 and ADX at compile time {}",
                ::std::env::consts::ARCH,
                cfg!(feature = "asm"),
                ark_ff::mulx_kernel_active(),
                cfg!(all(target_feature = "bmi2", target_feature = "adx")),
            );
            field_rows::<$fq>("Fq");
            field_rows::<Fr>("Fr");

            let mut rng = ark_std::test_rng();
            let ps: Vec<G1> = (0..SAMPLES).map(|_| G1::rand(&mut rng)).collect();
            let add = time(INNER, |i| {
                let _ = black_box(ps[i % SAMPLES] + ps[(i + 1) % SAMPLES]);
            });
            ::std::println!("  G1 add          {add:>12.1} ns");
            let double = time(INNER, |i| {
                let _ = black_box(ps[i % SAMPLES].double());
            });
            ::std::println!("  G1 double       {double:>12.1} ns");

            let bases = G1::normalize_batch(&ps[..]);
            let bases: Vec<_> = bases.iter().cycle().take(1 << 12).copied().collect();
            let scalars: Vec<Fr> = (0..bases.len()).map(|_| Fr::rand(&mut rng)).collect();
            let msm = time(10, |_| {
                let _ = black_box(G1::msm(&bases, &scalars).unwrap());
            });
            ::std::println!("  G1 msm 2^12     {msm:>12.1} ns");

            // Rotating inputs, so the branch predictor cannot memorize one pairing.
            let g1 = G1::normalize_batch(&ps[..16]);
            let g2: Vec<_> = (0..16).map(|_| G2::rand(&mut rng).into_affine()).collect();
            let pairing = time(200, |i| {
                let _ = black_box(<$pairing>::pairing(g1[i % 16], g2[i % 16]));
            });
            ::std::println!("  pairing         {pairing:>12.1} ns");
        }
    };
}
