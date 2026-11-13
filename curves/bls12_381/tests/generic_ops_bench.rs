//! BLS12-381 G1 point addition, doubling and MSM timings. Run with
//! `cargo test --release --manifest-path curves/bls12_381/Cargo.toml [--features asm] --test generic_ops_bench -- --ignored --nocapture`

ark_algebra_test_templates::bench_sw_ops!(ark_bls12_381::G1Projective);
