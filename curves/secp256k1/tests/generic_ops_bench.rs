//! secp256k1 point addition, doubling and MSM timings. Run with
//! `cargo test --release --manifest-path curves/secp256k1/Cargo.toml [--features asm] --test generic_ops_bench -- --ignored --nocapture`

ark_algebra_test_templates::bench_sw_ops!(ark_secp256k1::Projective);
