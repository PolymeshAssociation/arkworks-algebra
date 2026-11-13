//! Cost of the run-time CPUID check that selects the x86-64 MULX/ADX Montgomery kernel. Compare a
//! default build, which reaches the kernel through the check, with one that folds it away by
//! enabling BMI2 and ADX at compile time, in separate target directories:
//! `cargo test --release --target-dir target-cpuid --manifest-path curves/bls12_381/Cargo.toml --features asm --test mulx_dispatch_bench -- --ignored --nocapture`
//! and the same command as `cargo --config .cargo/bmi2-adx.toml test ...` with
//! `--target-dir target-folded`.

ark_algebra_test_templates::bench_mulx_dispatch!(ark_bls12_381::Bls12_381, ark_bls12_381::Fq);
