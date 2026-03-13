#![cfg_attr(not(feature = "std"), no_std)]
#![deny(
    warnings,
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms
)]
#![forbid(unsafe_code)]

//! This library implements the wei25519 short Weierstrass curve.
//!
//! Curve information:
//! * Base field: Same as curve25519 - q =
//!   57896044618658097711785492504343953926634992332820282019728792003956564819949
//! * Scalar field: Same as curve25519 - r =
//!   7237005577332262213973186563042994240857116359379907606001950938285454250989
//! * Curve equation: y^2 = x^3 + A*x + B
//!
//! Reference: https://datatracker.ietf.org/doc/html/draft-ietf-lwig-curve-representations-19

mod curves;

pub use curves::*;

// Re-export field types from curve25519
pub use ark_curve25519::{Fq, Fr};

#[cfg(test)]
mod tests {
    use crate::{Fq, Fr};
    use ark_algebra_test_templates::*;

    test_field!(fr; Fr; mont_prime_field);
    test_field!(fq; Fq; mont_prime_field);
}