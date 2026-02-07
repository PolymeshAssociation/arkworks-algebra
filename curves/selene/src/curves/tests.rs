use ark_std::rand::Rng;
use crate::{Projective, Affine, SeleneConfig};
use ark_algebra_test_templates::*;

test_group!(g1; Projective; sw);

test_compact_serialization!(Affine; 32; 64);

test_h2c_swu!(hash_arbitrary_string_to_curve_swu; Projective; SeleneConfig);