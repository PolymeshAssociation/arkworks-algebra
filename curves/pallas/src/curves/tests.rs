use crate::{Projective, Affine};
use ark_algebra_test_templates::*;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

test_group!(g1; Projective; sw);
test_group!(g1_glv; Projective; glv);
test_compact_serialization!(Affine; 32; 64);
test_h2c!(pallas_h2c; "./src/curves/test_vectors"; "Pallas"; crate::PallasConfig; crate::Fq; crate::Fq; 1);
