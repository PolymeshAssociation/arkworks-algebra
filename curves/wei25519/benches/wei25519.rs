use ark_algebra_bench_templates::*;
use ark_wei25519::{Fq, Fr, Projective as G};

bench!(
    Name = "Wei25519",
    Group = G,
    ScalarField = Fr,
    PrimeBaseField = Fq,
);
