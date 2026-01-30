use ark_algebra_bench_templates::*;
use ark_helios::{fq::Fq, fr::Fr, Projective as G};

bench!(
    Name = "Helios",
    Group = G,
    ScalarField = Fr,
    PrimeBaseField = Fq,
);
