use ark_algebra_bench_templates::*;
use ark_selene::{fq::Fq, fr::Fr, Projective as G};

bench!(
    Name = "Selene",
    Group = G,
    ScalarField = Fr,
    PrimeBaseField = Fq,
);
