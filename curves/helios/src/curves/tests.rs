use crate::{Projective, Affine};
use ark_algebra_test_templates::*;
use ark_ec::{AffineRepr};
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use ark_std::{test_rng, UniformRand, vec};
use ark_ff::Field;

test_group!(g1; Projective; sw);

#[test]
fn compact_serialization() {
    let mut rng = test_rng();

    fn check(g: Affine) {
        assert_eq!(g.compressed_size(), 32);

        let mut compressed_bytes = vec![];
        g.serialize_compressed(&mut compressed_bytes).unwrap();
        assert_eq!(compressed_bytes.len(), 32);
        let g1_compressed: Affine = CanonicalDeserialize::deserialize_compressed(compressed_bytes.as_slice()).unwrap();
        assert_eq!(g, g1_compressed);

        let mut uncompressed_bytes = vec![];
        g.serialize_uncompressed(&mut uncompressed_bytes).unwrap();
        assert_eq!(uncompressed_bytes.len(), 64);
        let g1_uncompressed: Affine = CanonicalDeserialize::deserialize_uncompressed(uncompressed_bytes.as_slice()).unwrap();
        assert_eq!(g, g1_uncompressed);

        let g_wrong: Result<Affine, _> = CanonicalDeserialize::deserialize_uncompressed(compressed_bytes.as_slice());
        assert!(g_wrong.is_err());
    }

    let iterations = 100;
    for _ in 0..iterations {
        let g = Affine::rand(&mut rng);
        assert!(!g.is_zero());
        check(g);
    }

    let g_zero = Affine::zero();
    check(g_zero);
}

#[cfg(feature = "std")]
#[test]
fn test_helios_selene_timing() {
    use crate::Fr;
    use ark_std::{rand::prelude::SliceRandom, vec::Vec};
    use ark_selene::{Projective as SProjective, Fr as SFr};
    use ark_ec::{CurveGroup, PrimeGroup, VariableBaseMSM};
    use std::time::{Instant, Duration};

    let mut rng = test_rng();

    let count = 2000;

    let mut helios_points = Vec::with_capacity(count);
    let mut scalars_helios = Vec::with_capacity(count);
    let mut selene_points = Vec::with_capacity(count);
    let mut scalars_selene = Vec::with_capacity(count);

    for i in 0..count {
        helios_points.push((Affine::generator() * Fr::from(i as u64 + 1)).into_affine());
        scalars_helios.push(Fr::rand(&mut rng));
        selene_points.push((SProjective::generator() * SFr::from(i as u64 + 1)).into_affine());
        scalars_selene.push(SFr::rand(&mut rng));
    }

    // Helios MSM
    let start = Instant::now();
    let r: Projective = VariableBaseMSM::msm_unchecked(&helios_points, &scalars_helios);
    let _ = core::hint::black_box(r);
    let helios_msm_duration = start.elapsed();
    println!("Helios MSM for {} pairs: {:?}", count, helios_msm_duration);

    // Selene MSM
    let start = Instant::now();
    let r: SProjective = VariableBaseMSM::msm_unchecked(&selene_points, &scalars_selene);
    let _ = core::hint::black_box(r);
    let selene_msm_duration = start.elapsed();
    println!("Selene MSM for {} pairs: {:?}", count, selene_msm_duration);

    // Helios scalar multiplications
    let mut helios_mul_times = Vec::new();
    for (scalar, point) in scalars_helios.iter().zip(helios_points.iter()) {
        let instant = Instant::now();
        let r = *point * *scalar;
        let _ = core::hint::black_box(r);
        helios_mul_times.push(instant.elapsed().as_micros());
    }

    helios_mul_times.sort();
    let helios_median_mul_time = helios_mul_times[helios_mul_times.len() / 2];
    println!("Helios median scalar multiplication time ({} operations): {} microsecond", count, helios_median_mul_time);

    // Selene scalar multiplications
    let mut selene_mul_times = Vec::new();
    for (scalar, point) in scalars_selene.iter().zip(selene_points.iter()) {
        let instant = Instant::now();
        let r = *point * *scalar;
        let _ = core::hint::black_box(r);
        selene_mul_times.push(instant.elapsed().as_micros());
    }

    selene_mul_times.sort();
    let selene_median_mul_time = selene_mul_times[selene_mul_times.len() / 2];
    println!("Selene median scalar multiplication time ({} operations): {} microsecond", count, selene_median_mul_time);

    // Benchmark field element operations using existing scalars
    // Helios field element multiplications
    let mut helios_field_mul_duration = Duration::default();
    for i in 0..count-1 {
        let a = scalars_helios[i];
        let b = scalars_helios[i + 1];
        let instant = Instant::now();
        let r = a * b;
        let _ = core::hint::black_box(r);
        helios_field_mul_duration += instant.elapsed();
    }

    let helios_mean_field_mul_time = helios_field_mul_duration.as_nanos() / (count-1) as u128;
    println!("Helios mean field multiplication time ({} operations): {} nanosecond", count-1, helios_mean_field_mul_time);

    // Selene field element multiplications
    let mut selene_field_mul_duration = Duration::default();
    for i in 0..count-1 {
        let a = scalars_selene[i];
        let b = scalars_selene[i + 1];
        let instant = Instant::now();
        let r = a * b;
        let _ = core::hint::black_box(r);
        selene_field_mul_duration += instant.elapsed();
    }

    let selene_mean_field_mul_time = selene_field_mul_duration.as_nanos() / (count-1) as u128;
    println!("Selene mean field multiplication time ({} operations): {} nanosecond", count-1, selene_mean_field_mul_time);

    // Helios field element additions
    let mut helios_field_add_duration = Duration::from_nanos(0);
    for i in 0..count-1 {
        let a = scalars_helios[i];
        let b = scalars_helios[i + 1];
        let instant = Instant::now();
        let r = a + b;
        let _ = core::hint::black_box(r);
        helios_field_add_duration += instant.elapsed();
    }

    let helios_mean_field_add_time = helios_field_add_duration.as_nanos() / (count-1) as u128;
    println!("Helios mean field addition time ({} operations): {} nanosecond", count-1, helios_mean_field_add_time);

    // Selene field element additions
    let mut selene_field_add_duration = Duration::from_nanos(0);
    for i in 0..count-1 {
        let a = scalars_selene[i];
        let b = scalars_selene[i + 1];
        let instant = Instant::now();
        let r = a + b;
        let _ = core::hint::black_box(r);
        selene_field_add_duration += instant.elapsed();
    }

    let selene_mean_field_add_time = selene_field_add_duration.as_nanos() / (count-1) as u128;
    println!("Selene mean field addition time ({} operations): {} nanosecond", count-1, selene_mean_field_add_time);

    // Helios field element inversions
    let mut helios_field_inv_times = Vec::new();
    for i in 0..count {
        let a = scalars_helios[i];
        let instant = Instant::now();
        let r = a.inverse();
        let _ = core::hint::black_box(r);
        helios_field_inv_times.push(instant.elapsed().as_nanos());
    }

    helios_field_inv_times.sort();
    let helios_median_field_inv_time = helios_field_inv_times[helios_field_inv_times.len() / 2];
    println!("Helios median field inversion time ({} operations): {} nanosecond", count, helios_median_field_inv_time);

    // Selene field element inversions
    let mut selene_field_inv_times = Vec::new();
    for i in 0..count {
        let a = scalars_selene[i];
        let instant = Instant::now();
        let r = a.inverse();
        let _ = core::hint::black_box(r);
        selene_field_inv_times.push(instant.elapsed().as_nanos());
    }

    selene_field_inv_times.sort();
    let selene_median_field_inv_time = selene_field_inv_times[selene_field_inv_times.len() / 2];
    println!("Selene median field inversion time ({} operations): {} nanosecond", count, selene_median_field_inv_time);

    // Benchmark point additions
    // Helios point additions
    let mut helios_add_times = Vec::new();

    // Shuffle points
    helios_points.shuffle(&mut rng);

    for i in 0..count / 2 {
        let point1 = helios_points[i * 2];
        let point2 = helios_points[i * 2 + 1];
        let instant = Instant::now();
        let r = point1 + point2;
        let _ = core::hint::black_box(r);
        helios_add_times.push(instant.elapsed().as_nanos());
    }

    helios_add_times.sort();
    let helios_median_add_time = helios_add_times[helios_add_times.len() / 2];
    println!("Helios median point addition time ({} operations): {} nanosecond", count / 2, helios_median_add_time);

    // Selene point additions
    let mut selene_add_times = Vec::new();

    selene_points.shuffle(&mut rng);

    for i in 0..count / 2 {
        let point1 = selene_points[i * 2];
        let point2 = selene_points[i * 2 + 1];
        let instant = Instant::now();
        let r = point1 + point2;
        let _ = core::hint::black_box(r);
        selene_add_times.push(instant.elapsed().as_nanos());
    }

    selene_add_times.sort();
    let selene_median_add_time = selene_add_times[selene_add_times.len() / 2];
    println!("Selene median point addition time ({} operations): {} nanosecond", count / 2, selene_median_add_time);
}