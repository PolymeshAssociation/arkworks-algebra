#![cfg_attr(not(feature = "std"), no_std)]

use ark_ec::scalar_mul::sw_pippenger::msm_batch_affine;
use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
use ark_ec::VariableBaseMSM;
pub use ark_host_msm::{pack_fat_pointer, unpack_fat_pointer, CurveMSMId, CURVE_ID_LEN};
use ark_pallas::PallasConfig;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::boxed::Box;
use ark_std::collections::BTreeMap;
use ark_std::vec::Vec;
use ark_vesta::VestaConfig;

#[cfg(feature = "std")]
pub use table_cache::{clear_tables, register_table, register_table_with_given_size};

/// Native-only fixed-base table cache for the host MSM. Populated at node init from the
/// deterministic DART generators.
#[cfg(feature = "std")]
pub mod table_cache {
    use super::CurveMSMId;
    use ark_ec::scalar_mul::fixed_base::FixedBaseMSM;
    use ark_ec::scalar_mul::sw_pippenger::msm_batch_affine;
    use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
    use ark_ec::VariableBaseMSM;
    use ark_ff::Zero;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{Arc, RwLock};
    use std::vec::Vec;

    /// Below this base count the per-base filtering isn't worth it; use the plain MSM.
    const MIN_BASES_FOR_TABLE: usize = 256;

    /// Type-erased per-curve table so the registry can hold any curve behind one trait.
    pub trait HostTable: Send + Sync {
        /// Deserialize `(bases, scalars)` from `buffer[CURVE_ID_LEN..buf_len]`, compute the MSM with
        /// the fixed-base table for registered bases and a batch-affine MSM for the rest, serialize
        /// the result into `buffer`, return its length.
        fn msm(&self, buffer: &mut [u8], buf_len: u32) -> u32;
    }

    pub struct CurveTable<P: SWCurveConfig> {
        num_bases: usize,
        tables: FixedBaseMSM<P>,
        /// Index of base in the `tables`
        base_index: HashMap<Affine<P>, usize>,
    }

    impl<P: SWCurveConfig> CurveTable<P> {
        fn new(bases: &[Affine<P>]) -> Self {
            let tables = FixedBaseMSM::new(bases);
            let base_index = bases.into_iter().enumerate().map(|(i, b)| (*b, i)).collect();
            Self {
                num_bases: bases.len(),
                tables,
                base_index,
            }
        }

        pub fn new_given_window_size(bases: &[Affine<P>], size: usize) -> Self {
            let tables = FixedBaseMSM::new_given_window_size(bases, size);
            let base_index = bases.into_iter().enumerate().map(|(i, b)| (*b, i)).collect();
            Self {
                num_bases: bases.len(),
                tables,
                base_index,
            }
        }

        /// Bytes held by the fixed-base table (excludes the point->slot index).
        #[cfg(test)]
        pub(crate) fn table_bytes(&self) -> usize {
            self.tables.table_bytes()
        }

        /// Split `(bases, scalars)` into the fixed part — an indexed scalar vector aligned with
        /// the table's bases — and the variable part (bases/scalars not in the table).
        pub(crate) fn split(
            &self,
            bases: &[Affine<P>],
            scalars: &[P::ScalarField],
        ) -> (Vec<P::ScalarField>, Vec<Affine<P>>, Vec<P::ScalarField>) {
            let mut fixed = vec![P::ScalarField::zero(); self.num_bases];
            let mut var_bases = Vec::new();
            let mut var_scalars = Vec::new();
            for (p, s) in bases.iter().zip(scalars.iter()) {
                match self.base_index.get(p) {
                    Some(&i) => fixed[i] += *s,
                    None => {
                        var_bases.push(*p);
                        var_scalars.push(*s);
                    }
                }
            }
            (fixed, var_bases, var_scalars)
        }

        /// The table-aware MSM: the fixed part goes through the fixed-base tables.
        pub fn table_aware_msm(
            &self,
            bases: &[Affine<P>],
            scalars: &[P::ScalarField],
        ) -> Projective<P> {
            let (fixed, var_bases, var_scalars) = self.split(bases, scalars);
            self.tables.msm(&fixed) + msm_batch_affine::<P>(&var_bases, &var_scalars)
        }
    }

    impl<P: SWCurveConfig> HostTable for CurveTable<P> {
        fn msm(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
            let (bases, scalars) = match super::read_msm_input::<P>(buffer, buf_len as usize) {
                Some(input) => input,
                None => return 0,
            };
            let res: Projective<P> = if bases.len() < MIN_BASES_FOR_TABLE {
                msm_batch_affine::<P>(&bases, &scalars)
            } else {
                self.table_aware_msm(&bases, &scalars)
            };
            super::write_msm_result(res, buffer)
        }
    }

    lazy_static::lazy_static! {
        static ref TABLES: RwLock<BTreeMap<CurveMSMId, Arc<dyn HostTable>>> =
            RwLock::new(BTreeMap::new());
    }

    /// If a table is registered for `curve_id`, run the table-aware MSM and return its serialized
    /// length, else `None`.
    pub(super) fn try_table_msm(
        curve_id: &CurveMSMId,
        buffer: &mut [u8],
        buf_len: u32,
    ) -> Option<u32> {
        let table = {
            let tables = TABLES.read().ok()?;
            tables.get(curve_id).cloned()
        }?;
        Some(table.msm(buffer, buf_len))
    }

    /// Register/replace the fixed-base table for curve `P` over `bases`. Called natively at node
    /// init. A curve with no registration uses the plain MSM.
    pub fn register_table<P: SWCurveConfig>(bases: &[Affine<P>]) {
        insert_table(CurveTable::new(bases));
    }

    /// [`register_table`] but with an explicit fixed-base window size instead of the one chosen
    /// by arkworks. A smaller `c` gives more windows — a larger table but a smaller per-window
    /// bucket array. Lets a benchmark compare table memory and eval time across window sizes.
    pub fn register_table_with_given_size<P: SWCurveConfig>(bases: &[Affine<P>], window_size: usize) {
        insert_table(CurveTable::new_given_window_size(bases, window_size));
    }

    fn insert_table<P: SWCurveConfig>(table: CurveTable<P>) {
        let name = match <Projective<P> as VariableBaseMSM>::curve_name() {
            Some(n) => n,
            None => return,
        };
        let curve_id = CurveMSMId::from_curve_name(name);
        let table: Arc<dyn HostTable> = Arc::new(table);
        if let Ok(mut tables) = TABLES.write() {
            tables.insert(curve_id, table);
        }
    }

    /// Remove all registered tables (fall back to the plain MSM everywhere).
    pub fn clear_tables() {
        if let Ok(mut tables) = TABLES.write() {
            tables.clear();
        }
    }

}

type CurveMSMFn = Box<dyn Fn(&mut [u8], u32) -> u32 + Send + Sync>;

pub struct RegisteredCurves {
    curves: BTreeMap<CurveMSMId, CurveMSMFn>,
}

impl RegisteredCurves {
    pub fn new() -> Self {
        let mut curves = RegisteredCurves {
            curves: BTreeMap::new(),
        };
        curves.register_curve::<PallasConfig>();
        curves.register_curve::<VestaConfig>();
        curves
    }

    pub fn register_curve<P: SWCurveConfig + 'static>(&mut self) -> bool {
        if let Some(name) = <Projective<P> as VariableBaseMSM>::curve_name() {
            let curve_id = CurveMSMId::from_curve_name(name);
            self.curves
                .insert(curve_id, Box::new(host_msm_unchecked_impl::<P>));
            true
        } else {
            false
        }
    }

    pub fn msm_unchecked(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
        if (buf_len as usize) < CURVE_ID_LEN {
            return 0; // Buffer too small to contain curve ID
        }
        if let Some(curve_id) = CurveMSMId::deserialize_uncompressed_unchecked(&buffer[..]).ok() {
            if let Some(msm_fn) = self.curves.get(&curve_id) {
                return if buf_len as usize > CURVE_ID_LEN {
                    // Prefer the fixed-base table path when a table is registered for this curve.
                    if let Some(res_len) = table_cache::try_table_msm(&curve_id, buffer, buf_len) {
                        return res_len;
                    }
                    msm_fn(buffer, buf_len)
                } else {
                    0 // Curve is supported, but no MSM data provided
                }
            }
        }
        0
    }
}

#[cfg(feature = "std")]
lazy_static::lazy_static! {
    pub static ref SUPPORTED_CURVES: RegisteredCurves = {
        RegisteredCurves::new()
    };
}

#[cfg(not(feature = "std"))]
static mut SUPPORTED_CURVES: Option<RegisteredCurves> = None;

#[cfg(not(feature = "std"))]
#[allow(static_mut_refs)]
fn get_supported_curves() -> &'static RegisteredCurves {
    unsafe {
        if SUPPORTED_CURVES.is_none() {
            SUPPORTED_CURVES = Some(RegisteredCurves::new());
        }
        SUPPORTED_CURVES.as_ref().unwrap()
    }
}

#[cfg(feature = "std")]
pub fn host_msm_unchecked(buffer: &mut [u8], buf_len: u32) -> u32 {
    SUPPORTED_CURVES.msm_unchecked(buffer, buf_len)
}

#[cfg(not(feature = "std"))]
pub fn host_msm_unchecked(buffer: &mut [u8], buf_len: u32) -> u32 {
    get_supported_curves().msm_unchecked(buffer, buf_len)
}

/// Deserialize the `(bases, scalars)` MSM input from `buffer[CURVE_ID_LEN..buf_len]`. Returns
/// `None` on a deserialization failure instead of panicking across the host-function boundary.
fn read_msm_input<P: SWCurveConfig>(
    buffer: &[u8],
    buf_len: usize,
) -> Option<(Vec<Affine<P>>, Vec<P::ScalarField>)> {
    let mut cursor = ark_std::io::Cursor::new(&buffer[CURVE_ID_LEN..buf_len]);
    let bases: Vec<Affine<P>> = CanonicalDeserialize::deserialize_uncompressed(&mut cursor).ok()?;
    let scalars: Vec<P::ScalarField> =
        CanonicalDeserialize::deserialize_uncompressed(&mut cursor).ok()?;
    Some((bases, scalars))
}

/// Serialize the MSM result into the front of `buffer` and return its byte length. Returns `0`
/// when the buffer is too small or serialization fails.
fn write_msm_result<P: SWCurveConfig>(res: Projective<P>, buffer: &mut [u8]) -> u32 {
    let res_len = res.serialized_size(Compress::No);
    if res_len > buffer.len() {
        return 0;
    }
    match res.serialize_uncompressed(&mut buffer[..res_len]) {
        Ok(()) => res_len as u32,
        Err(_) => 0,
    }
}

/// The stateless (no fixed-base table) host MSM for curve `P`, registered under the extern
/// host-function name `host_msm_unchecked`.
///
/// NOTE: despite the `unchecked` in the name — and despite `VariableBaseMSM::msm_unchecked` (via
/// `msm_unchecked_inner`) being the natural dispatch for this entry point — this deliberately
/// computes the MSM with `msm_batch_affine`, which benchmarks faster than `msm_unchecked`.
/// Avoiding adding new host functions, so the batch-affine path is added to this existing
/// entry point rather than exposed as a separate one.
fn host_msm_unchecked_impl<P: SWCurveConfig>(buffer: &mut [u8], buf_len: u32) -> u32 {
    if buf_len as usize == CURVE_ID_LEN {
        // The curve is supported.
        return 1;
    }
    let (bases, scalars) = match read_msm_input::<P>(buffer, buf_len as usize) {
        Some(input) => input,
        None => return 0,
    };
    write_msm_result(msm_batch_affine::<P>(&bases, &scalars), buffer)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{clear_tables, host_msm_unchecked, register_table};
    use ark_ec::short_weierstrass::Projective;
    use ark_ec::VariableBaseMSM;
    use ark_host_msm::{CurveMSMId, CURVE_ID_LEN};
    use ark_pallas::{Affine as PallasAffine, Fr as PallasFr, PallasConfig};
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ark_std::{test_rng, vec::Vec, UniformRand};
    use std::time::Instant;
    use crate::table_cache::CurveTable;

    fn build_buffer(bases: &[PallasAffine], scalars: &[PallasFr]) -> Vec<u8> {
        let name = <Projective<PallasConfig> as VariableBaseMSM>::curve_name().unwrap();
        let curve_id = CurveMSMId::from_curve_name(name);
        let mut buf = Vec::new();
        curve_id.serialize_uncompressed(&mut buf).unwrap();
        bases.serialize_uncompressed(&mut buf).unwrap();
        scalars.serialize_uncompressed(&mut buf).unwrap();
        buf
    }

    fn run(bases: &[PallasAffine], scalars: &[PallasFr]) -> Projective<PallasConfig> {
        let mut buf = build_buffer(bases, scalars);
        let len = buf.len() as u32;
        let res_len = host_msm_unchecked(&mut buf, len) as usize;
        Projective::<PallasConfig>::deserialize_uncompressed_unchecked(&buf[..res_len]).unwrap()
    }

    #[test]
    fn round_trip_matches_in_process() {
        let mut rng = test_rng();
        clear_tables();
        for n in [1usize, 2, 64, 65, 1023, 1024, 2000] {
            let bases: Vec<PallasAffine> = (0..n).map(|_| PallasAffine::rand(&mut rng)).collect();
            let scalars: Vec<PallasFr> = (0..n).map(|_| PallasFr::rand(&mut rng)).collect();
            let expected = Projective::<PallasConfig>::msm_unchecked(&bases, &scalars);
            assert_eq!(run(&bases, &scalars), expected, "n = {n}");
        }
    }

    #[test]
    fn unknown_curve_is_declined() {
        let mut buffer = Vec::new();
        CurveMSMId::from_curve_name("unregistered_curve")
            .serialize_uncompressed(&mut buffer)
            .unwrap();
        let len = buffer.len() as u32;
        assert_eq!(host_msm_unchecked(&mut buffer, len), 0);
        assert_eq!(buffer.len(), CURVE_ID_LEN);
    }

    #[test]
    fn table_aware_matches_plain() {
        let mut rng = test_rng();
        // Registered ("fixed") bases + variable, mixed up so the fixed set is not a contiguous
        // array.
        let num_fixed = 514;
        let num_variable = 64;
        let fixed: Vec<PallasAffine> = (0..num_fixed).map(|_| PallasAffine::rand(&mut rng)).collect();
        let var: Vec<PallasAffine> = (0..num_variable).map(|_| PallasAffine::rand(&mut rng)).collect();
        let mut bases = Vec::new();
        bases.extend_from_slice(&var[..20]);
        bases.extend_from_slice(&fixed);
        bases.extend_from_slice(&var[20..]);
        let scalars: Vec<PallasFr> = (0..bases.len()).map(|_| PallasFr::rand(&mut rng)).collect();

        let reference = Projective::<PallasConfig>::msm_unchecked(&bases, &scalars);

        clear_tables();
        assert_eq!(run(&bases, &scalars), reference, "plain host MSM");

        register_table::<PallasConfig>(&fixed);
        assert_eq!(run(&bases, &scalars), reference, "table-aware host MSM");

        clear_tables();
        assert_eq!(run(&bases, &scalars), reference, "after clear");
    }

    /// Sweep the fixed-base window `c` on a fixed base count (~ the DART affirmation fixed set,
    /// `2 + 2*256`): table build time, table memory, the filtering (`split`) time, the total
    /// table-aware MSM time, and the filtering share. Larger `c` = fewer windows = smaller table
    /// but a larger bucket array, so memory falls and eval time is U-shaped.
    #[test]
    fn table_gen_and_filter_costs() {
        let mut rng = test_rng();
        let n_bases = 514usize;
        let bases: Vec<PallasAffine> =
            (0..n_bases).map(|_| PallasAffine::rand(&mut rng)).collect();
        // MSM input = all tabled bases (all hit) + 80 variable ones.
        let mut input: Vec<PallasAffine> = bases.clone();
        input.extend((0..80).map(|_| PallasAffine::rand(&mut rng)));
        let scalars: Vec<PallasFr> =
            (0..input.len()).map(|_| PallasFr::rand(&mut rng)).collect();

        println!(
            "\n=== Fixed-base table: window-size sweep ({} bases, Pallas) ===",
            n_bases
        );
        println!(
            "{:>6} | {:>13} | {:>9} | {:>12} | {:>12} | {:>7}",
            "window", "build", "table MB", "filter", "total", "share"
        );
        let reps = 50u32;
        for &c in &[8usize, 10, 12, 14, 16, 18, 20] {
            let t = Instant::now();
            let table = CurveTable::<PallasConfig>::new_given_window_size(&bases, c);
            let build = t.elapsed();
            let mb = table.table_bytes() as f64 / (1024.0 * 1024.0);

            let t = Instant::now();
            for _ in 0..reps {
                let _ = table.split(&input, &scalars);
            }
            let filter = t.elapsed() / reps;

            let t = Instant::now();
            for _ in 0..reps {
                let _ = table.table_aware_msm(&input, &scalars);
            }
            let total = t.elapsed() / reps;

            println!(
                "{:>6} | {:>13?} | {:>9.2} | {:>12?} | {:>12?} | {:>6.1}%",
                c,
                build,
                mb,
                filter,
                total,
                100.0 * filter.as_secs_f64() / total.as_secs_f64()
            );
        }
    }
}
