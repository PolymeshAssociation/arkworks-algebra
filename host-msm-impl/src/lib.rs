#![cfg_attr(not(feature = "std"), no_std)]

use ark_bls12_381::{g1::Config as Bls12_381G1Config, g2::Config as Bls12_381G2Config};
use ark_bn254::{g1::Config as Bn254G1Config, g2::Config as Bn254G2Config};
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

    /// Most rayon threads at which the table path is used. On Pallas (Apple M3 Max, 514 to 32768
    /// tabled bases plus 0 or 80 others) the table is 0.69x to 0.93x of the plain MSM's time on
    /// one thread and mostly below 1x up to 8, but on 12 and 16 threads it is 1.03x to 1.17x at
    /// 514 and from 8192 bases, where the plain MSM's GLV split and window parallelism win.
    #[cfg(feature = "parallel")]
    const MAX_THREADS_FOR_TABLE: usize = 8;

    /// Whether the caller's rayon pool is small enough for the table path to pay.
    fn table_pays() -> bool {
        #[cfg(feature = "parallel")]
        return rayon::current_num_threads() <= MAX_THREADS_FOR_TABLE;
        #[cfg(not(feature = "parallel"))]
        true
    }

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

        /// The table-aware MSM: the fixed part goes through the fixed-base tables, alongside the
        /// variable part.
        pub fn table_aware_msm(
            &self,
            bases: &[Affine<P>],
            scalars: &[P::ScalarField],
        ) -> Projective<P> {
            let (fixed, var_bases, var_scalars) = self.split(bases, scalars);
            self.msm_split(&fixed, &var_bases, &var_scalars)
        }

        fn msm_split(
            &self,
            fixed: &[P::ScalarField],
            var_bases: &[Affine<P>],
            var_scalars: &[P::ScalarField],
        ) -> Projective<P> {
            let fixed_part = || self.tables.msm(fixed);
            let var_part = || msm_batch_affine::<P>(var_bases, var_scalars);
            #[cfg(feature = "parallel")]
            let (f, v) = rayon::join(fixed_part, var_part);
            #[cfg(not(feature = "parallel"))]
            let (f, v) = (fixed_part(), var_part());
            f + v
        }
    }

    impl<P: SWCurveConfig> HostTable for CurveTable<P> {
        fn msm(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
            let (bases, scalars) = match super::read_msm_input::<P>(buffer, buf_len as usize) {
                Some(input) => input,
                None => return 0,
            };
            if bases.len().min(scalars.len()) < MIN_BASES_FOR_TABLE || !table_pays() {
                return super::write_msm_result(super::plain_msm::<P>(&bases, &scalars), buffer);
            }
            let (fixed, var_bases, var_scalars) = self.split(&bases, &scalars);
            let matched = bases.len().min(scalars.len()) - var_bases.len();
            let res = if matched < MIN_BASES_FOR_TABLE {
                super::plain_msm::<P>(&bases, &scalars)
            } else {
                self.msm_split(&fixed, &var_bases, &var_scalars)
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
    /// init. Returns `false`, registering nothing, when the host MSM does not serve `P`
    /// (`RegisteredCurves::new` lists the curves it serves). A curve with no table uses the plain
    /// MSM.
    pub fn register_table<P: SWCurveConfig>(bases: &[Affine<P>]) -> bool {
        is_served::<P>() && insert_table(CurveTable::new(bases))
    }

    /// [`register_table`] but with an explicit fixed-base window size instead of the one chosen
    /// by arkworks. A smaller `c` gives more windows — a larger table but a smaller per-window
    /// bucket array. Lets a benchmark compare table memory and eval time across window sizes.
    pub fn register_table_with_given_size<P: SWCurveConfig>(
        bases: &[Affine<P>],
        window_size: usize,
    ) -> bool {
        is_served::<P>() && insert_table(CurveTable::new_given_window_size(bases, window_size))
    }

    /// Whether the host MSM serves `P`, the only curves whose tables it consults.
    fn is_served<P: SWCurveConfig>() -> bool {
        <Projective<P> as VariableBaseMSM>::curve_name()
            .is_some_and(|n| super::SUPPORTED_CURVES.serves(&CurveMSMId::from_curve_name(n)))
    }

    fn insert_table<P: SWCurveConfig>(table: CurveTable<P>) -> bool {
        let Some(name) = <Projective<P> as VariableBaseMSM>::curve_name() else {
            return false;
        };
        let curve_id = CurveMSMId::from_curve_name(name);
        let table: Arc<dyn HostTable> = Arc::new(table);
        match TABLES.write() {
            Ok(mut tables) => {
                tables.insert(curve_id, table);
                true
            },
            Err(_) => false,
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
    /// Pallas, Vesta, and G1 and G2 of BLS12-381 and BN254.
    pub fn new() -> Self {
        let mut curves = RegisteredCurves {
            curves: BTreeMap::new(),
        };
        let registered = [
            curves.register_curve::<PallasConfig>(),
            curves.register_curve::<VestaConfig>(),
            curves.register_curve::<Bls12_381G1Config>(),
            curves.register_curve::<Bls12_381G2Config>(),
            curves.register_curve::<Bn254G1Config>(),
            curves.register_curve::<Bn254G2Config>(),
        ];
        debug_assert!(registered.iter().all(|r| *r), "host MSM curve IDs collide");
        curves
    }

    /// Serves `P` under its [`SWCurveConfig::curve_name`] ID. Returns `false`, registering
    /// nothing, when `P` has no name or another curve already holds the ID, as G1 and G2 of one
    /// crate do under the default name.
    pub fn register_curve<P: SWCurveConfig + 'static>(&mut self) -> bool {
        let Some(name) = <Projective<P> as VariableBaseMSM>::curve_name() else {
            return false;
        };
        let curve_id = CurveMSMId::from_curve_name(name);
        if self.curves.contains_key(&curve_id) {
            return false;
        }
        self.curves
            .insert(curve_id, Box::new(host_msm_unchecked_impl::<P>));
        true
    }

    /// Whether a curve is registered under `curve_id`.
    pub fn serves(&self, curve_id: &CurveMSMId) -> bool {
        self.curves.contains_key(curve_id)
    }

    /// Returns `1` for a probe (`buf_len == CURVE_ID_LEN`) of a registered curve, the serialized
    /// result length for an MSM request, and `0` for an unregistered curve, a malformed request,
    /// or `buf_len` outside `CURVE_ID_LEN..=buffer.len()`.
    pub fn msm_unchecked(&self, buffer: &mut [u8], buf_len: u32) -> u32 {
        if (buf_len as usize) < CURVE_ID_LEN || buf_len as usize > buffer.len() {
            return 0;
        }
        if let Some(curve_id) = CurveMSMId::deserialize_uncompressed_unchecked(&buffer[..]).ok() {
            if let Some(msm_fn) = self.curves.get(&curve_id) {
                return if buf_len as usize > CURVE_ID_LEN {
                    // Prefer the fixed-base table path when a table is registered for this curve.
                    #[cfg(feature = "std")]
                    if let Some(res_len) = table_cache::try_table_msm(&curve_id, buffer, buf_len) {
                        return res_len;
                    }
                    msm_fn(buffer, buf_len)
                } else {
                    1 // Curve is supported, but no MSM data provided
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
static SUPPORTED_CURVES: core::sync::atomic::AtomicPtr<RegisteredCurves> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

/// The registry, built on first use. Racing first calls each build one and the loser frees its
/// copy, so every caller sees the same `'static` registry.
#[cfg(not(feature = "std"))]
fn get_supported_curves() -> &'static RegisteredCurves {
    use core::sync::atomic::Ordering;
    let current = SUPPORTED_CURVES.load(Ordering::Acquire);
    if !current.is_null() {
        // SAFETY: a non-null pointer came from `Box::into_raw` below and is never freed.
        return unsafe { &*current };
    }
    let fresh = Box::into_raw(Box::new(RegisteredCurves::new()));
    match SUPPORTED_CURVES.compare_exchange(
        core::ptr::null_mut(),
        fresh,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        // SAFETY: `fresh` is now owned by the static and never freed.
        Ok(_) => unsafe { &*fresh },
        Err(winner) => {
            // SAFETY: `fresh` was never shared; `winner` is owned by the static.
            drop(unsafe { Box::from_raw(fresh) });
            unsafe { &*winner }
        },
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

/// Below this base count [`plain_msm`] uses `msm_unchecked` instead of `msm_batch_affine`. There
/// `msm_batch_affine` is up to 1.9x slower on BLS12-381 and BN254 G1 and up to 1.25x on their
/// G2, and within 4% on Pallas. From 32 bases it ties or is up to 18% faster.
const MIN_BASES_FOR_BATCH_AFFINE: usize = 32;

/// MSM without a fixed-base table. `msm_batch_affine` from [`MIN_BASES_FOR_BATCH_AFFINE`] bases,
/// `msm_unchecked` below.
fn plain_msm<P: SWCurveConfig>(bases: &[Affine<P>], scalars: &[P::ScalarField]) -> Projective<P> {
    if bases.len() < MIN_BASES_FOR_BATCH_AFFINE {
        Projective::<P>::msm_unchecked(bases, scalars)
    } else {
        msm_batch_affine::<P>(bases, scalars)
    }
}

/// The stateless (no fixed-base table) host MSM for curve `P`, registered under the extern
/// host-function name `host_msm_unchecked`.
///
/// NOTE: despite the `unchecked` in the name — and despite `VariableBaseMSM::msm_unchecked` (via
/// `msm_unchecked_inner`) being the natural dispatch for this entry point — this computes the MSM
/// with [`plain_msm`], which uses `msm_batch_affine` from [`MIN_BASES_FOR_BATCH_AFFINE`] bases,
/// where it benchmarks faster than `msm_unchecked`. Avoiding adding new host functions, so the
/// batch-affine path is added to this existing entry point rather than exposed as a separate one.
fn host_msm_unchecked_impl<P: SWCurveConfig>(buffer: &mut [u8], buf_len: u32) -> u32 {
    if buf_len as usize == CURVE_ID_LEN {
        // The curve is supported.
        return 1;
    }
    let (bases, scalars) = match read_msm_input::<P>(buffer, buf_len as usize) {
        Some(input) => input,
        None => return 0,
    };
    write_msm_result(plain_msm::<P>(&bases, &scalars), buffer)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::{
        clear_tables, host_msm_unchecked, register_table, Bls12_381G1Config, Bls12_381G2Config,
        Bn254G1Config, Bn254G2Config, VestaConfig,
    };
    use ark_ec::short_weierstrass::{Affine, Projective, SWCurveConfig};
    use ark_ec::{CurveGroup, VariableBaseMSM};
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

    /// Host MSM of curve `P` through the host function, against the in-process MSM.
    fn round_trip<P: SWCurveConfig>() {
        let mut rng = test_rng();
        let name = <Projective<P> as VariableBaseMSM>::curve_name().unwrap();
        let curve_id = CurveMSMId::from_curve_name(name);

        let mut probe = Vec::new();
        curve_id.serialize_uncompressed(&mut probe).unwrap();
        assert_eq!(host_msm_unchecked(&mut probe, CURVE_ID_LEN as u32), 1, "{name}");

        for n in [1usize, 2, 64, 65, 300] {
            let bases: Vec<Affine<P>> = (0..n).map(|_| Projective::<P>::rand(&mut rng).into_affine()).collect();
            let scalars: Vec<P::ScalarField> = (0..n).map(|_| P::ScalarField::rand(&mut rng)).collect();
            let mut buf = Vec::new();
            curve_id.serialize_uncompressed(&mut buf).unwrap();
            bases.serialize_uncompressed(&mut buf).unwrap();
            scalars.serialize_uncompressed(&mut buf).unwrap();
            let len = buf.len() as u32;
            let res_len = host_msm_unchecked(&mut buf, len) as usize;
            let got = Projective::<P>::deserialize_uncompressed_unchecked(&buf[..res_len]).unwrap();
            assert_eq!(got, Projective::<P>::msm_unchecked(&bases, &scalars), "{name}, n = {n}");
        }
    }

    #[test]
    fn registered_curves_round_trip() {
        clear_tables();
        round_trip::<PallasConfig>();
        round_trip::<VestaConfig>();
        round_trip::<Bls12_381G1Config>();
        round_trip::<Bls12_381G2Config>();
        round_trip::<Bn254G1Config>();
        round_trip::<Bn254G2Config>();
    }

    /// G1 and G2 of a pairing-friendly curve have distinct IDs, and Pallas and Vesta keep theirs.
    #[test]
    fn curve_ids_are_distinct() {
        fn id<P: SWCurveConfig>() -> CurveMSMId {
            CurveMSMId::from_curve_name(<Projective<P> as VariableBaseMSM>::curve_name().unwrap())
        }
        assert_eq!(id::<PallasConfig>(), CurveMSMId::from_curve_name("pallas"));
        assert_eq!(id::<VestaConfig>(), CurveMSMId::from_curve_name("vesta"));
        assert_eq!(id::<Bls12_381G1Config>(), CurveMSMId::from_curve_name("bls12_381_g1"));
        assert_eq!(id::<Bls12_381G2Config>(), CurveMSMId::from_curve_name("bls12_381_g2"));
        assert_eq!(id::<Bn254G1Config>(), CurveMSMId::from_curve_name("bn254_g1"));
        assert_eq!(id::<Bn254G2Config>(), CurveMSMId::from_curve_name("bn254_g2"));
    }

    /// `buf_len` past the end of `buffer` returns `0` instead of panicking.
    #[test]
    fn buf_len_beyond_buffer_is_declined() {
        let mut rng = test_rng();
        let bases: Vec<PallasAffine> = (0..4).map(|_| PallasAffine::rand(&mut rng)).collect();
        let scalars: Vec<PallasFr> = (0..4).map(|_| PallasFr::rand(&mut rng)).collect();
        let mut buf = build_buffer(&bases, &scalars);
        let len = buf.len() as u32;
        assert_eq!(host_msm_unchecked(&mut buf, len + 1), 0);
        assert_eq!(host_msm_unchecked(&mut buf, u32::MAX), 0);
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

        assert!(register_table::<PallasConfig>(&fixed));
        assert_eq!(run(&bases, &scalars), reference, "table-aware host MSM");
        // Both sides of the thread gate, and a request matching fewer tabled bases than the
        // table needs.
        #[cfg(feature = "parallel")]
        for threads in [1usize, 16] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| assert_eq!(run(&bases, &scalars), reference, "{threads} threads"));
        }
        let few = [&fixed[..100], &var[..]].concat();
        let few_scalars = &scalars[..few.len()];
        assert_eq!(
            run(&few, few_scalars),
            Projective::<PallasConfig>::msm_unchecked(&few, few_scalars),
            "few tabled bases"
        );

        clear_tables();
        assert_eq!(run(&bases, &scalars), reference, "after clear");
    }

    /// Sweep the fixed-base window `c` on a fixed base count (~ the DART affirmation fixed set,
    /// `2 + 2*256`): table build time, table memory, the filtering (`split`) time, the total
    /// table-aware MSM time, and the filtering share. Larger `c` = fewer windows = smaller table
    /// but a larger bucket array, so memory falls and eval time is U-shaped.
    #[test]
    #[ignore]
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
