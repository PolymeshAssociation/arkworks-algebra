use ark_ff::prelude::*;
use ark_std::{
    borrow::Borrow,
    cfg_chunks, cfg_into_iter, cfg_iter,
    iterable::Iterable,
    ops::{AddAssign, SubAssign},
    vec,
    vec::Vec,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub mod stream_pippenger;
pub use stream_pippenger::*;

use super::{ln_without_floats, ScalarMul};

#[cfg(all(
    target_has_atomic = "8",
    target_has_atomic = "16",
    target_has_atomic = "32",
    target_has_atomic = "64",
    target_has_atomic = "ptr"
))]
type DefaultHasher = ahash::AHasher;

#[cfg(not(all(
    target_has_atomic = "8",
    target_has_atomic = "16",
    target_has_atomic = "32",
    target_has_atomic = "64",
    target_has_atomic = "ptr"
)))]
type DefaultHasher = fnv::FnvHasher;

pub trait VariableBaseMSM: ScalarMul + for<'a> AddAssign<&'a Self::Bucket> {
    type Bucket: Default
        + Copy
        + Clone
        + for<'a> AddAssign<&'a Self::Bucket>
        + for<'a> SubAssign<&'a Self::Bucket>
        + AddAssign<Self::MulBase>
        + SubAssign<Self::MulBase>
        + for<'a> AddAssign<&'a Self::MulBase>
        + for<'a> SubAssign<&'a Self::MulBase>
        + Send
        + Sync
        + Into<Self>;

    const ZERO_BUCKET: Self::Bucket;

    /// This is used by the host MSM implementation to identify the curve type for MSM. If `None` is returned, the host MSM implementation will not be used.
    ///
    /// If the host MSM implementation doesn't support the curve, it will fall back to the slower non-host implementation.
    fn curve_name() -> Option<&'static str> {
        None
    }

    /// Computes an inner product between the [`PrimeField`] elements in `scalars`
    /// and the corresponding group elements in `bases`.
    ///
    /// If the elements have different length, it will chop the slices to the
    /// shortest length between `scalars.len()` and `bases.len()`.
    ///
    /// Reference: [`VariableBaseMSM::msm`]
    fn msm_unchecked(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Self {
        #[cfg(all(feature = "host_msm", not(feature = "std")))]
        if let Some(curve_name) = Self::curve_name() {
            if let Some(res) = ark_host_msm::use_host_msm_unchecked(curve_name, bases, scalars) {
                return res;
            }
            // fallback to non-host implementation if the host doesn't support this curve or if an error occurs during host MSM.
        }

        let bigints = cfg_into_iter!(scalars)
            .map(|s| s.into_bigint())
            .collect::<Vec<_>>();
        Self::msm_bigint(bases, bigints.as_slice())
    }

    /// Performs multi-scalar multiplication.
    ///
    /// # Warning
    ///
    /// This method checks that `bases` and `scalars` have the same length.
    /// If they are unequal, it returns an error containing
    /// the shortest length over which the MSM can be performed.
    fn msm(bases: &[Self::MulBase], scalars: &[Self::ScalarField]) -> Result<Self, usize> {
        (bases.len() == scalars.len())
            .then(|| Self::msm_unchecked(bases, scalars))
            .ok_or_else(|| bases.len().min(scalars.len()))
    }

    /// Optimized implementation of multi-scalar multiplication.
    fn msm_bigint(
        bases: &[Self::MulBase],
        bigints: &[<Self::ScalarField as PrimeField>::BigInt],
    ) -> Self {
        msm_signed(bases, bigints)
    }

    /// Performs multi-scalar multiplication when the scalars are known to be boolean.
    /// The default implementation is faster than [`Self::msm_bigint`].
    fn msm_u1(bases: &[Self::MulBase], scalars: &[bool]) -> Self {
        msm_binary(bases, scalars)
    }

    /// Performs multi-scalar multiplication when the scalars are known to be `u8`-sized.
    /// The default implementation is faster than [`Self::msm_bigint`].
    fn msm_u8(bases: &[Self::MulBase], scalars: &[u8]) -> Self {
        msm_u8(bases, scalars)
    }

    /// Performs multi-scalar multiplication when the scalars are known to be `u16`-sized.
    /// The default implementation is faster than [`Self::msm_bigint`].
    fn msm_u16(bases: &[Self::MulBase], scalars: &[u16]) -> Self {
        msm_u16(bases, scalars)
    }

    /// Performs multi-scalar multiplication when the scalars are known to be `u32`-sized.
    /// The default implementation is faster than [`Self::msm_bigint`].
    fn msm_u32(bases: &[Self::MulBase], scalars: &[u32]) -> Self {
        msm_u32(bases, scalars)
    }

    /// Performs multi-scalar multiplication when the scalars are known to be `u64`-sized.
    /// The default implementation is faster than [`Self::msm_bigint`].
    fn msm_u64(bases: &[Self::MulBase], scalars: &[u64]) -> Self {
        msm_u64(bases, scalars)
    }

    /// Streaming multi-scalar multiplication algorithm with hard-coded chunk
    /// size.
    fn msm_chunks<I, J>(bases_stream: &J, scalars_stream: &I) -> Self
    where
        I: Iterable + ?Sized,
        I::Item: Borrow<Self::ScalarField>,
        J: Iterable,
        J::Item: Borrow<Self::MulBase>,
    {
        assert!(scalars_stream.len() <= bases_stream.len());

        // remove offset
        let bases_init = bases_stream.iter();
        let mut scalars = scalars_stream.iter();

        // align the streams
        // TODO: change `skip` to `advance_by` once rust-lang/rust#7774 is fixed.
        // See <https://github.com/rust-lang/rust/issues/77404>
        let mut bases = bases_init.skip(bases_stream.len() - scalars_stream.len());
        let step: usize = 1 << 20;
        let mut result = Self::zero();
        for _ in 0..scalars_stream.len().div_ceil(step) {
            let bases_step = (&mut bases)
                .take(step)
                .map(|b| *b.borrow())
                .collect::<Vec<_>>();
            let scalars_step = (&mut scalars)
                .take(step)
                .map(|s| s.borrow().into_bigint())
                .collect::<Vec<_>>();
            result += Self::msm_bigint(bases_step.as_slice(), scalars_step.as_slice());
        }
        result
    }
}

#[inline]
fn large_value_unzip<A: Send + Sync, B: Send + Sync>(
    grouped: &[PackedIndex],
    f: impl Fn(usize) -> (A, B) + Send + Sync,
) -> (Vec<A>, Vec<B>) {
    cfg_iter!(grouped)
        .map(|&i| f(i.index()))
        .unzip::<_, _, Vec<_>, Vec<_>>()
}

#[inline]
fn small_value_unzip<A: Send + Sync, B: Send + Sync>(
    grouped: &[PackedIndex],
    f: impl Fn(usize, u16) -> (A, B) + Send + Sync,
) -> (Vec<A>, Vec<B>) {
    cfg_iter!(grouped)
        .map(|&i| f(i.index(), i.value()))
        .unzip::<_, _, Vec<_>, Vec<_>>()
}

#[inline(always)]
fn sub<B: BigInteger>(m: &B, scalar: &B) -> u64 {
    let mut negated = *m;
    negated.sub_with_borrow(scalar);
    negated.as_ref()[0]
}

// 44 zeroes, 1 in the next 16 bits, 0 rest
const VALUE_MASK: u64 = (u16::MAX as u64) << 44;

/// Represents groups for bit size of scalars.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarSize {
    /// if scalar is at most 1 bit
    U1 = 0,
    /// if -scalar is at most 1 bit
    NegU1 = 1,
    /// if scalar is bigger than 1 bit but at most 8 bits
    U8 = 2,
    /// if -scalar is bigger than 1 bit but at most 8 bits
    NegU8 = 3,
    /// if scalar is bigger than 8 bits but at most 16 bits
    U16 = 4,
    /// if -scalar is bigger than 8 bits but at most 16 bits
    NegU16 = 5,
    U32 = 6,
    NegU32 = 7,
    U64 = 8,
    NegU64 = 9,
    /// scalars in the range `[2^64, M-2^64)` where `M` is the modulus
    BigInt = 10,
}

impl ScalarSize {
    #[inline]
    fn partition_point(self, v: &[PackedIndex]) -> usize {
        v.partition_point(|i| i.group() < self as u8 + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedIndex(pub u64);

impl PackedIndex {
    #[inline(always)]
    fn new(index: usize, group: ScalarSize, value: u16) -> Self {
        // Pack the index, group, and value into a single u64 as [<44 bits for index> || <16 bits for value> || <4 bits for group>]
        // where LSB is the leftmost bit.
        let index_bits = ((index as u64) << 20) >> 20;
        let group_bits = (group as u64) << 60;
        let value_bits = (value as u64) << 44;

        PackedIndex(index_bits | value_bits | group_bits)
    }
    /// Extracts the index from the packed value.
    #[inline(always)]
    fn index(self) -> usize {
        ((self.0 << 20) >> 20) as usize
    }

    /// Extracts the group from the packed value.
    #[inline(always)]
    fn group(self) -> u8 {
        (self.0 >> 60) as u8
    }

    #[inline(always)]
    fn value(self) -> u16 {
        ((self.0 & VALUE_MASK) >> 44) as u16
    }
}

/// Computes multi-scalar multiplication where the scalars
/// can be negative, zero, or positive.
/// Tries to convert large scalars to negative (modulus - scalar) so that their bit size is small.
/// Partitions the scalars based on size and uses different algorithms based on the size
/// Uses wNAF when `V::NEGATION_IS_CHEAP` is `true`.
fn msm_signed<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = bases.len().min(scalars.len());
    let bases = &bases[..size];
    let scalars = &scalars[..size];

    // Partition scalars according to their size. For scalars (or -scalar) that fit in 16 bits,
    // store the value, rest wont fit in 64-bit PackedIndex.
    let mut grouped = cfg_iter!(scalars)
        .enumerate()
        .filter(|(_, scalar)| !scalar.is_zero())
        .map(|(i, scalar)| {
            use ScalarSize::*;
            let mut value = 0;
            let group = match scalar.num_bits() {
                0..=1 => U1,
                2..=8 => U8,
                9..=16 => U16,
                17..=32 => U32,
                33..=64 => U64,
                _ => {
                    // take bit size of -scalar
                    let mut p_minus_scalar = V::ScalarField::MODULUS;
                    p_minus_scalar.sub_with_borrow(scalar);
                    let group = match p_minus_scalar.num_bits() {
                        0..=1 => NegU1,
                        2..=8 => NegU8,
                        9..=16 => NegU16,
                        17..=32 => NegU32,
                        33..=64 => NegU64,
                        _ => ScalarSize::BigInt,
                    };
                    if matches!(group, NegU1 | NegU8 | NegU16) {
                        value = p_minus_scalar.as_ref()[0] as u16
                    }
                    group
                },
            };
            if matches!(group, U1 | U8 | U16) {
                value = (scalar.as_ref()[0]) as u16;
            };
            PackedIndex::new(i, group, value)
        })
        .collect::<Vec<_>>();

    #[cfg(feature = "parallel")]
    grouped.par_sort_unstable_by_key(|i| i.group());
    #[cfg(not(feature = "parallel"))]
    grouped.sort_unstable_by_key(|i| i.group());

    // Split scalars based on their bit sizes
    // u1s are scalars of 1-bit
    let (u1s, rest) = grouped.split_at(ScalarSize::U1.partition_point(&grouped));
    // i1s are scalars where negative of them is 1-bit
    let (i1s, rest) = rest.split_at(ScalarSize::NegU1.partition_point(rest));
    // u8s are scalars bigger than 1 bit but at most 8-bit
    let (u8s, rest) = rest.split_at(ScalarSize::U8.partition_point(rest));
    let (i8s, rest) = rest.split_at(ScalarSize::NegU8.partition_point(rest));
    let (u16s, rest) = rest.split_at(ScalarSize::U16.partition_point(rest));
    let (i16s, rest) = rest.split_at(ScalarSize::NegU16.partition_point(rest));
    let (u32s, rest) = rest.split_at(ScalarSize::U32.partition_point(rest));
    let (i32s, rest) = rest.split_at(ScalarSize::NegU32.partition_point(rest));
    let (u64s, rest) = rest.split_at(ScalarSize::U64.partition_point(rest));
    let (i64s, rest) = rest.split_at(ScalarSize::NegU64.partition_point(rest));
    // bigints are scalars in the range `[2^64, M-2^64)` where `M` is the modulus
    let (bigints, _) = rest.split_at(ScalarSize::BigInt.partition_point(rest));

    let m = V::ScalarField::MODULUS;
    // MSM contribution of positive scalars
    let mut add_result: V;
    // MSM contribution of negative scalars
    let mut sub_result: V;

    // Handle the scalars in the range {-1, 0, 1}.
    let (ub, us) = small_value_unzip(&u1s, |i, v| (bases[i], v == 1));
    let (ib, is) = small_value_unzip(&i1s, |i, v| (bases[i], v == 1));
    add_result = msm_binary::<V>(&ub, &us);
    sub_result = msm_binary::<V>(&ib, &is);

    // Handle positive and negative u8 scalars.
    let (ub, us) = small_value_unzip(u8s, |i, v| (bases[i], v as u8));
    let (ib, is) = small_value_unzip(i8s, |i, v| (bases[i], v as u8));
    add_result += msm_u8::<V>(&ub, &us);
    sub_result += msm_u8::<V>(&ib, &is);

    // Handle positive and negative u16 scalars.
    let (ub, us) = small_value_unzip(u16s, |i, v| (bases[i], v as u16));
    let (ib, is) = small_value_unzip(i16s, |i, v| (bases[i], v as u16));
    add_result += msm_u16::<V>(&ub, &us);
    sub_result += msm_u16::<V>(&ib, &is);

    // 32 and 64 bit negative scalars are not stored in PackedIndex so calculate them again
    // Handle positive and negative u32 scalars.
    let (ub, us) = large_value_unzip(u32s, |i| (bases[i], scalars[i].as_ref()[0] as u32));
    let (ib, is) = large_value_unzip(i32s, |i| (bases[i], sub(&m, &scalars[i]) as u32));
    add_result += msm_u32::<V>(&ub, &us);
    sub_result += msm_u32::<V>(&ib, &is);

    // Handle positive and negative u64 scalars.
    let (ub, us) = large_value_unzip(u64s, |i| (bases[i], scalars[i].as_ref()[0]));
    let (ib, is) = large_value_unzip(i64s, |i| (bases[i], sub(&m, &scalars[i])));
    add_result += msm_u64::<V>(&ub, &us);
    sub_result += msm_u64::<V>(&ib, &is);

    // Handle the rest of the scalars.
    let (bf, sf) = large_value_unzip(&bigints, |i| (bases[i], scalars[i]));
    if V::NEGATION_IS_CHEAP {
        add_result += msm_bigint_wnaf::<V>(&bf, &sf);
    } else {
        add_result += msm_bigint::<V>(&bf, &sf);
    }

    (add_result - sub_result).into()
}

fn preamble<A, B>(bases: &mut &[A], scalars: &mut &[B]) -> Option<usize> {
    let size = bases.len().min(scalars.len());
    if size == 0 {
        return None;
    }
    #[cfg(feature = "parallel")]
    let chunk_size = {
        let chunk_size = size / rayon::current_num_threads();
        if chunk_size == 0 {
            size
        } else {
            chunk_size
        }
    };
    #[cfg(not(feature = "parallel"))]
    let chunk_size = size;

    *bases = &bases[..size];
    *scalars = &scalars[..size];
    Some(chunk_size)
}

/// Computes multi-scalar multiplication where the scalars
/// lie in the range {-1, 0, 1}.
fn msm_binary<V: VariableBaseMSM>(mut bases: &[V::MulBase], mut scalars: &[bool]) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };

    // We only need to process the non-zero scalars.
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| {
            let mut res = V::ZERO_BUCKET;
            for (base, _) in bases.iter().zip(scalars).filter(|(_, &s)| s) {
                res += base;
            }
            res.into()
        })
        .sum()
}

fn msm_u8<V: VariableBaseMSM>(mut bases: &[V::MulBase], mut scalars: &[u8]) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial::<V>(bases, scalars))
        .sum()
}

fn msm_u16<V: VariableBaseMSM>(mut bases: &[V::MulBase], mut scalars: &[u16]) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial::<V>(bases, scalars))
        .sum()
}

fn msm_u32<V: VariableBaseMSM>(mut bases: &[V::MulBase], mut scalars: &[u32]) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial::<V>(bases, scalars))
        .sum()
}

fn msm_u64<V: VariableBaseMSM>(mut bases: &[V::MulBase], mut scalars: &[u64]) -> V {
    let chunk_size = match preamble(&mut bases, &mut scalars) {
        Some(chunk_size) => chunk_size,
        None => return V::zero(),
    };
    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| msm_serial::<V>(bases, scalars))
        .sum()
}

/// Combines per-window bucket sums into the final MSM result using Horner's method.
pub(crate) fn combine_window_sums<V: VariableBaseMSM>(window_sums: &[V::Bucket], window_size: usize) -> V {
    // Horner's rule
    window_sums
        .iter()
        .rev()
        .fold(V::zero(), |mut total, sum_i| {
            // total = total * 2^window_size
            for _ in 0..window_size {
                total.double_in_place();
            }

            total += sum_i;
            total
        })
}

/// Pippenger's bucket sum: prefix-sum reduction of a window's buckets into a single point:
/// returns `\sum_i{i * buckets[i-1]}` (bucket `i-1` holds the points whose digit is `i`),
/// computed by walking the buckets high-to-low while maintaining a running sum.
/// We do not normalize `buckets` to affine first: for the groups we care about
/// (Short Weierstrass, Twisted Edwards) mixed addition saves ~4 field muls per
/// addition, but batch normalization costs ~6 per element, so it is a net loss.
fn reduce_buckets<V: VariableBaseMSM>(buckets: Vec<V::Bucket>) -> V::Bucket {
    let mut running_sum = V::ZERO_BUCKET;
    let mut res = V::ZERO_BUCKET;
    buckets.into_iter().rev().for_each(|b| {
        running_sum += &b;
        res += &running_sum;
    });
    res
}

/// Computes one Pippenger window's contribution. `pairs` must already have the zero scalars filtered
/// out by the caller.
fn window_sum<'a, V, S, U, FX>(
    pairs: impl Iterator<Item = (&'a S, &'a V::MulBase)>,
    w_start: usize,
    window_size: usize,
    is_one: U,
    extract: FX,
) -> V::Bucket
where
    V: VariableBaseMSM,
    S: 'a,
    V::MulBase: 'a,
    U: Fn(&S) -> bool,
    FX: Fn(&S, usize, usize) -> u64,
{
    let mut res = V::ZERO_BUCKET;
    // No "zero" bucket: digit `d` in `[1, 2^c)` maps to `buckets[d - 1]`.
    let mut buckets = vec![V::ZERO_BUCKET; (1 << window_size) - 1];
    for (s, base) in pairs {
        if is_one(s) {
            // scalar = 1 contributes just base to the first window and 0 everywhere else.
            if w_start == 0 {
                res += base;
            }
        } else {
            // The `window_size` digit for this window. Its made of `window_size` bits starting from
            // offset `w_start`
            let digit = extract(s, w_start, window_size);
            if digit != 0 {
                buckets[(digit - 1) as usize] += base;
            }
        }
    }
    res += &reduce_buckets::<V>(buckets);
    res
}

/// Returns `true` if the [`BigInteger`] equals the integer `1`.
#[inline]
fn bigint_is_one<B: BigInteger>(b: &B) -> bool {
    let limbs = b.as_ref();
    limbs[0] == 1 && limbs[1..].iter().all(|&l| l == 0)
}

/// Compute msm using windowed non-adjacent form
pub fn msm_bigint_wnaf_parallel<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    bigints: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = bases.len().min(bigints.len());
    let scalars = &bigints[..size];
    let bases = &bases[..size];
    let num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;

    let c = window_size(size);

    let digits_count = num_bits.div_ceil(c);
    #[cfg(feature = "parallel")]
    let scalar_digits = scalars
        .into_par_iter()
        .flat_map_iter(|s| make_digits(s, c, num_bits))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "parallel"))]
    let scalar_digits = scalars
        .iter()
        .flat_map(|s| make_digits(s, c, num_bits))
        .collect::<Vec<_>>();

    // Bucket-array sizing. `make_digits` is a signed windowed encoding: every digit but the
    // most-significant lies in `[-2^(c-1), 2^(c-1)-1]`, so indexing buckets by `|digit| - 1`
    // needs only `2^(c-1)` slots. The most-significant window is the exception: it spans only
    // `ms_window_size = num_bits - c*(digits_count-1)` bits, and its (non-negative) digit can
    // reach `2^ms_window_size`. When `c` divides `num_bits` `ms_window_size = c`, so that digit can
    // exceed `2^(c-1)` and index past a half-sized array.

    // total size of all but most significant window
    let shift = (c * (digits_count - 1)) as u32;
    // size of most significant window
    let ms_window_size = num_bits - shift as usize;
    // value in the most significant window
    let mut ms_window_max = 1usize << ms_window_size;

    // Since all scalars are always smaller than the modulus, checking if modulus fits in to the
    // full window. If not then skip the optimization.
    let mod_shifted = V::ScalarField::MODULUS >> shift;
    let mod_limbs = mod_shifted.as_ref();
    if mod_limbs.iter().skip(1).all(|&l| l == 0) && mod_limbs[0] < (1u64 << c) {
        // +1 for carry
        ms_window_max = ms_window_max.min(mod_limbs[0] as usize + 1);
    }

    // number of buckets for the most significant window, as per above
    let ms_window_num_buckets = (1usize << (c - 1)).max(ms_window_max);
    // number of buckets for all except the most significant window
    let num_buckets = 1usize << (c - 1);

    let window_sums: Vec<_> = cfg_into_iter!(0..digits_count)
        .map(|i| {
            let mut buckets = vec![
                V::ZERO_BUCKET;
                if i == (digits_count-1) {ms_window_num_buckets} else {num_buckets}
            ];
            for (digits, base) in scalar_digits.chunks(digits_count).zip(bases) {
                use ark_std::cmp::Ordering;
                let scalar = digits[i];
                match 0.cmp(&scalar) {
                    Ordering::Less => buckets[(scalar - 1) as usize] += base,
                    Ordering::Greater => buckets[(-scalar - 1) as usize] -= base,
                    Ordering::Equal => (),
                }
            }

            reduce_buckets::<V>(buckets)
        })
        .collect();

    combine_window_sums::<V>(&window_sums, c)
}

#[cfg(feature = "parallel")]
const THREADS_PER_CHUNK: usize = 2;

/// Computes an MSM using the windowed non-adjacent form (WNAF) algorithm.
/// To improve parallelism, when number of threads is at least 2, this
/// function will split the input into enough chunks so that each chunk
/// can be processed with 2 threads.
/// Multi-scalar multiplication via windowed non-adjacent form, over the full
/// scalar width.
pub fn msm_bigint_wnaf<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    let size = bases.len().min(scalars.len());
    if size == 0 {
        return V::zero();
    }

    #[cfg(feature = "parallel")]
    let chunk_size = {
        let cur_num_threads = rayon::current_num_threads();
        let num_chunks = if cur_num_threads < THREADS_PER_CHUNK {
            1
        } else {
            cur_num_threads / THREADS_PER_CHUNK
        };
        let chunk_size = size / num_chunks;
        if chunk_size == 0 {
            size
        } else {
            chunk_size
        }
    };
    #[cfg(not(feature = "parallel"))]
    let chunk_size = size;

    bases = &bases[..size];
    scalars = &scalars[..size];

    cfg_chunks!(bases, chunk_size)
        .zip(cfg_chunks!(scalars, chunk_size))
        .map(|(bases, scalars)| {
            #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
            let result = rayon::ThreadPoolBuilder::new()
                .num_threads(THREADS_PER_CHUNK.min(rayon::current_num_threads()))
                .build()
                .unwrap()
                .install(|| msm_bigint_wnaf_parallel::<V>(bases, scalars));

            #[cfg(any(not(feature = "parallel"), target_arch = "wasm32"))]
            let result = msm_bigint_wnaf_parallel::<V>(bases, scalars);

            result
        })
        .sum()
}

/// Optimized implementation of multi-scalar multiplication.
pub fn msm_bigint<V: VariableBaseMSM>(
    mut bases: &[V::MulBase],
    mut scalars: &[<V::ScalarField as PrimeField>::BigInt],
) -> V {
    if preamble(&mut bases, &mut scalars).is_none() {
        return V::zero();
    }
    let size = scalars.len();
    let num_bits = V::ScalarField::MODULUS_BIT_SIZE as usize;
    let c = window_size(size);

    // Split each scalar into `c`-bit windows and accumulate each window's
    // contribution.
    let window_sums = cfg_into_iter!(0..num_bits)
        .step_by(c)
        .map(|w_start| {
            window_sum::<V, _, _, _>(
                scalars
                    .iter()
                    .zip(bases)
                    .filter(|(s, _)| !BigInteger::is_zero(*s)),
                w_start,
                c,
                bigint_is_one,
                |s, w_start, c| {
                    // Take the `c` bits at `w_start`: shift them down, keep the low limb mod 2^c.
                    let mut s = *s;
                    s >>= w_start as u32;
                    s.as_ref()[0] % (1u64 << c)
                },
            )
        })
        .collect::<Vec<_>>();

    combine_window_sums::<V>(&window_sums, c)
}

/// Serial (unsigned) Pippenger MSM for scalars that fit in a small unsigned integer. Similar
/// to [`msm_bigint`] but this drives the window loop serially its callers parallelize over base-chunks
/// instead and that it processes only the `size_of::<S>() * 8` bits the scalar type can hold.
pub fn msm_serial<V: VariableBaseMSM>(
    bases: &[V::MulBase],
    scalars: &[impl Into<u64> + Copy + Send + Sync],
) -> V {
    let size = bases.len();
    let c = window_size(size);

    // Scalars are passed as `u64`-convertible, so process the full 64-bit width
    // (matches the original `msm_serial`).
    let num_bits = core::mem::size_of::<u64>() * 8;

    // Split each scalar into windows and compute those scalar's contribution per window
    let window_sums: Vec<_> = (0..num_bits)
        .step_by(c)
        .map(|w_start| {
            window_sum::<V, _, _, _>(
                scalars
                    .iter()
                    .zip(bases)
                    .filter(|(&s, _)| Into::<u64>::into(s) != 0),
                w_start,
                c,
                |&s| Into::<u64>::into(s) == 1,
                |&s, w_start, c| (Into::<u64>::into(s) >> w_start) % (1u64 << c),
            )
        })
        .collect();

    combine_window_sums::<V>(&window_sums, c)
}

// From: https://github.com/arkworks-rs/gemini/blob/main/src/kzg/msm/variable_base.rs#L20
pub(crate) fn make_digits(
    a: &impl BigInteger,
    w: usize,
    num_bits: usize,
) -> impl Iterator<Item = i64> + '_ {
    let scalar = a.as_ref();
    let radix: u64 = 1 << w;
    let window_mask: u64 = radix - 1;

    let mut carry = 0u64;
    let num_bits = if num_bits == 0 {
        a.num_bits() as usize
    } else {
        num_bits
    };
    let digits_count = num_bits.div_ceil(w);

    (0..digits_count).map(move |i| {
        // Construct a buffer of bits of the scalar, starting at `bit_offset`.
        let bit_offset = i * w;
        let u64_idx = bit_offset / 64;
        let bit_idx = bit_offset % 64;
        // Read the bits from the scalar
        let bit_buf = if bit_idx < 64 - w || u64_idx == scalar.len() - 1 {
            // This window's bits are contained in a single u64,
            // or it's the last u64 anyway.
            scalar[u64_idx] >> bit_idx
        } else {
            // Combine the current u64's bits with the bits from the next u64
            (scalar[u64_idx] >> bit_idx) | (scalar[1 + u64_idx] << (64 - bit_idx))
        };

        // Read the actual coefficient value from the window
        let coef = carry + (bit_buf & window_mask); // coef = [0, 2^r)

        // Recenter coefficients from [0,2^w) to [-2^w/2, 2^w/2)
        carry = (coef + radix / 2) >> w;
        let mut digit = (coef as i64) - (carry << w) as i64;

        if i == digits_count - 1 {
            digit += (carry << w) as i64;
        }
        digit
    })
}

const fn window_size(num_scalars: usize) -> usize {
    if num_scalars < 32 {
        3
    } else {
        ln_without_floats(num_scalars) + 2
    }
}