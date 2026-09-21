//! Fixed-base multiscalar multiplication by precomputation, the Brickell–Gordon–McCurley–Wilson
//! method (`[BGMW92]`).
//!
//! When the same base set `[G_0, .., G_{n-1}]` is reused across many multiscalar multiplications,
//! precompute for each base and each radix-`2^c` window `j` the multiple `G_i^{(j)} = 2^{c j} G_i`.
//! Then
//!
//! ```
//! \sum_i k_i G_i  =  \sum_i \sum_j d_{i,j} G_i^{(j)}
//! ```
//!
//! where `d_{i,j}` are the signed radix-`2^c` digits of `k_i`, is a single multiscalar
//! multiplication over the `n W` precomputed points. The place value is baked into each point, so
//! every window shares one bucket array and one reduction. Pippenger reduces `2^c` buckets once per
//! window; this reduces them once total, which buys a wider `c` (fewer windows, fewer accumulation
//! additions) at the same reduction cost. This is the "Pippenger variant `[BGMW95]`" row of `[LFG23]`
//! Table 1 (storage `nh`, cost `nh + q/2`), and the shifted-multiples precompute ICICLE exposes as
//! `precompute_factor` (`[ICICLE]`).
//!
//! Under `parallel` the one bucket array is split into equal bucket ranges. Per-thread histograms
//! give the bucket sizes and bin each nonzero entry's index by thread group; each group then
//! gathers its ranges' points from the table, in table order, into its contiguous slice of one
//! bucket-sorted buffer; each range reduces and running-sums on its own thread with reused
//! staging; the ranges combine with additions and one short scalar multiplication. Costs: a table
//! of `n W` affine points (`64 n W` bytes, tens of MB at `n ~ 2^13`), a one-time precompute of
//! about `n MODULUS_BIT_SIZE` doublings, and per call `8 n W` bytes of digits and indices plus
//! `64 n W` bytes of bucket-sorted points.
//!
//! # References
//!
//! - `[BGMW92]` E. Brickell, D. Gordon, K. McCurley, D. Wilson, "Fast Exponentiation with
//!   Precomputation", EUROCRYPT 1992, LNCS 658, pp. 200-207.
//!   <https://link.springer.com/chapter/10.1007/3-540-47555-9_18>
//! - `[LFG23]` G. Luo, S. Fu, G. Gong, "Speeding Up Multi-Scalar Multiplication over Fixed Points
//!   Towards Efficient zkSNARKs", TCHES 2023(2), pp. 358-380, Table 1 (the "Pippenger variant
//!   `[BGMW95]`" row). <https://tches.iacr.org/index.php/TCHES/article/view/10287>
//! - `[ICICLE]` Ingonyama, ICICLE MSM precomputation (the `precompute_factor` shifted-multiples
//!   table `[2^l P, 2^{2l} P, ..]`). <https://dev.ingonyama.com/2.8.0/icicle/primitives/msm>

use crate::{
    scalar_mul::{
        sw_pippenger::{counting_sort, reduce_tree, running_bucket_sum},
        variable_base::make_digits,
    },
    short_weierstrass::{Affine, Projective, SWCurveConfig},
    AdditiveGroup, AffineRepr, CurveGroup,
};
use ark_ff::{PrimeField, Zero};
use ark_std::vec::Vec;
use itertools::Either;

#[cfg(feature = "parallel")]
use crate::{
    scalar_mul::sw_pippenger::{reduce_tree_with, signed_point, Point, ReduceScratch},
    short_weierstrass::Bucket,
};
#[cfg(feature = "parallel")]
use ark_std::vec;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Fewest table entries (`n*W`) the parallel evaluation fans out for. Below it one call is a few
/// hundred microseconds.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_ENTRIES: usize = 1 << 13;

/// Bucket-range chunks per thread in the parallel evaluation.
#[cfg(feature = "parallel")]
const CHUNKS_PER_THREAD: usize = 4;

/// Per-thread growth of the bucket-term weight in [`best_window_parallel`]. At `t` threads the
/// weight is `1 + (t-1) * PAR_BUCKET_SLOPE`. Tested on a 16-thread host (weight ~= 6) so the
/// parallel window stays at least as fast as a variable-base MSM across 2^10..2^16 bases.
/// Recheck via the `fixed_base_window_bench` test on other hosts.
#[cfg(feature = "parallel")]
const PAR_BUCKET_SLOPE: f64 = 0.35;

/// Floor on the parallel window: below it the per-window paralle and the narrower batch inversion
/// lose at small base counts (a measured regression at 2^10 bases). Capped at the serial pick so a
/// tiny field is unaffected.
#[cfg(feature = "parallel")]
const PAR_MIN_WINDOW: usize = 12;


/// A precomputed table of place-value multiples of a fixed base set.
pub struct FixedBaseMSM<P: SWCurveConfig> {
    /// Radix `2^c`.
    c: usize,
    /// The multiples stored per base.
    num_windows: usize,
    /// Size of the shared bucket array, covering the largest digit any window produces.
    num_buckets: usize,
    /// Number of bases.
    n: usize,
    /// Tables in base-major form: `table[i * num_windows + j] = 2^{c*j} G_i`.
    table: Vec<Affine<P>>,
}

impl<P: SWCurveConfig> FixedBaseMSM<P> {
    /// Builds a table with the window `c` minimizing the modeled addition count
    /// `n ceil(bits/c) + 2^{c-1}`.
    pub fn new(bases: &[Affine<P>]) -> Self {
        let n = bases.len().max(1);
        #[cfg(feature = "parallel")]
        let c = best_window_parallel::<P>(n, rayon::current_num_threads());
        #[cfg(not(feature = "parallel"))]
        let c = best_window::<P>(n);
        Self::new_given_window_size(bases, c)
    }

    /// Builds a table with an explicit window `c` (`2 <= c < 31`). Larger `c` gives fewer windows
    /// (a smaller table) but a `2^{c-1}`-entry bucket array.
    pub fn new_given_window_size(bases: &[Affine<P>], c: usize) -> Self {
        assert!(2 <= c && c < 31, "window size out of range");
        let n = bases.len();
        let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let num_windows = num_bits.div_ceil(c);
        let num_buckets = bucket_count::<P>(c, num_windows, num_bits);

        // Per base, `num_windows` multiples by `c` doublings each.
        let base_mults = |b: &Affine<P>| base_multiples(b, num_windows, c);

        #[cfg(feature = "parallel")]
        let proj: Vec<Projective<P>> = bases.par_iter().flat_map_iter(base_mults).collect();

        #[cfg(not(feature = "parallel"))]
        let proj: Vec<Projective<P>> = bases.iter().flat_map(base_mults).collect();

        let table = Projective::<P>::normalize_batch(&proj);

        Self {
            c,
            num_windows,
            num_buckets,
            n,
            table,
        }
    }

    /// The modeled-best window if its table fits in `max_bytes`, else the next wider `c` (fewer
    /// windows, a smaller table, a larger bucket array) that fits. `None` if even `c = 30` does
    /// not.
    pub fn new_given_size_limit(bases: &[Affine<P>], max_bytes: usize) -> Option<Self> {
        let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let entry = core::mem::size_of::<Affine<P>>();
        let fits = |c: usize| {
            bases
                .len()
                .saturating_mul(num_bits.div_ceil(c))
                .saturating_mul(entry)
                <= max_bytes
        };
        (best_window::<P>(bases.len().max(1))..31)
            .find(|&c| fits(c))
            .map(|c| Self::new_given_window_size(bases, c))
    }

    /// `sum_i scalars[i] G_i` over the prepared base set. `scalars` is truncated to the base count.
    pub fn msm(&self, scalars: &[P::ScalarField]) -> Projective<P> {
        let bigints = scalars.iter().map(|s| s.into_bigint()).collect::<Vec<_>>();
        self.msm_bigint(&bigints)
    }

    /// [`Self::msm`] taking scalars already reduced to `BigInt`. The batch-affine reduction cannot
    /// fail over a prime field of characteristic > 2, so there is no fallback path.
    pub fn msm_bigint(&self, scalars: &[<P::ScalarField as PrimeField>::BigInt]) -> Projective<P> {
        let n = scalars.len().min(self.n);
        if n == 0 {
            return Projective::zero();
        }
        let scalars = &scalars[..n];
        #[cfg(feature = "parallel")]
        if n * self.num_windows >= MIN_PARALLEL_ENTRIES && rayon::current_num_threads() > 1 {
            return self.par_eval(scalars);
        }
        self.serial_eval(scalars)
    }

    /// One counting sort of every nonzero entry into the shared buckets, keyed by digit magnitude,
    /// then one tree reduction and one running sum.
    fn serial_eval(
        &self,
        scalars: &[<P::ScalarField as PrimeField>::BigInt],
    ) -> Projective<P> {
        let (c, m) = (self.c, self.num_buckets);
        let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let digits = scalars
            .iter()
            .enumerate()
            .flat_map(|(i, s)| {
                if self.non_zero(i) {
                    Either::Left(make_digits(s, c, num_bits))
                } else {
                    Either::Right(ark_std::iter::repeat_n(0i64, self.num_windows))
                }
            })
            .collect::<Vec<_>>();

        // A digit's index is its table index: both are base-major with `W` per base.
        let (mut offsets, mut points) =
            counting_sort::<P>(m, digits.len(), |idx| (digits[idx], self.table[idx]));

        reduce_tree::<P>(&mut points, &mut offsets, false);
        running_bucket_sum::<P>(&points, &offsets).into()
    }

    /// The bucket array split into `k` equal ranges of `L` buckets. Segment `s` covers buckets
    /// `[s*L, (s+1)*L)` and bucket `b` holds magnitude `b + 1`, so with a local running sum
    /// `L_s = \sum_b (b - s*L + 1) B_b` and plain sum `S_s = \sum_b B_b` the total is
    /// `\sum_s L_s + L * \sum_s s S_s`, the last sum a running sum over the chunk totals.
    #[cfg(feature = "parallel")]
    fn par_eval(
        &self,
        scalars: &[<P::ScalarField as PrimeField>::BigInt],
    ) -> Projective<P> {
        let (c, w, m) = (self.c, self.num_windows, self.num_buckets);
        let n = scalars.len();
        let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
        let threads = rayon::current_num_threads().max(1);

        // Digits in place, base-major, so a digit's index is its table index.
        let mut digits = vec![0i32; n * w];
        let per = n.div_ceil(threads).max(1);
        digits
            .par_chunks_mut(per * w)
            .zip(scalars.par_chunks(per))
            .enumerate()
            .for_each(|(ci, (dch, sch))| {
                for (t, s) in sch.iter().enumerate() {
                    if self.non_zero(ci * per + t) {
                        for (j, d) in make_digits(s, c, num_bits).enumerate() {
                            dch[t * w + j] = d as i32;
                        }
                    }
                }
            });

        // `k` equal chunks of `L` buckets; a thread group owns `CHUNK_PER_THREAD`
        // consecutive chunks, so a group's bucket range is `group_len` long and a bucket's
        // group is `b / group_len`.
        let k = (threads * CHUNKS_PER_THREAD).clamp(1, m);
        let seg_len = m.div_ceil(k);
        let segs: Vec<(usize, usize)> = (0..k)
            .map(|s| (s * seg_len, ((s + 1) * seg_len).min(m)))
            .filter(|&(lo, hi)| lo < hi)
            .collect();
        let groups: Vec<&[(usize, usize)]> = segs.chunks(CHUNKS_PER_THREAD).collect();
        let group_len = seg_len * CHUNKS_PER_THREAD;
        let num_groups = groups.len();

        // Per digit chunk: the bucket histogram, and the nonzero entries' indices binned by
        // group in index order, so a group's gather below walks the table monotonically.
        let chunk = digits.len().div_ceil(threads).max(1);
        let binned: Vec<(Vec<u32>, Vec<Vec<u32>>)> = digits
            .par_chunks(chunk)
            .enumerate()
            .map(|(ci, ch)| {
                let mut h = vec![0u32; m];
                let mut lists: Vec<Vec<u32>> = (0..num_groups)
                    .map(|_| Vec::with_capacity(ch.len() / num_groups + 16))
                    .collect();
                for (t, &d) in ch.iter().enumerate() {
                    if d != 0 {
                        let b = d.unsigned_abs() as usize - 1;
                        h[b] += 1;
                        lists[b / group_len].push((ci * chunk + t) as u32);
                    }
                }
                (h, lists)
            })
            .collect();
        let mut offsets = vec![0usize; m + 1];
        for (h, _) in &binned {
            for (o, &x) in offsets[1..].iter_mut().zip(h) {
                *o += x as usize;
            }
        }
        for b in 0..m {
            offsets[b + 1] += offsets[b];
        }

        // One bucket-sorted buffer. A group's consecutive chunks are one contiguous slice of
        // it, filled from the group's index lists.
        let mut points = vec![Point::default(); offsets[m]];
        let mut slices: Vec<&mut [Point<P::BaseField>]> = Vec::with_capacity(num_groups);
        let mut rest = points.as_mut_slice();
        let mut cursor = 0;
        for group in &groups {
            let end = offsets[group[group.len() - 1].1];
            let (head, tail) = rest.split_at_mut(end - cursor);
            slices.push(head);
            rest = tail;
            cursor = end;
        }
        groups
            .par_iter()
            .zip(slices.into_par_iter())
            .enumerate()
            .for_each(|(g, (group, slice))| {
                let (glo, ghi) = (group[0].0, group[group.len() - 1].1);
                let base = offsets[glo];
                let mut positions: Vec<usize> = offsets[glo..ghi].iter().map(|o| o - base).collect();
                for (_, lists) in &binned {
                    for &idx in &lists[g] {
                        let idx = idx as usize;
                        let d = digits[idx];
                        let pos = &mut positions[d.unsigned_abs() as usize - 1 - glo];
                        slice[*pos] = signed_point(&self.table[idx], d < 0);
                        *pos += 1;
                    }
                }
            });

        let parts: Vec<(Bucket<P>, Bucket<P>)> = segs
            .par_iter()
            .map_init(ReduceScratch::<P::BaseField>::default, |scratch, &(lo, hi)| {
                let base = offsets[lo];
                let local: Vec<usize> = offsets[lo..=hi].iter().map(|o| o - base).collect();
                let (pts, offs) =
                    reduce_tree_with::<P>(&points[base..offsets[hi]], &local, false, scratch);
                let nb = offs.len() - 1;
                let mut running = Bucket::<P>::ZERO;
                let mut ls = Bucket::<P>::ZERO;
                let mut ss = Bucket::<P>::ZERO;
                for bi in (0..nb).rev() {
                    if offs[bi + 1] > offs[bi] {
                        let p = pts[offs[bi]];
                        let aff = Affine::<P>::new_unchecked(p.x, p.y);
                        running += aff;
                        ss += aff;
                    }
                    ls += &running;
                }
                (ls, ss)
            })
            .collect();

        let mut sum_ls = Bucket::<P>::ZERO;
        let mut running = Bucket::<P>::ZERO;
        let mut weighted = Bucket::<P>::ZERO;
        for (s, (ls, ss)) in parts.into_iter().enumerate().rev() {
            sum_ls += &ls;
            if s > 0 {
                running += &ss;
                weighted += &running;
            }
        }
        Projective::<P>::from(sum_ls)
            + Projective::<P>::from(weighted) * P::ScalarField::from(seg_len as u64)
    }

    /// The i-th base zero or not
    fn non_zero(&self, i: usize) -> bool {
        !self.table[i * self.num_windows].is_zero()
    }

    /// Radix `2^c`.
    pub fn window(&self) -> usize {
        self.c
    }

    /// Multiples stored per base, `W = ceil(bits / c)`.
    pub fn num_windows(&self) -> usize {
        self.num_windows
    }

    /// Shared bucket-array size.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Precomputed point count, `n W`.
    pub fn table_len(&self) -> usize {
        self.table.len()
    }

    /// Table size in bytes.
    pub fn table_bytes(&self) -> usize {
        // Ignoring a few `usize`s in the struct
        self.table.len() * core::mem::size_of::<Affine<P>>()
    }
}

/// `num_windows` multiples of `base` as `[base, 2^c base, 2^{2c} base, ..]`.
fn base_multiples<P: SWCurveConfig>(base: &Affine<P>, num_windows: usize, c: usize) -> Vec<Projective<P>> {
    let mut cur = base.into_group();
    let mut row = Vec::with_capacity(num_windows);
    for _ in 0..num_windows {
        row.push(cur);
        for _ in 0..c {
            cur.double_in_place();
        }
    }
    row
}

/// Window minimizing the modeled addition count `n ceil(bits/c) + 2^{c-1}`.
fn best_window<P: SWCurveConfig>(n: usize) -> usize {
    let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
    (2..=20)
        .min_by_key(|&c| n * num_bits.div_ceil(c) + (1usize << (c - 1)))
        .unwrap()
}

/// Window for the parallel evaluation. This divides the accumulation `n W` across threads,
/// but the shared bucket array's running sum and combine (size `2^{c-1}`) and its memory traffic
/// parallelize worse, so [`best_window`]'s bucket term is up-weighted by the thread count. The pick
/// is a smaller `c` than [`best_window`] (more, smaller windows), the more so at fewer bases, and
/// reduces to [`best_window`] at one thread. Never exceeds [`best_window`].
#[cfg(feature = "parallel")]
fn best_window_parallel<P: SWCurveConfig>(n: usize, threads: usize) -> usize {
    let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
    let weight = 1.0 + (threads.max(1) - 1) as f64 * PAR_BUCKET_SLOPE;
    let cost = |c: usize| n as f64 * num_bits.div_ceil(c) as f64 + weight * (1u64 << (c - 1)) as f64;
    let serial = best_window::<P>(n);
    let model = (2..=serial)
        .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
        .unwrap_or(serial);
    model.max(serial.min(PAR_MIN_WINDOW))
}

/// Bucket-array size for `num_windows` radix-`2^c` signed-digit windows. Every window but
/// the most significant produces digits in `[-2^{c-1}, 2^{c-1}]`; the most significant is not
/// recentered, so its digit reaches the top `bits - c(W-1)` bits, bounded by the modulus there.
fn bucket_count<P: SWCurveConfig>(c: usize, num_windows: usize, num_bits: usize) -> usize {
    let shift = c * (num_windows - 1);
    let wlast = num_bits - shift;
    let mut final_digit_bound = 1usize << wlast;
    let mod_shifted = P::ScalarField::MODULUS >> shift as u32;
    let mod_limbs = mod_shifted.as_ref();
    if mod_limbs.iter().skip(1).all(|&l| l == 0) && mod_limbs[0] < (1u64 << c) {
        final_digit_bound = final_digit_bound.min(mod_limbs[0] as usize + 1);
    }
    (1usize << (c - 1)).max(final_digit_bound)
}
