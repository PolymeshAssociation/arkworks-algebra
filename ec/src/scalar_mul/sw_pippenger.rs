//! Batch-affine bucket accumulation for short Weierstrass curves: per window, counting-sort the
//! signed points by bucket, then reduce every bucket by a pairwise tree whose levels each share
//! one field inversion.
//!
//! The generic path ([`crate::scalar_mul::variable_base`]) keeps each bucket in extended
//! Jacobian (`xyzz`) coordinates and adds an affine base into it with a mixed addition, 8M + 2S.
//! Affine addition needs the slope's inverse, but a whole level of the tree is independent, so
//! Montgomery's trick amortizes one inversion over all of it and the addition costs 5M + 1S.
//! Affine points are also half the size of `xyzz` buckets, so the working set fits the cache
//! better.
//!
//! Neither collision deferral nor point chunking appears: sorting removes the first and the tree
//! removes the second, which is what let the same shape run in parallel where an earlier attempt
//! here did not.
//!
//! # Sources
//!
//! Structure ported from Zakura `glv.rs`, at
//! <https://github.com/zakura-core/common/blob/98846ee/crates/pasta_curves/src/glv.rs>:
//!
//! Their blog: <https://zakura.com/engineering/prepared-multiscalar-zero-checks/>.

use crate::{
    scalar_mul::variable_base::{
        combine_window_sums, pippenger_setup, route_msm, PippengerSetup,
    },
    short_weierstrass::{Affine, Bucket, Projective, SWCurveConfig},
    AdditiveGroup, AffineRepr,
};
use ark_ff::{serial_batch_inversion_and_mul, Field, One, PrimeField, Zero};
use ark_std::{cfg_into_iter, vec, vec::Vec};
use itertools::Either;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Smallest multi scalar multiplication routed here; below it the counting sort and the tree's
/// bookkeeping outweigh the cheaper additions.
pub const BATCH_AFFINE_MIN_POINTS: usize = 1 << 10;

#[derive(Clone, Copy, Default)]
pub(crate) struct Point<F: Zero> {
    pub(crate) x: F,
    pub(crate) y: F,
}

/// One level's pending affine additions, as parallel arrays.
#[derive(Default)]
struct Pending<F: Zero> {
    /// Slot in the output level holding the left operand, which the result replaces.
    out: Vec<usize>,
    /// Sum of x-coords of pairwise point addition
    x_sum: Vec<F>,
    /// Numerators of pairwise point addition
    num: Vec<F>,
    /// Denominators of pairwise point addition
    den: Vec<F>,
}

impl<F: Field> Pending<F> {
    fn clear(&mut self) {
        self.out.clear();
        self.x_sum.clear();
        self.num.clear();
        self.den.clear();
    }

    fn reserve(&mut self, n: usize) {
        self.out.reserve(n);
        self.x_sum.reserve(n);
        self.num.reserve(n);
        self.den.reserve(n);
    }

    fn push(&mut self, out: usize, x_sum: F, num: F, den: F) {
        self.out.push(out);
        self.x_sum.push(x_sum);
        self.num.push(num);
        self.den.push(den);
    }
}

/// Two level buffers and the pending records, kept so reducing many bucket ranges allocates once.
#[derive(Default)]
pub(crate) struct ReduceScratch<F: Zero> {
    /// Store the points for the current level
    a: Vec<Point<F>>,
    a_off: Vec<usize>,
    /// Store the points for the next level
    b: Vec<Point<F>>,
    b_off: Vec<usize>,
    pending: Pending<F>,
}

/// `sum(bases_i * scalars_i)` with affine buckets. Mirrors
/// [`msm_unchecked`](crate::VariableBaseMSM::msm_unchecked): routes through [`route_msm`] (host
/// MSM on guest builds, then the GLV ladder), else convert to bigints and call
/// [`msm_batch_affine_bigint`].
pub fn msm_batch_affine<P: SWCurveConfig>(
    bases: &[Affine<P>],
    scalars: &[P::ScalarField],
) -> Projective<P> {
    match route_msm::<Projective<P>>(bases, scalars) {
        Either::Left(res) => res,
        Either::Right(bigints) => msm_batch_affine_bigint::<P>(bases, &bigints),
    }
}

/// `sum(bases_i * bigints_i)` with affine buckets.
pub fn msm_batch_affine_bigint<P: SWCurveConfig>(
    bases: &[Affine<P>],
    bigints: &[<P::ScalarField as PrimeField>::BigInt],
) -> Projective<P> {
    let size = bases.len().min(bigints.len());
    if size == 0 {
        return Projective::zero();
    }
    let bases = &bases[..size];

    let PippengerSetup {
        c,
        digits_count,
        scalar_digits,
        ms_window_num_buckets,
        num_buckets,
    } = pippenger_setup::<P::ScalarField>(bigints, size);

    let window_sums: Vec<_> = cfg_into_iter!(0..digits_count)
        .map(|i| {
            let n = if i == (digits_count - 1) {
                ms_window_num_buckets
            } else {
                num_buckets
            };
            window_sum::<P>(&scalar_digits, digits_count, i, bases, n)
        })
        .collect();

    combine_window_sums::<Projective<P>>(&window_sums, c)
}

/// One window's contribution: counting-sort the signed points into buckets, reduce each bucket,
/// then the usual high-to-low running sum.
fn window_sum<P: SWCurveConfig>(
    scalar_digits: &[i64],
    digits_count: usize,
    window_index: usize,
    bases: &[Affine<P>],
    num_buckets: usize,
) -> Bucket<P> {
    // `scalar_digits` is scalar-major, so base `i`'s digit for this window is
    // `scalar_digits[i * digits_count + window_index]`. An identity base is mapped to digit 0 so it
    // is ignored in the bucket.
    let (mut offsets, mut points) = counting_sort::<P>(num_buckets, bases.len(), |i| {
        let base = bases[i];
        let d = if base.is_zero() {
            0
        } else {
            scalar_digits[i * digits_count + window_index]
        };
        (d, base)
    });

    reduce_tree::<P>(&mut points, &mut offsets, true);
    running_bucket_sum::<P>(&points, &offsets)
}

/// Same idea as [crate::scalar_mul::variable_base::reduce_buckets]. bucket `b` (magnitude `b + 1`)
/// is non-empty when `offsets[b + 1] > offsets[b]`. The bucket count is `offsets.len() - 1`.
pub(crate) fn running_bucket_sum<P: SWCurveConfig>(
    points: &[Point<P::BaseField>],
    offsets: &[usize],
) -> Bucket<P> {
    let mut running = Bucket::<P>::ZERO;
    let mut sum = Bucket::<P>::ZERO;
    for b in (0..offsets.len() - 1).rev() {
        if offsets[b + 1] > offsets[b] {
            let p = points[offsets[b]];
            running += Affine::<P>::new_unchecked(p.x, p.y);
        }
        sum += &running;
    }
    sum
}

/// Reduces every bucket of `points`, delimited by `offsets`, to at most one point, by pairing
/// neighbors level by level. Each level is one batch inversion.
///
/// The first pass assumes distinct x-coordinates, which is what makes the denominator a plain
/// `r.x - l.x`; a zero denominator means some pair collided, and the level is redone with
/// doublings and inverse pairs handled. The retry cannot itself hit a zero denominator over a
/// prime field of characteristic > 2, so this never fails for the curves it serves.
pub(crate) fn reduce_tree<P: SWCurveConfig>(
    points: &mut Vec<Point<P::BaseField>>,
    offsets: &mut Vec<usize>,
    parallelize: bool,
) {
    let mut scratch = ReduceScratch::default();
    match reduce_levels::<P>(points, offsets, parallelize, &mut scratch) {
        0 => {}
        1 => {
            ark_std::mem::swap(points, &mut scratch.a);
            ark_std::mem::swap(offsets, &mut scratch.a_off);
        }
        _ => {
            ark_std::mem::swap(points, &mut scratch.b);
            ark_std::mem::swap(offsets, &mut scratch.b_off);
        }
    }
}

/// The level loop. The first level reads the caller's slices and later levels alternate between
/// the two scratch buffers, so the level a zero denominator declines is still intact for the
/// complete-formula retry. Returns which holds the reduced level: `0` means the input,
/// `1` means `a`, `2` means `b`. Panics if the retry also hits a zero denominator, which needs a
/// characteristic-2 field and so cannot happen for the prime-field curves this path serves.
fn reduce_levels<P: SWCurveConfig>(
    points: &[Point<P::BaseField>],
    offsets: &[usize],
    parallelize: bool,
    scratch: &mut ReduceScratch<P::BaseField>,
) -> u8 {
    // Tracks current source and destination buffer
    let mut cur = 0u8;
    // On retrying the level, same x-coord case is handled. Assumption is that retries should not
    // be needed in most cases but when they are, an extra iteration is done
    let mut current_level_retried = false;
    loop {
        let ReduceScratch {
            a,
            a_off,
            b,
            b_off,
            pending,
        } = &mut *scratch;
        let (src, src_off, dest, dest_off): (&[Point<_>], &[usize], &mut Vec<Point<_>>, &mut Vec<usize>) =
            match cur {
                0 => (points, offsets, a, a_off),   // only during the first iteration
                1 => (&a[..], &a_off[..], b, b_off),
                _ => (&b[..], &b_off[..], a, a_off),
            };

        // Done if no bucket has a size greater than 1
        if !src_off.windows(2).any(|r| r[1] - r[0] > 1) {
            break;
        }

        dest.clear();
        dest.reserve(src.len().div_ceil(2) + src_off.len());
        dest_off.clear();
        dest_off.reserve(src_off.len());
        dest_off.push(0);
        pending.clear();
        pending.reserve(src.len().div_ceil(2));

        for range in src_off.windows(2) {
            // Take each bucket and take pairs of points from each bucket
            let bucket = &src[range[0]..range[1]];
            for pair in bucket.chunks_exact(2) {
                let (l, r) = (pair[0], pair[1]);
                let (num, den) = if current_level_retried && l.x == r.x {
                    if l.y != r.y || l.y.is_zero() {
                        // If inverses or 2-torsion, then sum is the 0, so ignore both.
                        continue;
                    }
                    let l_x_sqr = l.x.square();
                    // (3*l_x^2 + a, 2*y)
                    (l_x_sqr.double() + l_x_sqr + P::COEFF_A, l.y.double())
                } else {
                    (r.y - l.y, r.x - l.x)
                };
                let slot = dest.len();
                dest.push(l);
                pending.push(slot, l.x + r.x, num, den);
            }
            // If bucket has odd number of point, process the last point in next iteration
            if bucket.len() % 2 == 1 {
                dest.push(bucket[bucket.len() - 1]);
            }
            dest_off.push(dest.len());
        }

        // Add the current level pending points
        if batch_add::<P>(pending, dest, parallelize).is_none() {
            debug_assert!(!current_level_retried);
            // The source level is untouched, so it simply runs again.
            current_level_retried = true;
            continue;
        }
        cur = if cur == 1 { 2 } else { 1 };
    }
    cur
}

/// Records per parallel batch, so a level of `k` records becomes at most `k / PAR_CHUNK` tasks,
/// each paying its own inversion. Only the window fan-out's leftovers are there to fill, so the
/// split is neutral while the windows already occupy every thread and helps only once the levels
/// grow large enough that the fan-out's tail dominates.
#[cfg(feature = "parallel")]
const PAR_CHUNK: usize = 512;

/// Computes final addition points whose inverted denominators are in `inv`. `out` is the
/// sub-slice of the level starting at slot `start`.
fn finalize_points<F: Field>(
    slots: &[usize],
    x_sum: &[F],
    num: &[F],
    inv: &[F],
    out: &mut [Point<F>],
    start: usize,
) {
    for i in 0..slots.len() {
        let k = slots[i] - start;
        let left = out[k];
        let lambda = num[i] * inv[i];
        let x = lambda.square() - x_sum[i];
        let y = lambda * (left.x - x) - left.y;
        out[k] = Point { x, y };
    }
}

/// Affine `base` as a [`Point`], its `y` negated when `negate`.
pub(crate) fn signed_point<P: SWCurveConfig>(base: &Affine<P>, negate: bool) -> Point<P::BaseField> {
    Point {
        x: base.x,
        y: if negate { -base.y } else { base.y },
    }
}

/// Counting-sort `len` signed entries into `num_buckets` buckets by digit magnitude. `entry(i)`
/// gives entry `i`'s signed digit and its affine point. Returns the bucket prefix sums,
/// (`offsets[b]..offsets[b + 1]` is bucket `b`, holding magnitude `b + 1`) and the points laid out
/// bucket-major.
pub(crate) fn counting_sort<P: SWCurveConfig>(
    num_buckets: usize,
    len: usize,
    entry: impl Fn(usize) -> (i64, Affine<P>),
) -> (Vec<usize>, Vec<Point<P::BaseField>>) {
    // Bucket sizes, then their prefix sums.
    let mut offsets = vec![0usize; num_buckets + 1];
    for i in 0..len {
        let d = entry(i).0;
        if d != 0 {
            offsets[d.unsigned_abs() as usize] += 1;
        }
    }
    for b in 0..num_buckets {
        offsets[b + 1] += offsets[b];
    }

    // Place every nonzero entry into its bucket, contiguous.
    let mut positions = offsets[..num_buckets].to_vec();
    let mut points = vec![Point::default(); offsets[num_buckets]];
    for i in 0..len {
        let (d, base) = entry(i);
        if d == 0 {
            continue;
        }
        let b = d.unsigned_abs() as usize - 1;
        points[positions[b]] = signed_point(&base, d < 0);
        positions[b] += 1;
    }
    (offsets, points)
}

/// Finishes every pending addition, one batch inversion for the whole level. `None`, with
/// nothing written, when any denominator is zero, which tells the caller to redo the level with
/// the exceptional cases handled.
fn batch_add<P: SWCurveConfig>(
    pending: &mut Pending<P::BaseField>,
    out: &mut [Point<P::BaseField>],
    parallelize: bool,
) -> Option<()> {
    // Since some pair of x-coords is same, special case handling needed which this can't do
    if pending.den.iter().any(Zero::is_zero) {
        return None;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallelize;

    #[cfg(feature = "parallel")]
    if parallelize && pending.out.len() >= 2 * PAR_CHUNK {
        par_batch_add::<P>(pending, out);
        return Some(());
    }
    // The batch inversion runs in place over the denominators.
    serial_batch_inversion_and_mul(&mut pending.den, &P::BaseField::one());
    finalize_points(
        &pending.out,
        &pending.x_sum,
        &pending.num,
        &pending.den,
        out,
        0,
    );
    Some(())
}

/// [`batch_add`] split across threads, one inversion per chunk. The output slots are strictly
/// increasing along `pending`, so a chunk's record stores a contiguous slot range disjoint from
/// every other chunk's, and the left operand each record reads is the slot it writes.
#[cfg(feature = "parallel")]
fn par_batch_add<P: SWCurveConfig>(
    pending: &mut Pending<P::BaseField>,
    out: &mut [Point<P::BaseField>],
) {
    let n = pending.out.len();
    // At least `PAR_CHUNK` records per chunk, and no more chunks than threads, which bounds the
    // inversions the split adds.
    let chunk_size = PAR_CHUNK.max(n.div_ceil(rayon::current_num_threads()));
    let Pending {
        out: slots,
        x_sum,
        num,
        den,
    } = pending;

    let mut chunks = Vec::with_capacity(n.div_ceil(chunk_size));
    let mut rest = &mut out[..];
    let mut offset = 0;
    let mut lo = 0;
    while lo < n {
        let hi = (lo + chunk_size).min(n);
        let (first, last) = (slots[lo], slots[hi - 1]);
        // Slots between chunks belong to odd-tail points nothing writes.
        let (_, tail) = rest.split_at_mut(first - offset);
        let (mine, tail) = tail.split_at_mut(last - first + 1);
        chunks.push((mine, first));
        rest = tail;
        offset = last + 1;
        lo = hi;
    }

    rayon::scope(|s| {
        for ((((slots, x_sum), num), den), (chunk, start)) in slots
            .chunks(chunk_size)
            .zip(x_sum.chunks(chunk_size))
            .zip(num.chunks(chunk_size))
            .zip(den.chunks_mut(chunk_size))
            .zip(chunks)
        {
            s.spawn(move |_| {
                serial_batch_inversion_and_mul(den, &P::BaseField::one());
                finalize_points(slots, x_sum, num, den, chunk, start);
            });
        }
    });
}

/// [`reduce_tree`] over borrowed input, staging every level in `scratch`. Returns the reduced
/// level, at most one point per bucket, which lives in `scratch` (or is the input when no bucket
/// held two points).
#[cfg(feature = "parallel")]
pub(crate) fn reduce_tree_with<'a, P: SWCurveConfig>(
    points: &'a [Point<P::BaseField>],
    offsets: &'a [usize],
    parallelize: bool,
    scratch: &'a mut ReduceScratch<P::BaseField>,
) -> (&'a [Point<P::BaseField>], &'a [usize]) {
    match reduce_levels::<P>(points, offsets, parallelize, scratch) {
        0 => (points, offsets),
        1 => (&scratch.a[..], &scratch.a_off[..]),
        _ => (&scratch.b[..], &scratch.b_off[..]),
    }
}


