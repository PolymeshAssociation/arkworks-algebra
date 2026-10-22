//! x86-64 inline-assembly Montgomery kernels for the Pasta modulus shape, `modulus[2] = 0` and
//! `modulus[3] = 2^62`. `mul` and `square` use MULX (BMI2) and ADCX/ADOX (ADX), so the caller
//! enters them only when the CPU has both. Per round `mul` issues 3 MULX and one `imul` for the
//! Montgomery factor, 28 multiplies against the generic `ff-asm` kernel's 36, and `square` issues
//! 10 MULX for the 512-bit square instead of 16. The operands are read through pointers and
//! `modulus` and `INV` share one `[u64; 5]` block, since twelve limbs plus staging registers do not
//! fit x86-64's allocatable registers. `mul` and `square` stay `#[inline(never)]`: they use nearly
//! every register, and inlining them into a register-heavy caller can make allocation fail. The
//! loose kernels and `sum_of_two_products` are `#[inline(always)]`, as in Zakura Udon.
//!
//! The round structure and the canonicity contract are those of the AArch64 kernels in
//! `aarch64.rs`: every input of `add`, `sub`, `mul` and `square` must be canonical, which the
//! wrappers debug-assert, and the `_loose` kernels take and return values below `2p`. The blocks
//! are straight-line with CMOV selects.
//!
//! Ported from Zakura `pasta_curves` [`x86_64_asm.rs`][x86_64_asm.rs]: [`add`][add], [`sub`][sub],
//! [`mul`][mul], [`square`][square].
//!
//! [x86_64_asm.rs]: https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/fields/x86_64_asm.rs
//! [add]: https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/fields/x86_64_asm.rs#L86-L132
//! [sub]: https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/fields/x86_64_asm.rs#L140-L184
//! [mul]: https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/fields/x86_64_asm.rs#L195-L415
//! [square]: https://github.com/zakura-core/common/blob/61aeb7e066f3e9f0afa13355d0f73f02170bcd50/crates/pasta_curves/src/fields/x86_64_asm.rs#L433-L619

use core::arch::asm;

type Limbs = [u64; 4];

const PASTA_HIGH_LIMB: u64 = 1 << 62;

/// `lhs + rhs mod p`. `params` is `[modulus[0..4], INV]`.
#[inline(always)]
pub(super) fn add(lhs: &Limbs, rhs: &Limbs, params: &[u64; 5]) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line arithmetic reading only the words behind the two
    // passed references (`readonly`); no stack use, and outputs depend only
    // on the declared inputs. `params` starts with the four modulus limbs.
    // All memory addresses are input-independent.
    unsafe {
        asm!(
            "add {r0}, qword ptr [{b}]",
            "adc {r1}, qword ptr [{b} + 8]",
            "adc {r2}, qword ptr [{b} + 16]",
            "adc {r3}, qword ptr [{b} + 24]",
            "mov {t0}, {r0}",
            "mov {t1}, {r1}",
            "mov {t2}, {r2}",
            "mov {t3}, {r3}",
            "sub {t0}, qword ptr [{p}]",
            "sbb {t1}, qword ptr [{p} + 8]",
            "sbb {t2}, 0",
            "sbb {t3}, qword ptr [{p} + 24]",
            "cmovnc {r0}, {t0}",
            "cmovnc {r1}, {t1}",
            "cmovnc {r2}, {t2}",
            "cmovnc {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b = in(reg) rhs,
            p = in(reg) params,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, readonly, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// `lhs - rhs mod p`.
#[inline(always)]
pub(super) fn sub(lhs: &Limbs, rhs: &Limbs, params: &[u64; 5]) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line arithmetic reading only the words behind the two
    // passed references (`readonly`); no stack use, and outputs depend only
    // on the declared inputs. The conditional loads use fixed,
    // input-independent addresses.
    unsafe {
        asm!(
            "sub {r0}, qword ptr [{b}]",
            "sbb {r1}, qword ptr [{b} + 8]",
            "sbb {r2}, qword ptr [{b} + 16]",
            "sbb {r3}, qword ptr [{b} + 24]",
            // MOV and CMOV preserve the borrow flag from the subtraction.
            "mov {m0}, 0",
            "mov {m1}, 0",
            "mov {m3}, 0",
            "cmovc {m0}, qword ptr [{p}]",
            "cmovc {m1}, qword ptr [{p} + 8]",
            "cmovc {m3}, qword ptr [{p} + 24]",
            "add {r0}, {m0}",
            "adc {r1}, {m1}",
            "adc {r2}, 0",
            "adc {r3}, {m3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b = in(reg) rhs,
            p = in(reg) params,
            m0 = out(reg) _,
            m1 = out(reg) _,
            m3 = out(reg) _,
            options(pure, readonly, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// The body of [`mul`] and [`mul_loose`]. The bracketed template literals are appended to the
/// instruction sequence: the closing conditional subtraction for [`mul`], nothing for
/// [`mul_loose`]. Tokens after the closing bracket are extra operands for those literals.
macro_rules! mul_kernel {
    ($lhs:expr, $rhs:expr, $params:expr, [$($tail:literal),* $(,)?] $($extra:tt)*) => {{
        let (lhs, rhs, params): (&Limbs, &Limbs, &[u64; 5]) = ($lhs, $rhs, $params);
        let (o0, o1, o2, o3): (u64, u64, u64, u64);
        // SAFETY: straight-line arithmetic reading only the thirteen words
        // behind the three passed references (`readonly`); no stack use, and
        // outputs depend only on the declared inputs. `params` contains the four
        // modulus limbs followed by `inv`.
        //
        // Register roles: the five-limb accumulator lives in {ae}/{be}/{ce}/
        // {de}/{ee} and its window rotates down one register per round (the
        // register cancelled by the round's Montgomery step becomes the next
        // round's fifth limb), so after four rounds the candidate sits in
        // {ee},{ae},{be},{ce}. {s1}/{s2}/{s3} stage multiplier halves and
        // shifted `q * modulus[3]` terms so no flag-writing instruction lands
        // inside a carry chain. RDX is the implicit MULX source: each round's
        // `rhs` limb, then the round's Montgomery factor `q`.
        unsafe {
            asm!(
                // Round 0: initialize the accumulator with lhs * rhs[0].
                "mov rdx, qword ptr [{b}]",          // rdx = b[0].
                // ae = low(a[0]*b[0]); be = high.
                "mulx {be}, {ae}, qword ptr [{a}]",
                "mulx {ce}, {s1}, qword ptr [{a} + 8]",
                // Fold low(a[1]*b[0]) into limb 1.
                "add {be}, {s1}",
                "mulx {de}, {s1}, qword ptr [{a} + 16]",
                "adc {ce}, {s1}",                    // Fold low(a[2]*b[0]) and carry.
                "mulx {ee}, {s1}, qword ptr [{a} + 24]",
                "adc {de}, {s1}",                    // Fold low(a[3]*b[0]) and carry.
                "adc {ee}, 0",                       // Fifth limb of lhs * b[0].

                // Montgomery step 0: q = limb0 * inv; add q*p; shift one limb.
                "mov rdx, {ae}",
                "imul rdx, qword ptr [{p} + 32]",    // rdx = q (low 64 bits only).
                // s1 = low(q*p[1]); s2 = high (kept).
                "mulx {s2}, {s1}, qword ptr [{p} + 8]",
                "mov {s3}, rdx",
                // s3 = low(q*p[3]); p[2] contributes nothing.
                "shl {s3}, 62",
                // low(q*p[0]) cancels limb 0; its carry is one exactly when the
                // limb is nonzero, which NEG leaves in CF.
                "neg {ae}",                          // CF = (limb0 != 0); ae is dead.
                // Add low(q*p[1]) and the cancellation carry.
                "adc {be}, {s1}",
                "adc {ce}, 0",                       // Propagate across zero p[2].
                "adc {de}, {s3}",                    // Add low(q*p[3]) and carry.
                "adc {ee}, 0",                       // Propagate into the fifth limb.
                // s1 = high(q*p[0]); the low half is spent.
                "mulx {s1}, {s3}, qword ptr [{p}]",
                "mov {s3}, rdx",
                "shr {s3}, 2",                       // s3 = high(q*p[3]).
                // Next round's fifth limb (MOV keeps flags).
                "mov {ae}, 0",
                // New limbs include the high halves of q*p.
                "add {be}, {s1}",
                "adc {ce}, {s2}",
                "adc {de}, 0",                       // p[2] contributes zero.
                "adc {ee}, {s3}",
                // Capture the reduction carry as limb 4.
                "adc {ae}, 0",

                // Round 1: accumulator window is [be,ce,de,ee,ae]; add lhs*b[1]
                // on dual carry chains (CF: low halves, OF: high halves).
                "mov rdx, qword ptr [{b} + 8]",      // rdx = b[1].
                "xor {s1}, {s1}",                    // Clear CF and OF.
                "mulx {s2}, {s1}, qword ptr [{a}]",
                "adcx {be}, {s1}",
                "adox {ce}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 8]",
                "adcx {ce}, {s1}",
                "adox {de}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 16]",
                "adcx {de}, {s1}",
                "adox {ee}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 24]",
                "adcx {ee}, {s1}",
                "adox {ae}, {s2}",
                "mov {s1}, 0",
                // Close both carry chains into limb 4.
                "adcx {ae}, {s1}",
                "adox {ae}, {s1}",

                // Montgomery step 1.
                "mov rdx, {be}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {s2}, {s1}, qword ptr [{p} + 8]",
                "mov {s3}, rdx",
                "shl {s3}, 62",
                "neg {be}",
                "adc {ce}, {s1}",
                "adc {de}, 0",
                "adc {ee}, {s3}",
                "adc {ae}, 0",
                "mulx {s1}, {s3}, qword ptr [{p}]",
                "mov {s3}, rdx",
                "shr {s3}, 2",
                "mov {be}, 0",
                "add {ce}, {s1}",
                "adc {de}, {s2}",
                "adc {ee}, 0",
                "adc {ae}, {s3}",
                "adc {be}, 0",

                // Round 2: window [ce,de,ee,ae,be]; add lhs*b[2].
                "mov rdx, qword ptr [{b} + 16]",
                "xor {s1}, {s1}",
                "mulx {s2}, {s1}, qword ptr [{a}]",
                "adcx {ce}, {s1}",
                "adox {de}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 8]",
                "adcx {de}, {s1}",
                "adox {ee}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 16]",
                "adcx {ee}, {s1}",
                "adox {ae}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 24]",
                "adcx {ae}, {s1}",
                "adox {be}, {s2}",
                "mov {s1}, 0",
                "adcx {be}, {s1}",
                "adox {be}, {s1}",

                // Montgomery step 2.
                "mov rdx, {ce}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {s2}, {s1}, qword ptr [{p} + 8]",
                "mov {s3}, rdx",
                "shl {s3}, 62",
                "neg {ce}",
                "adc {de}, {s1}",
                "adc {ee}, 0",
                "adc {ae}, {s3}",
                "adc {be}, 0",
                "mulx {s1}, {s3}, qword ptr [{p}]",
                "mov {s3}, rdx",
                "shr {s3}, 2",
                "mov {ce}, 0",
                "add {de}, {s1}",
                "adc {ee}, {s2}",
                "adc {ae}, 0",
                "adc {be}, {s3}",
                "adc {ce}, 0",

                // Round 3: window [de,ee,ae,be,ce]; add lhs*b[3].
                "mov rdx, qword ptr [{b} + 24]",
                "xor {s1}, {s1}",
                "mulx {s2}, {s1}, qword ptr [{a}]",
                "adcx {de}, {s1}",
                "adox {ee}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 8]",
                "adcx {ee}, {s1}",
                "adox {ae}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 16]",
                "adcx {ae}, {s1}",
                "adox {be}, {s2}",
                "mulx {s2}, {s1}, qword ptr [{a} + 24]",
                "adcx {be}, {s1}",
                "adox {ce}, {s2}",
                "mov {s1}, 0",
                "adcx {ce}, {s1}",
                "adox {ce}, {s1}",

                // Montgomery step 3. Canonical rhs bounds the candidate below
                // 2p < R, so the final shift produces no fifth limb (see the
                // AArch64 module docs); the shift's carry adc is omitted.
                // Loose inputs keep it below 2p by the closure argument cited
                // at `mul_loose`.
                "mov rdx, {de}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {s2}, {s1}, qword ptr [{p} + 8]",
                "mov {s3}, rdx",
                "shl {s3}, 62",
                "neg {de}",
                "adc {ee}, {s1}",
                "adc {ae}, 0",
                "adc {be}, {s3}",
                "adc {ce}, 0",
                "mulx {s1}, {s3}, qword ptr [{p}]",
                "mov {s3}, rdx",
                "shr {s3}, 2",
                "add {ee}, {s1}",                    // Final candidate limb 0.
                "adc {ae}, {s2}",                    // Final candidate limb 1.
                "adc {be}, 0",                       // Final candidate limb 2.
                "adc {ce}, {s3}",                    // Final candidate limb 3.
                $($tail,)*
                a = in(reg) lhs.as_ptr(),
                b = in(reg) rhs.as_ptr(),
                p = in(reg) params.as_ptr(),
                ae = out(reg) o1,
                be = out(reg) o2,
                ce = out(reg) o3,
                de = out(reg) _,
                ee = out(reg) o0,
                s1 = out(reg) _,
                s2 = out(reg) _,
                s3 = out(reg) _,
                out("rdx") _,
                $($extra)*
                options(pure, readonly, nostack),
            );
        }
        [o0, o1, o2, o3]
    }};
}

/// `lhs * rhs * R^{-1} mod p`. Requires BMI2 and ADX.
#[inline(never)]
pub(super) fn mul(lhs: &Limbs, rhs: &Limbs, params: &[u64; 5]) -> Limbs {
    mul_kernel!(
        lhs,
        rhs,
        params,
        [
            // Conditional subtraction of p = [p0, p1, 0, 2^62].
            "movabs rdx, {p3}",                  // Materialize p[3] = 2^62.
            "mov {s1}, {ee}",
            "mov {s2}, {ae}",
            "mov {s3}, {be}",
            "mov {de}, {ce}",
            // Tentative limb 0 = candidate - p[0].
            "sub {s1}, qword ptr [{p}]",
            "sbb {s2}, qword ptr [{p} + 8]",     // Tentative limb 1 minus p[1].
            "sbb {s3}, 0",                       // Limb 2; p[2] is zero.
            "sbb {de}, rdx",                     // Tentative limb 3 minus p[3].
            // No borrow (CF clear) means the candidate is at least p, so the
            // subtracted value is the canonical output.
            "cmovnc {ee}, {s1}",
            "cmovnc {ae}, {s2}",
            "cmovnc {be}, {s3}",
            "cmovnc {ce}, {de}",
        ]
        p3 = const PASTA_HIGH_LIMB,
    )
}

/// The body of [`square`] and [`square_loose`], with the closing template literals and extra
/// operands as for [`mul_kernel`].
macro_rules! square_kernel {
    ($value:expr, $params:expr, [$($tail:literal),* $(,)?] $($extra:tt)*) => {{
        let (value, params): (&Limbs, &[u64; 5]) = ($value, $params);
        let (o0, o1, o2, o3): (u64, u64, u64, u64);
        // SAFETY: straight-line arithmetic reading only the words behind the two
        // passed references (`readonly`); no stack use, and outputs depend only
        // on the declared inputs. `params` contains the four modulus limbs
        // followed by `inv`. The input pointer's register is reclaimed as the
        // reduction's carry limb once phase 1 has consumed the last load.
        unsafe {
            asm!(
                // Phase 1: the 512-bit square in z0..z7.
                // Cross products a[i]*a[j] (i < j), accumulated as they stream.
                "xor {z5:e}, {z5:e}",
                "xor {z6:e}, {z6:e}",
                "xor {z7:e}, {z7:e}",
                "mov rdx, qword ptr [{a}]",
                "mulx {t1}, {z1}, qword ptr [{a} + 8]",  // a0*a1.
                "mulx {t2}, {z2}, qword ptr [{a} + 16]", // a0*a2.
                "mulx {z4}, {z3}, qword ptr [{a} + 24]", // a0*a3.
                "add {z2}, {t1}",                        // Fold high(a0*a1).
                // Fold high(a0*a2) and carry.
                "adc {z3}, {t2}",
                "adc {z4}, 0",
                "mov rdx, qword ptr [{a} + 8]",
                "mulx {t2}, {t1}, qword ptr [{a} + 16]", // a1*a2.
                "add {z3}, {t1}",
                "adc {z4}, {t2}",
                "adc {z5}, 0",
                "mulx {t2}, {t1}, qword ptr [{a} + 24]", // a1*a3.
                "add {z4}, {t1}",
                "adc {z5}, {t2}",
                "adc {z6}, 0",
                "mov rdx, qword ptr [{a} + 16]",
                "mulx {t2}, {t1}, qword ptr [{a} + 24]", // a2*a3.
                "add {z5}, {t1}",
                "adc {z6}, {t2}",
                "adc {z7}, 0",
                // Double the cross products. The doubled sum is below 2^512, so
                // no carry leaves z7.
                "add {z1}, {z1}",
                "adc {z2}, {z2}",
                "adc {z3}, {z3}",
                "adc {z4}, {z4}",
                "adc {z5}, {z5}",
                "adc {z6}, {z6}",
                "adc {z7}, {z7}",
                // Add the diagonal squares in one carry chain (MOV and MULX
                // preserve flags).
                "mov rdx, qword ptr [{a}]",
                "mulx {t2}, {z0}, rdx",                  // z0 = low(a0^2).
                "add {z1}, {t2}",                        // High(a0^2).
                "mov rdx, qword ptr [{a} + 8]",
                "mulx {t2}, {t1}, rdx",
                "adc {z2}, {t1}",
                "adc {z3}, {t2}",
                "mov rdx, qword ptr [{a} + 16]",
                "mulx {t2}, {t1}, rdx",
                "adc {z4}, {t1}",
                "adc {z5}, {t2}",
                "mov rdx, qword ptr [{a} + 24]",
                "mulx {t2}, {t1}, rdx",
                "adc {z6}, {t1}",
                "adc {z7}, {t2}",                        // a^2 < 2^510: no carry out.

                // Phase 2: four Montgomery cancellations on the low half, the
                // same two-sweep step as [`mul`]'s. The window rotates down one
                // register per step; {a} (its loads are done) serves as the
                // first carried fifth limb.
                // Step 0: window [z0, z1, z2, z3], carry into {a}.
                "mov rdx, {z0}",
                "imul rdx, qword ptr [{p} + 32]",        // rdx = q.
                "mulx {t2}, {t1}, qword ptr [{p} + 8]",  // t1/t2 = low/high(q*p1).
                "mov {a}, rdx",
                "shl {a}, 62",                           // low(q*p3); p2 is zero.
                "neg {z0}",                              // CF = (limb0 != 0).
                "adc {z1}, {t1}",
                "adc {z2}, 0",
                "adc {z3}, {a}",
                "mov {a}, 0",
                "adc {a}, 0",                            // Carry above limb 3.
                // t1 = high(q*p0); low is spent.
                "mulx {t1}, {z0}, qword ptr [{p}]",
                "mov {z0}, rdx",
                "shr {z0}, 2",                           // high(q*p3).
                "add {z1}, {t1}",                        // New limb 0.
                "adc {z2}, {t2}",                        // New limb 1 += high(q*p1).
                "adc {z3}, 0",                           // New limb 2.
                "adc {a}, {z0}",                         // New limb 3 += high(q*p3).
                // Step 1: window [z1, z2, z3, a], carry into z0.
                "mov rdx, {z1}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {t2}, {t1}, qword ptr [{p} + 8]",
                "mov {z0}, rdx",
                "shl {z0}, 62",
                "neg {z1}",
                "adc {z2}, {t1}",
                "adc {z3}, 0",
                "adc {a}, {z0}",
                "mov {z0}, 0",
                "adc {z0}, 0",
                "mulx {t1}, {z1}, qword ptr [{p}]",
                "mov {z1}, rdx",
                "shr {z1}, 2",
                "add {z2}, {t1}",
                "adc {z3}, {t2}",
                "adc {a}, 0",
                "adc {z0}, {z1}",
                // Step 2: window [z2, z3, a, z0], carry into z1.
                "mov rdx, {z2}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {t2}, {t1}, qword ptr [{p} + 8]",
                "mov {z1}, rdx",
                "shl {z1}, 62",
                "neg {z2}",
                "adc {z3}, {t1}",
                "adc {a}, 0",
                "adc {z0}, {z1}",
                "mov {z1}, 0",
                "adc {z1}, 0",
                "mulx {t1}, {z2}, qword ptr [{p}]",
                "mov {z2}, rdx",
                "shr {z2}, 2",
                "add {z3}, {t1}",
                "adc {a}, {t2}",
                "adc {z0}, 0",
                "adc {z1}, {z2}",
                // Step 3: window [z3, a, z0, z1], carry into z2.
                "mov rdx, {z3}",
                "imul rdx, qword ptr [{p} + 32]",
                "mulx {t2}, {t1}, qword ptr [{p} + 8]",
                "mov {z2}, rdx",
                "shl {z2}, 62",
                "neg {z3}",
                "adc {a}, {t1}",
                "adc {z0}, 0",
                "adc {z1}, {z2}",
                "mov {z2}, 0",
                "adc {z2}, 0",
                "mulx {t1}, {z3}, qword ptr [{p}]",
                "mov {z3}, rdx",
                "shr {z3}, 2",
                "add {a}, {t1}",
                "adc {z0}, {t2}",
                "adc {z1}, 0",
                "adc {z2}, {z3}",

                // Fold in the high product half; the sum stays below 2p for
                // inputs below 2p, so no carry escapes and a four-limb
                // conditional subtraction suffices.
                "add {a}, {z4}",
                "adc {z0}, {z5}",
                "adc {z1}, {z6}",
                "adc {z2}, {z7}",
                $($tail,)*
                a = inout(reg) value.as_ptr() => o0,
                p = in(reg) params.as_ptr(),
                z0 = out(reg) o1,
                z1 = out(reg) o2,
                z2 = out(reg) o3,
                z3 = out(reg) _,
                z4 = out(reg) _,
                z5 = out(reg) _,
                z6 = out(reg) _,
                z7 = out(reg) _,
                t1 = out(reg) _,
                t2 = out(reg) _,
                out("rdx") _,
                $($extra)*
                options(pure, readonly, nostack),
            );
        }
        [o0, o1, o2, o3]
    }};
}

/// `value^2 * R^{-1} mod p`. Requires BMI2 and ADX.
#[inline(never)]
pub(super) fn square(value: &Limbs, params: &[u64; 5]) -> Limbs {
    square_kernel!(
        value,
        params,
        [
            "movabs rdx, {p3}",                      // p3 = 2^62.
            "mov {t1}, {a}",
            "mov {t2}, {z0}",
            "mov {z3}, {z1}",
            "mov {z4}, {z2}",
            "sub {t1}, qword ptr [{p}]",
            "sbb {t2}, qword ptr [{p} + 8]",
            "sbb {z3}, 0",
            "sbb {z4}, rdx",
            "cmovnc {a}, {t1}",
            "cmovnc {z0}, {t2}",
            "cmovnc {z1}, {z3}",
            "cmovnc {z2}, {z4}",
        ]
        p3 = const PASTA_HIGH_LIMB,
    )
}

/// [`mul`] without its closing subtraction, for inputs below `2p`. The result is below `2p`, by the
/// closure argument above Zakura Udon
/// [`square_run`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/pasta/montgomery.rs#L262-L272).
/// The instructions are those of Udon's x86-64
/// [`montgomery_multiply_loose`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/x86_64_asm/mod.rs#L473-L506),
/// which inlines it. Requires BMI2 and ADX.
#[inline(always)]
pub(super) fn mul_loose(lhs: &Limbs, rhs: &Limbs, params: &[u64; 5]) -> Limbs {
    mul_kernel!(lhs, rhs, params, [])
}

/// [`square`] without its closing subtraction, for inputs below `2p`. The result is below `2p`, as
/// for [`mul_loose`]. Udon's x86-64
/// [`square`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/x86_64_asm/mod.rs#L632-L671).
/// Requires BMI2 and ADX.
#[inline(always)]
pub(super) fn square_loose(value: &Limbs, params: &[u64; 5]) -> Limbs {
    square_kernel!(value, params, [])
}

/// Loads `rdx` from `$y` and adds the row `$x * rdx` into the window `$w0..$w4`, low halves on the
/// CF chain and high halves on the OF chain, both closed into `$w4`. The sum fits the window, so
/// neither chain carries out of it. Clobbers `{s1}` and `{s2}`.
#[rustfmt::skip]
macro_rules! add_row {
    ($y:literal, [$x0:literal, $x1:literal, $x2:literal, $x3:literal],
     [$w0:literal, $w1:literal, $w2:literal, $w3:literal, $w4:literal]) => {
        concat!(
            "mov rdx, qword ptr ", $y, "\n",
            "xor {s1:e}, {s1:e}\n",
            "mulx {s2}, {s1}, qword ptr ", $x0, "\n",
            "adcx ", $w0, ", {s1}\n",
            "adox ", $w1, ", {s2}\n",
            "mulx {s2}, {s1}, qword ptr ", $x1, "\n",
            "adcx ", $w1, ", {s1}\n",
            "adox ", $w2, ", {s2}\n",
            "mulx {s2}, {s1}, qword ptr ", $x2, "\n",
            "adcx ", $w2, ", {s1}\n",
            "adox ", $w3, ", {s2}\n",
            "mulx {s2}, {s1}, qword ptr ", $x3, "\n",
            "adcx ", $w3, ", {s1}\n",
            "adox ", $w4, ", {s2}\n",
            "mov {s1}, 0\n",
            "adcx ", $w4, ", {s1}\n",
            "adox ", $w4, ", {s1}\n",
        )
    };
}

/// The Montgomery step of [`mul`] on the window `$w0..$w4`: `q = $w0 * INV`, the window gains
/// `q * p` and drops `$w0`, which restarts at zero as the next window's fifth limb. Clobbers
/// `{s1}`, `{s2}`, `{s3}` and `rdx`.
#[rustfmt::skip]
macro_rules! cancel_and_shift {
    ($w0:literal, $w1:literal, $w2:literal, $w3:literal, $w4:literal) => {
        concat!(
            "mov rdx, ", $w0, "\n",
            "imul rdx, qword ptr [{p} + 32]\n",
            "mulx {s2}, {s1}, qword ptr [{p} + 8]\n",
            "mov {s3}, rdx\n",
            "shl {s3}, 62\n",
            "neg ", $w0, "\n",
            "adc ", $w1, ", {s1}\n",
            "adc ", $w2, ", 0\n",
            "adc ", $w3, ", {s3}\n",
            "adc ", $w4, ", 0\n",
            "mulx {s1}, {s3}, qword ptr [{p}]\n",
            "mov {s3}, rdx\n",
            "shr {s3}, 2\n",
            "mov ", $w0, ", 0\n",
            "add ", $w1, ", {s1}\n",
            "adc ", $w2, ", {s2}\n",
            "adc ", $w3, ", 0\n",
            "adc ", $w4, ", {s3}\n",
        )
    };
}

/// `(a * b + c * d) * R^{-1} mod p` with one interleaved Montgomery reduction, for `a` and `c`
/// below `p` and `b` and `d` below `2p`; the bounds are those of the AArch64
/// `sum_of_two_products`. The result is canonical for canonical `b` and `d` and below `2p`
/// otherwise. Each round adds the rows `a * b[j]` and `c * d[j]` to the five-limb window of
/// [`mul`] and cancels its low limb. The shifted window stays below `3p < 2^256`, so its fifth limb
/// restarts at zero without the carry capture of [`mul`]. `a` and `c`, and `b` and `d`, are copied
/// next to each other so that two pointers address all sixteen input limbs. Requires BMI2 and ADX.
#[inline(always)]
pub(super) fn sum_of_two_products(
    a: &Limbs,
    b: &Limbs,
    c: &Limbs,
    d: &Limbs,
    params: &[u64; 5],
) -> Limbs {
    let (lhs, rhs): ([Limbs; 2], [Limbs; 2]) = ([*a, *c], [*b, *d]);
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line arithmetic reading only the 37 words behind the three passed pointers
    // (`readonly`); no stack use, and outputs depend only on the declared inputs. All memory
    // addresses are input-independent.
    unsafe {
        asm!(
            // Round 0: the window [ae, be, ce, de, ee] starts as a * b[0].
            "mov rdx, qword ptr [{r}]",
            "mulx {be}, {ae}, qword ptr [{l}]",
            "mulx {ce}, {s1}, qword ptr [{l} + 8]",
            "add {be}, {s1}",
            "mulx {de}, {s1}, qword ptr [{l} + 16]",
            "adc {ce}, {s1}",
            "mulx {ee}, {s1}, qword ptr [{l} + 24]",
            "adc {de}, {s1}",
            "adc {ee}, 0",
            add_row!(
                "[{r} + 32]",
                ["[{l} + 32]", "[{l} + 40]", "[{l} + 48]", "[{l} + 56]"],
                ["{ae}", "{be}", "{ce}", "{de}", "{ee}"]
            ),
            cancel_and_shift!("{ae}", "{be}", "{ce}", "{de}", "{ee}"),
            // Round 1: window [be, ce, de, ee, ae].
            add_row!(
                "[{r} + 8]",
                ["[{l}]", "[{l} + 8]", "[{l} + 16]", "[{l} + 24]"],
                ["{be}", "{ce}", "{de}", "{ee}", "{ae}"]
            ),
            add_row!(
                "[{r} + 40]",
                ["[{l} + 32]", "[{l} + 40]", "[{l} + 48]", "[{l} + 56]"],
                ["{be}", "{ce}", "{de}", "{ee}", "{ae}"]
            ),
            cancel_and_shift!("{be}", "{ce}", "{de}", "{ee}", "{ae}"),
            // Round 2: window [ce, de, ee, ae, be].
            add_row!(
                "[{r} + 16]",
                ["[{l}]", "[{l} + 8]", "[{l} + 16]", "[{l} + 24]"],
                ["{ce}", "{de}", "{ee}", "{ae}", "{be}"]
            ),
            add_row!(
                "[{r} + 48]",
                ["[{l} + 32]", "[{l} + 40]", "[{l} + 48]", "[{l} + 56]"],
                ["{ce}", "{de}", "{ee}", "{ae}", "{be}"]
            ),
            cancel_and_shift!("{ce}", "{de}", "{ee}", "{ae}", "{be}"),
            // Round 3: window [de, ee, ae, be, ce]; the candidate ends in [ee, ae, be, ce].
            add_row!(
                "[{r} + 24]",
                ["[{l}]", "[{l} + 8]", "[{l} + 16]", "[{l} + 24]"],
                ["{de}", "{ee}", "{ae}", "{be}", "{ce}"]
            ),
            add_row!(
                "[{r} + 56]",
                ["[{l} + 32]", "[{l} + 40]", "[{l} + 48]", "[{l} + 56]"],
                ["{de}", "{ee}", "{ae}", "{be}", "{ce}"]
            ),
            cancel_and_shift!("{de}", "{ee}", "{ae}", "{be}", "{ce}"),
            // The closing conditional subtraction of `p` from [`mul`].
            "movabs rdx, {p3}",
            "mov {s1}, {ee}",
            "mov {s2}, {ae}",
            "mov {s3}, {be}",
            "mov {de}, {ce}",
            "sub {s1}, qword ptr [{p}]",
            "sbb {s2}, qword ptr [{p} + 8]",
            "sbb {s3}, 0",
            "sbb {de}, rdx",
            "cmovnc {ee}, {s1}",
            "cmovnc {ae}, {s2}",
            "cmovnc {be}, {s3}",
            "cmovnc {ce}, {de}",
            l = in(reg) lhs.as_ptr(),
            r = in(reg) rhs.as_ptr(),
            p = in(reg) params.as_ptr(),
            p3 = const PASTA_HIGH_LIMB,
            ae = out(reg) o1,
            be = out(reg) o2,
            ce = out(reg) o3,
            de = out(reg) _,
            ee = out(reg) o0,
            s1 = out(reg) _,
            s2 = out(reg) _,
            s3 = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// `lhs + rhs mod 2p`, below `2p`, for inputs below `2p`. `twice_modulus` is `2p`, of the shape
/// `[t0, t1, 0, 2^63]`. The sum can exceed `2^256`, so the top carry joins the one conditional
/// subtraction of `2p`. Zakura Udon x86-64
/// [`add_loose`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/x86_64_asm/mod.rs#L508-L559)
/// with the zero limb of `2p` as an immediate.
#[inline(always)]
pub(super) fn add_loose(lhs: &Limbs, rhs: &Limbs, twice_modulus: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line arithmetic reading only the words behind the two
    // passed references (`readonly`); no stack use, and outputs depend only
    // on the declared inputs. All memory addresses are input-independent.
    unsafe {
        asm!(
            "mov {c}, 0",
            "add {r0}, qword ptr [{b}]",
            "adc {r1}, qword ptr [{b} + 8]",
            "adc {r2}, qword ptr [{b} + 16]",
            "adc {r3}, qword ptr [{b} + 24]",
            "adc {c}, 0",                        // c = carry out of the sum.
            "mov {t0}, {r0}",
            "mov {t1}, {r1}",
            "mov {t2}, {r2}",
            "mov {t3}, {r3}",
            "sub {t0}, qword ptr [{m}]",
            "sbb {t1}, qword ptr [{m} + 8]",
            "sbb {t2}, 0",
            "sbb {t3}, qword ptr [{m} + 24]",
            // CF is set exactly when the sum did not carry and the
            // subtraction borrowed: keep the sum, otherwise reduce.
            "sbb {c}, 0",
            "cmovnc {r0}, {t0}",
            "cmovnc {r1}, {t1}",
            "cmovnc {r2}, {t2}",
            "cmovnc {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b = in(reg) rhs.as_ptr(),
            m = in(reg) twice_modulus.as_ptr(),
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            c = out(reg) _,
            options(pure, readonly, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// `lhs - rhs mod 2p`, below `2p`, for inputs below `2p`, adding `2p` back exactly when the
/// subtraction borrows. `twice_modulus` as for [`add_loose`]. Zakura Udon x86-64
/// [`sub_loose`](https://github.com/zakura-core/common/blob/c4d62d2e84f0db8a19f075a8d42fb24c2f373d4a/crates/udon/src/field/x86_64_asm/mod.rs#L561-L606)
/// with the zero limb of `2p` as an immediate.
#[inline(always)]
pub(super) fn sub_loose(lhs: &Limbs, rhs: &Limbs, twice_modulus: &Limbs) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line arithmetic reading only the words behind the two
    // passed references (`readonly`); no stack use, and outputs depend only
    // on the declared inputs. The conditional loads use fixed,
    // input-independent addresses.
    unsafe {
        asm!(
            "sub {r0}, qword ptr [{b}]",
            "sbb {r1}, qword ptr [{b} + 8]",
            "sbb {r2}, qword ptr [{b} + 16]",
            "sbb {r3}, qword ptr [{b} + 24]",
            // MOV and CMOV preserve the borrow flag from the subtraction.
            "mov {m0}, 0",
            "mov {m1}, 0",
            "mov {m3}, 0",
            "cmovc {m0}, qword ptr [{m}]",
            "cmovc {m1}, qword ptr [{m} + 8]",
            "cmovc {m3}, qword ptr [{m} + 24]",
            "add {r0}, {m0}",
            "adc {r1}, {m1}",
            "adc {r2}, 0",
            "adc {r3}, {m3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b = in(reg) rhs.as_ptr(),
            m = in(reg) twice_modulus.as_ptr(),
            m0 = out(reg) _,
            m1 = out(reg) _,
            m3 = out(reg) _,
            options(pure, readonly, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// `-value mod 2p`, below `2p`, for an input below `2p`: [`sub_loose`] from zero.
#[inline(always)]
pub(super) fn neg_loose(value: &Limbs, twice_modulus: &Limbs) -> Limbs {
    sub_loose(&[0; 4], value, twice_modulus)
}

/// The canonical representative of a value below `2p`: one conditional subtraction of `p`, the
/// closing instructions of [`mul`] with `p[3]` read from `params`.
#[inline(always)]
pub(super) fn reduce_loose(value: &Limbs, params: &[u64; 5]) -> Limbs {
    let [mut r0, mut r1, mut r2, mut r3] = *value;
    // SAFETY: straight-line arithmetic reading only the words behind the
    // passed reference (`readonly`); no stack use, and outputs depend only
    // on the declared inputs. All memory addresses are input-independent.
    unsafe {
        asm!(
            "mov {t0}, {r0}",
            "mov {t1}, {r1}",
            "mov {t2}, {r2}",
            "mov {t3}, {r3}",
            "sub {t0}, qword ptr [{p}]",
            "sbb {t1}, qword ptr [{p} + 8]",
            "sbb {t2}, 0",
            "sbb {t3}, qword ptr [{p} + 24]",
            "cmovnc {r0}, {t0}",
            "cmovnc {r1}, {t1}",
            "cmovnc {r2}, {t2}",
            "cmovnc {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            p = in(reg) params.as_ptr(),
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, readonly, nostack),
        );
    }
    [r0, r1, r2, r3]
}
