//! x86-64 inline-assembly kernels for the 4-limb [`super::MontAccumulator`], computing the same
//! functions as `deferred/aarch64.rs` under the same carry analysis. Each schoolbook row adds its
//! low product words on the CF chain (`adcx`) and its high words on the OF chain (`adox`), consumes
//! the previous row's pending carry at word `i + 4` and leaves its own at word `i + 5`. The carry
//! out of word `i + 4` is 0 or 1, so at most one chain carries and their sum is the next pending
//! carry. The eight accumulator limbs, the pending carry, both pointers, two product halves and
//! `rdx` exceed the thirteen registers inline assembly gets, so the rows run in two blocks of two
//! that hand the pending carry over in a register. Both kernels need BMI2 and ADX.
//!
//! Ported from Zakura Udon (#554) [`x86_64_asm/products.rs`][products.rs]:
//! [`mul_accumulate`][mul_accumulate], [`partial_reduce`][partial_reduce].
//!
//! [products.rs]: https://github.com/zakura-core/common/blob/3c6598da967555d95a641ae95d80f07bd8df6354/crates/udon/src/field/x86_64_asm/products.rs
//! [mul_accumulate]: https://github.com/zakura-core/common/blob/3c6598da967555d95a641ae95d80f07bd8df6354/crates/udon/src/field/x86_64_asm/products.rs#L25-L138
//! [partial_reduce]: https://github.com/zakura-core/common/blob/3c6598da967555d95a641ae95d80f07bd8df6354/crates/udon/src/field/x86_64_asm/products.rs#L140-L229

use core::arch::asm;

/// One schoolbook row. `rdx` takes the multiplier limb at `$a`, and the pending carry in `{ovf}`
/// is consumed at the row's fifth word `$d4` before `{ovf}` receives the row's own.
macro_rules! accumulate_row {
    ($a:literal, $d0:literal, $d1:literal, $d2:literal, $d3:literal, $d4:literal) => {
        concat!(
            "mov rdx, qword ptr ",
            $a,
            "\n",
            // Clear CF and OF; the register is overwritten by the first MULX.
            "xor {lo:e}, {lo:e}\n",
            "mulx {hi}, {lo}, qword ptr [{b}]\n",
            "adcx ",
            $d0,
            ", {lo}\n",
            "adox ",
            $d1,
            ", {hi}\n",
            "mulx {hi}, {lo}, qword ptr [{b} + 8]\n",
            "adcx ",
            $d1,
            ", {lo}\n",
            "adox ",
            $d2,
            ", {hi}\n",
            "mulx {hi}, {lo}, qword ptr [{b} + 16]\n",
            "adcx ",
            $d2,
            ", {lo}\n",
            "adox ",
            $d3,
            ", {hi}\n",
            "mulx {hi}, {lo}, qword ptr [{b} + 24]\n",
            "adcx ",
            $d3,
            ", {lo}\n",
            "adox ",
            $d4,
            ", {hi}\n",
            // Close the CF chain into the fifth word with the pending carry, then collect both
            // chains' carries.
            "mov {lo}, 0\n",
            "adcx ",
            $d4,
            ", {ovf}\n",
            "mov {ovf}, 0\n",
            "adcx {ovf}, {lo}\n",
            "adox {ovf}, {lo}\n",
        )
    };
}

/// `accumulator + lhs * rhs`, as eight words and a one-bit overflow.
#[inline(always)]
pub(super) fn mul_accumulate(
    accumulator: [u64; 8],
    lhs: &[u64; 4],
    rhs: &[u64; 4],
) -> ([u64; 8], u64) {
    let [mut d0, mut d1, mut d2, mut d3, mut d4, mut d5, mut d6, mut d7] = accumulator;
    let mut overflow: u64 = 0;
    // SAFETY: straight-line arithmetic reading only the eight words behind the two references
    // (`readonly`), with all inputs and clobbers declared. No stack use, no data-dependent
    // addresses or control flow.
    unsafe {
        // Rows 0 and 1 touch accumulator words 0 through 5.
        asm!(
            accumulate_row!("[{a}]", "{d0}", "{d1}", "{d2}", "{d3}", "{d4}"),
            accumulate_row!("[{a} + 8]", "{d1}", "{d2}", "{d3}", "{d4}", "{d5}"),
            d0 = inout(reg) d0,
            d1 = inout(reg) d1,
            d2 = inout(reg) d2,
            d3 = inout(reg) d3,
            d4 = inout(reg) d4,
            d5 = inout(reg) d5,
            ovf = inout(reg) overflow,
            a = in(reg) lhs.as_ptr(),
            b = in(reg) rhs.as_ptr(),
            lo = out(reg) _,
            hi = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
        // Rows 2 and 3 touch accumulator words 2 through 7.
        asm!(
            accumulate_row!("[{a} + 16]", "{d2}", "{d3}", "{d4}", "{d5}", "{d6}"),
            accumulate_row!("[{a} + 24]", "{d3}", "{d4}", "{d5}", "{d6}", "{d7}"),
            d2 = inout(reg) d2,
            d3 = inout(reg) d3,
            d4 = inout(reg) d4,
            d5 = inout(reg) d5,
            d6 = inout(reg) d6,
            d7 = inout(reg) d7,
            ovf = inout(reg) overflow,
            a = in(reg) lhs.as_ptr(),
            b = in(reg) rhs.as_ptr(),
            lo = out(reg) _,
            hi = out(reg) _,
            out("rdx") _,
            options(pure, readonly, nostack),
        );
    }
    ([d0, d1, d2, d3, d4, d5, d6, d7], overflow)
}

/// `limbs[0..7] + limbs[7] * b448 + carry * r2`, where `b448 = 2^448 mod p` and
/// `r2 = 2^512 mod p`. Requires `b448 < 2^253` and `r2 < 2^252`, so the folding term is below
/// `2^318` and fits in five words, and the result is below `2^449`. Out of line: the block takes
/// every register inline assembly gets.
#[inline(never)]
pub(super) fn partial_reduce(
    limbs: [u64; 8],
    carry: u64,
    b448: &[u64; 4],
    r2: &[u64; 4],
) -> [u64; 8] {
    debug_assert!(b448[3] < (1 << 61));
    debug_assert!(r2[3] < (1 << 60));
    let [mut d0, mut d1, mut d2, mut d3, mut d4, mut d5, mut d6, b7] = limbs;
    let d7;
    let fold = [
        b448[0], b448[1], b448[2], b448[3], r2[0], r2[1], r2[2], r2[3],
    ];
    // SAFETY: straight-line arithmetic reading only the eight folding words behind the declared
    // pointer (`readonly`), with all inputs and clobbers declared. No stack use. The bounds above
    // ensure the folded value cannot carry out of the eighth word.
    unsafe {
        asm!(
            // Clear CF and OF and zero the eighth word.
            "xor {d7:e}, {d7:e}",
            // Add b7 * b448 at words 0 through 4.
            "mulx {hi}, {lo}, qword ptr [{f}]",
            "adcx {d0}, {lo}",
            "adox {d1}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 8]",
            "adcx {d1}, {lo}",
            "adox {d2}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 16]",
            "adcx {d2}, {lo}",
            "adox {d3}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 24]",
            "adcx {d3}, {lo}",
            "adox {d4}, {hi}",
            // Close both chains through the top word.
            "mov {lo}, 0",
            "adcx {d4}, {lo}",
            "adox {d5}, {lo}",
            "adcx {d5}, {lo}",
            "adox {d6}, {lo}",
            "adcx {d6}, {lo}",
            "adox {d7}, {lo}",
            "adcx {d7}, {lo}",
            // Add carry * r2 the same way. The spent carry register becomes the zero register,
            // and its XOR clears both chains.
            "mov rdx, {c}",
            "xor {c:e}, {c:e}",
            "mulx {hi}, {lo}, qword ptr [{f} + 32]",
            "adcx {d0}, {lo}",
            "adox {d1}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 40]",
            "adcx {d1}, {lo}",
            "adox {d2}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 48]",
            "adcx {d2}, {lo}",
            "adox {d3}, {hi}",
            "mulx {hi}, {lo}, qword ptr [{f} + 56]",
            "adcx {d3}, {lo}",
            "adox {d4}, {hi}",
            "adcx {d4}, {c}",
            "adox {d5}, {c}",
            "adcx {d5}, {c}",
            "adox {d6}, {c}",
            "adcx {d6}, {c}",
            "adox {d7}, {c}",
            "adcx {d7}, {c}",
            d0 = inout(reg) d0,
            d1 = inout(reg) d1,
            d2 = inout(reg) d2,
            d3 = inout(reg) d3,
            d4 = inout(reg) d4,
            d5 = inout(reg) d5,
            d6 = inout(reg) d6,
            d7 = out(reg) d7,
            f = in(reg) fold.as_ptr(),
            c = inout(reg) carry => _,
            lo = out(reg) _,
            hi = out(reg) _,
            inout("rdx") b7 => _,
            options(pure, readonly, nostack),
        );
    }
    [d0, d1, d2, d3, d4, d5, d6, d7]
}
