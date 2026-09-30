//! Run-time detection of BMI2 (`mulx`) and ADX (`adcx`, `adox`), which the x86-64 Montgomery
//! kernels need. A binary built with `-C target-feature=+bmi2,+adx` faults on a CPU older than
//! Broadwell or Zen, so a distributed binary cannot carry the flags. The kernels are raw `asm!`
//! text that assembles on a baseline target, so a cached CPUID check selects them with one
//! predictable branch and no `#[target_feature]` call boundary.

use core::sync::atomic::{AtomicU8, Ordering};

const UNKNOWN: u8 = 0;
const ABSENT: u8 = 1;
const PRESENT: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Whether this CPU has both BMI2 and ADX, and the portable path has not been forced.
///
/// Cached after the first call. A race computes the same answer twice and stores it twice, so
/// `Relaxed` is enough.
#[inline]
pub(crate) fn has_bmi2_adx() -> bool {
    match STATE.load(Ordering::Relaxed) {
        PRESENT => true,
        ABSENT => false,
        _ => {
            let yes = detect();
            STATE.store(if yes { PRESENT } else { ABSENT }, Ordering::Relaxed);
            yes
        },
    }
}

#[cold]
fn detect() -> bool {
    #[cfg(feature = "std")]
    {
        // Differential testing needs a way to run the whole stack on the fallback in one process.
        extern crate std;
        if std::env::var("ARK_FF_FORCE_PORTABLE").as_deref() == Ok("1") {
            return false;
        }
    }
    cpu_has_bmi2_adx()
}

/// Reads CPUID directly rather than going through `std::is_x86_feature_detected!`, which needs
/// `std`. Leaf 7 subleaf 0: `EBX` bit 8 is BMI2, bit 19 is ADX.
#[inline]
fn cpu_has_bmi2_adx() -> bool {
    // Safe intrinsics on current toolchains, `unsafe` on older ones.
    #[allow(unsafe_code, unused_unsafe)]
    unsafe {
        use core::arch::x86_64::{__cpuid, __cpuid_count};
        // Leaf 7 postdates the original CPUID, so confirm the CPU reports it before asking.
        if __cpuid(0).eax < 7 {
            return false;
        }
        leaf7_has_bmi2_adx(__cpuid_count(7, 0).ebx)
    }
}

#[inline(always)]
const fn leaf7_has_bmi2_adx(ebx: u32) -> bool {
    ebx & (1 << 8) != 0 && ebx & (1 << 19) != 0
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn detection_is_cached_and_stable() {
        let first = has_bmi2_adx();
        for _ in 0..1000 {
            assert_eq!(
                has_bmi2_adx(),
                first,
                "detection must not change within a process"
            );
        }
        assert_ne!(
            STATE.load(Ordering::Relaxed),
            UNKNOWN,
            "state must be cached after a call"
        );
    }

    /// A CPU with ADX but not BMI2, or the reverse, must not select the kernel: it needs both.
    #[test]
    fn both_features_are_required() {
        let bmi2_only = 1u32 << 8;
        let adx_only = 1u32 << 19;
        assert!(!leaf7_has_bmi2_adx(0));
        assert!(!leaf7_has_bmi2_adx(bmi2_only));
        assert!(!leaf7_has_bmi2_adx(adx_only));
        assert!(leaf7_has_bmi2_adx(bmi2_only | adx_only));
    }

    #[test]
    fn agrees_with_std_detection() {
        assert_eq!(
            cpu_has_bmi2_adx(),
            std::is_x86_feature_detected!("bmi2") && std::is_x86_feature_detected!("adx"),
            "hand-rolled CPUID disagrees with std's detection"
        );
    }
}
