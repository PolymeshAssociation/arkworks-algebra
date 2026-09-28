use quote::quote;

pub(super) fn add_with_carry_impl(num_limbs: usize) -> proc_macro2::TokenStream {
    let mut body = proc_macro2::TokenStream::new();
    body.extend(quote! {
        use ark_ff::biginteger::arithmetic::adc_for_add_with_carry as adc;
        let mut carry = 0;
    });
    for i in 0..num_limbs {
        body.extend(quote! {
            carry = adc(&mut a.0[#i], b.0[#i], carry);
        });
    }
    body.extend(quote! {
        carry != 0
    });
    quote! {
        #[inline(always)]
        fn __add_with_carry(
            a: &mut B,
            b: & B,
        ) -> bool {
            #body
        }
    }
}

pub(super) fn sub_with_borrow_impl(num_limbs: usize) -> proc_macro2::TokenStream {
    let mut body = proc_macro2::TokenStream::new();
    body.extend(quote! {
        use ark_ff::biginteger::arithmetic::sbb_for_sub_with_borrow as sbb;
        let mut borrow = 0;
    });
    for i in 0..num_limbs {
        body.extend(quote! {
            borrow = sbb(&mut a.0[#i], b.0[#i], borrow);
        });
    }
    body.extend(quote! {
        borrow != 0
    });
    quote! {
        #[inline(always)]
        fn __sub_with_borrow(
            a: &mut B,
            b: & B,
        ) -> bool {
            #body
        }
    }
}

pub(super) fn subtract_modulus_impl(
    modulus: &proc_macro2::TokenStream,
    num_limbs: usize,
) -> proc_macro2::TokenStream {
    // Mask selection beats the branch for 6 limbs and loses for 4 (rotating-input final
    // exponentiation: BLS12-381 -3.5%, BN254 +5%), so only fields wider than 4 limbs use it.
    // The mask select is zkcrypto's `Fp::subtract_p` (https://github.com/zkcrypto/bls12_381/blob/5d22dd74c2a14fb9f3d3b85ae2c39d0c669ddd99/src/fp.rs#L361-L379).
    if num_limbs > 4 {
        quote! {
            /// `a -= MODULUS` if `a >= MODULUS`, selected by mask rather than a branch.
            #[inline(always)]
            fn __subtract_modulus(a: &mut F) {
                let mut t = a.0;
                let borrow = __sub_with_borrow(&mut t, &#modulus);
                let keep = 0u64.wrapping_sub(borrow as u64);
                for i in 0..t.0.len() {
                    a.0.0[i] = (a.0.0[i] & keep) | (t.0[i] & !keep);
                }
            }

            /// `a -= MODULUS` if `carry` or `a >= MODULUS`, selected by mask.
            #[inline(always)]
            fn __subtract_modulus_with_carry(a: &mut F, carry: bool) {
                let mut t = a.0;
                let borrow = __sub_with_borrow(&mut t, &#modulus);
                let keep = 0u64.wrapping_sub((borrow & !carry) as u64);
                for i in 0..t.0.len() {
                    a.0.0[i] = (a.0.0[i] & keep) | (t.0[i] & !keep);
                }
            }
        }
    } else {
        quote! {
            #[inline(always)]
            fn __subtract_modulus(a: &mut F) {
                if a.is_geq_modulus() {
                    __sub_with_borrow(&mut a.0, &#modulus);
                }
            }

            #[inline(always)]
            fn __subtract_modulus_with_carry(a: &mut F, carry: bool) {
                if a.is_geq_modulus() || carry {
                    __sub_with_borrow(&mut a.0, &#modulus);
                }
            }
        }
    }
}
