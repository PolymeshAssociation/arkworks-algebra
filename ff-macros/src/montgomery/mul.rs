use quote::quote;

/// `with_asm == false` emits the body without the assembly branch, for
/// `MontConfig::mul_assign_portable`: the oracle the assembly kernels are checked and timed
/// against is this modulus-literal code, which a build without the assembly runs, not the trait's
/// generic body, which measures 6% apart on aarch64.
pub(super) fn mul_assign_impl(
    can_use_no_carry_mul_opt: bool,
    num_limbs: usize,
    modulus_limbs: &[u64],
    modulus_has_spare_bit: bool,
    with_asm: bool,
) -> proc_macro2::TokenStream {
    let mut body = proc_macro2::TokenStream::new();
    let modulus_0 = modulus_limbs[0];
    if can_use_no_carry_mul_opt {
        // This modular multiplication algorithm uses Montgomery
        // reduction for efficient implementation. It also additionally
        // uses the "no-carry optimization" outlined
        // [here](https://hackmd.io/@gnark/modular_multiplication) if
        // `MODULUS` has (a) a non-zero MSB, and (b) at least one
        // zero bit in the rest of the modulus.

        let mut default = proc_macro2::TokenStream::new();
        default.extend(quote! { let mut r = [0u64; #num_limbs]; });
        for i in 0..num_limbs {
            default.extend(quote! {
                let mut carry1 = 0u64;
                r[0] = fa::mac(r[0], (a.0).0[0], (b.0).0[#i], &mut carry1);
                let k = r[0].wrapping_mul(Self::INV);
                let mut carry2 = 0u64;
                fa::mac_discard(r[0], k, #modulus_0, &mut carry2);
            });
            for (j, modulus_j) in modulus_limbs.iter().enumerate().take(num_limbs).skip(1) {
                let idx = j - 1;
                default.extend(quote! {
                    r[#j] = fa::mac_with_carry(r[#j], (a.0).0[#j], (b.0).0[#i], &mut carry1);
                    r[#idx] = fa::mac_with_carry(r[#j], k, #modulus_j, &mut carry2);
                });
            }
            default.extend(quote!(r[#num_limbs - 1] = carry1 + carry2;));
        }
        default.extend(quote!((a.0).0 = r;));
        // Avoid using assembly for `N == 1`.
        if with_asm && (2..=6).contains(&num_limbs) {
            body.extend(quote!({
                if ark_ff::asm_backend_active() {
                    Self::asm_mul_assign(a, b);
                    return;
                }
            }))
        }
        body.extend(quote!({ #default }));
        body.extend(quote!(__subtract_modulus(a);));
    } else {
        // We use standard CIOS
        let double_limbs = num_limbs * 2;
        body.extend(quote! {
            let mut scratch = [0u64; #double_limbs];
        });
        for i in 0..num_limbs {
            body.extend(quote! { let mut carry = 0u64; });
            for j in 0..num_limbs {
                let k = i + j;
                body.extend(quote!{scratch[#k] = fa::mac_with_carry(scratch[#k], (a.0).0[#i], (b.0).0[#j], &mut carry);});
            }
            body.extend(quote! { scratch[#i + #num_limbs] = carry; });
        }
        body.extend(quote!( let mut carry2 = 0u64; ));
        for i in 0..num_limbs {
            body.extend(quote! {
                let tmp = scratch[#i].wrapping_mul(Self::INV);
                let mut carry = 0u64;
                fa::mac(scratch[#i], tmp, #modulus_0, &mut carry);
            });
            for (j, modulus_j) in modulus_limbs.iter().enumerate().take(num_limbs).skip(1) {
                let k = i + j;
                body.extend(quote!(scratch[#k] = fa::mac_with_carry(scratch[#k], tmp, #modulus_j, &mut carry);));
            }
            body.extend(quote!(carry2 = fa::adc(&mut scratch[#i + #num_limbs], carry, carry2);));
        }
        body.extend(quote! {
            (a.0).0 = scratch[#num_limbs..].try_into().unwrap();
        });
        if modulus_has_spare_bit {
            body.extend(quote!(__subtract_modulus(a);));
        } else {
            body.extend(quote!(__subtract_modulus_with_carry(a, carry2 != 0);));
        }
    }
    body
}
