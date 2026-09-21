mod expander;

use core::marker::PhantomData;

use crate::{Field, PrimeField};

use digest::{crypto_common::BlockSizeUser, FixedOutputReset, XofReader};
use expander::Expander;

use self::expander::ExpanderXmd;

/// Trait for hashing messages to field elements.
pub trait HashToField<F: Field>: Sized {
    /// Initialises a new hash-to-field helper struct.
    ///
    /// # Arguments
    ///
    /// * `domain` - bytes that get concatenated with the `msg` during hashing, in order to separate potentially interfering instantiations of the hasher.
    fn new(domain: &[u8]) -> Self;

    /// Hash an arbitrary `msg` to `N` elements of the field `F`.
    fn hash_to_field<const N: usize>(&self, msg: &[u8]) -> [F; N];
}

/// This field hasher constructs a Hash-To-Field based on a fixed-output hash function,
/// like SHA2, SHA3 or Blake2.
///
/// The implementation aims to follow the specification in [Hashing to Elliptic Curves (draft)](https://tools.ietf.org/pdf/draft-irtf-cfrg-hash-to-curve-13.pdf).
///
/// # Examples
///
/// ```
/// use ark_ff::fields::field_hashers::{DefaultFieldHasher, HashToField};
/// use ark_test_curves::bls12_381::Fq;
/// use sha2::Sha256;
///
/// let hasher = <DefaultFieldHasher<Sha256> as HashToField<Fq>>::new(&[1, 2, 3]);
/// let field_elements: [Fq; 2] = hasher.hash_to_field(b"Hello, World!");
///
/// assert_eq!(field_elements.len(), 2);
/// ```
pub struct DefaultFieldHasher<H: FixedOutputReset + Default + Clone, const SEC_PARAM: usize = 128> {
    expander: ExpanderXmd<H>,
    len_per_base_elem: usize,
}

impl<F: Field, H: FixedOutputReset + Default + Clone + BlockSizeUser, const SEC_PARAM: usize>
    HashToField<F> for DefaultFieldHasher<H, SEC_PARAM>
{
    fn new(dst: &[u8]) -> Self {
        // The final output of `hash_to_field` will be an array of field
        // elements from F::BaseField, each of size `len_per_elem`.
        let len_per_base_elem = get_len_per_elem::<F, SEC_PARAM>();

        let expander = ExpanderXmd {
            hasher: PhantomData,
            dst: dst.to_vec(),
            // `expand_message_xmd`'s Z_pad is `s_in_bytes` long: the hash's input block size
            // ([RFC 9380](https://www.rfc-editor.org/rfc/rfc9380.html) Section 5.3.1).
            block_size: H::block_size(),
        };

        DefaultFieldHasher {
            expander,
            len_per_base_elem,
        }
    }

    fn hash_to_field<const N: usize>(&self, message: &[u8]) -> [F; N] {
        let m = F::extension_degree() as usize;

        // The user requests `N` of elements of F_p^m to output per input msg,
        // each field element comprising `m` BasePrimeField elements.
        let len_in_bytes = N * m * self.len_per_base_elem;
        let uniform_bytes = self.expander.expand(message, len_in_bytes);

        let cb = |i| {
            let base_prime_field_elem = |j| {
                let elm_offset = self.len_per_base_elem * (j + i * m);
                F::BasePrimeField::from_be_bytes_mod_order(
                    &uniform_bytes[elm_offset..][..self.len_per_base_elem],
                )
            };
            F::from_base_prime_field_elems((0..m).map(base_prime_field_elem)).unwrap()
        };
        ark_std::array::from_fn(cb)
    }
}

pub fn hash_to_field<F: Field, H: XofReader, const SEC_PARAM: usize>(h: &mut H) -> F {
    // The final output of `hash_to_field` will be an array of field
    // elements from F::BaseField, each of size `len_per_elem`.
    let len_per_base_elem = get_len_per_elem::<F, SEC_PARAM>();
    // Rust *still* lacks alloca, hence this ugly hack.
    let mut alloca = [0u8; 2048];
    let alloca = &mut alloca[0..len_per_base_elem];

    let m = F::extension_degree() as usize;

    let base_prime_field_elem = |_| {
        h.read(alloca);
        F::BasePrimeField::from_be_bytes_mod_order(alloca)
    };
    F::from_base_prime_field_elems((0..m).map(base_prime_field_elem)).unwrap()
}

/// This function computes the length in bytes that a hash function should output
/// for hashing an element of type `Field`.
/// See section 5.1 and 5.3 of the
/// [IETF hash standardization draft](https://datatracker.ietf.org/doc/draft-irtf-cfrg-hash-to-curve/14/)
const fn get_len_per_elem<F: Field, const SEC_PARAM: usize>() -> usize {
    // ceil(log(p))
    let base_field_size_in_bits = F::BasePrimeField::MODULUS_BIT_SIZE as usize;
    // ceil(log(p)) + security_parameter
    let base_field_size_with_security_padding_in_bits = base_field_size_in_bits + SEC_PARAM;
    // ceil( (ceil(log(p)) + security_parameter) / 8)
    let bytes_per_base_field_elem =
        base_field_size_with_security_padding_in_bits.div_ceil(8) as u64;
    bytes_per_base_field_elem as usize
}

#[cfg(test)]
mod test {
    use ark_test_curves::{
        ark_ff::{
            field_hashers::{DefaultFieldHasher, HashToField},
            PrimeField,
        },
        secp256k1::Fq,
    };
    use sha2::Sha256;

    /// `hash_to_field` for secp256k1's base field, whose 48-byte elements differ from SHA-256's
    /// 64-byte block, against RFC 9380 appendix J.8.1
    /// (<https://www.rfc-editor.org/rfc/rfc9380#appendix-J.8.1>).
    #[test]
    fn test_hash_to_field_secp256k1_rfc9380() {
        let hasher = <DefaultFieldHasher<Sha256, 128> as HashToField<Fq>>::new(
            b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_",
        );
        let vectors: [(&str, [&str; 2]); 3] = [
            (
                "",
                [
                    "6b0f9910dd2ba71c78f2ee9f04d73b5f4c5f7fc773a701abea1e573cab002fb3",
                    "1ae6c212e08fe1a5937f6202f929a2cc8ef4ee5b9782db68b0d5799fd8f09e16",
                ],
            ),
            (
                "abc",
                [
                    "128aab5d3679a1f7601e3bdf94ced1f43e491f544767e18a4873f397b08a2b61",
                    "5897b65da3b595a813d0fdcc75c895dc531be76a03518b044daaa0f2e4689e00",
                ],
            ),
            (
                "abcdef0123456789",
                [
                    "ea67a7c02f2cd5d8b87715c169d055a22520f74daeb080e6180958380e2f98b9",
                    "7434d0d1a500d38380d1f9615c021857ac8d546925f5f2355319d823a478da18",
                ],
            ),
        ];
        for (msg, want) in vectors {
            let got: [Fq; 2] = hasher.hash_to_field(msg.as_bytes());
            let want = want.map(|u| Fq::from_be_bytes_mod_order(&hex::decode(u).unwrap()));
            assert_eq!(got, want, "msg = {msg:?}");
        }
    }
}
