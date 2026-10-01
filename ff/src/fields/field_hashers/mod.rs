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
        // `expand_message_xmd`'s Z_pad is `s_in_bytes` long: the hash's input block size
        // ([RFC 9380](https://www.rfc-editor.org/rfc/rfc9380.html) Section 5.3.1).
        Self::new_given_z_pad_len::<F>(dst, H::block_size())
    }

    fn hash_to_field<const N: usize>(&self, message: &[u8]) -> [F; N] {
        self.hash::<F, N>(message)
    }
}

impl<H: FixedOutputReset + Default + Clone, const SEC_PARAM: usize>
    DefaultFieldHasher<H, SEC_PARAM>
{
    /// A hasher whose `expand_message_xmd` Z_pad is `z_pad_len` bytes long.
    fn new_given_z_pad_len<F: Field>(dst: &[u8], z_pad_len: usize) -> Self {
        // The final output of `hash_to_field` will be an array of field
        // elements from F::BaseField, each of size `len_per_elem`.
        let len_per_base_elem = get_len_per_elem::<F, SEC_PARAM>();

        let expander = ExpanderXmd {
            hasher: PhantomData,
            dst: dst.to_vec(),
            block_size: z_pad_len,
        };

        DefaultFieldHasher {
            expander,
            len_per_base_elem,
        }
    }

    fn hash<F: Field, const N: usize>(&self, message: &[u8]) -> [F; N] {
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

/// [`DefaultFieldHasher`] as released in ark-ff 0.4 and 0.5, whose `expand_message_xmd` Z_pad is
/// the per-element length `ceil((log2(p) + SEC_PARAM) / 8)` instead of the hash's input block size.
/// Output matches RFC 9380 only when the two are equal, as for SHA-256 on 381-bit fields.
/// Reproduces field elements, and so curve points, hashed with those releases.
pub struct LegacyFieldHasher<H: FixedOutputReset + Default + Clone, const SEC_PARAM: usize = 128>(
    DefaultFieldHasher<H, SEC_PARAM>,
);

impl<F: Field, H: FixedOutputReset + Default + Clone, const SEC_PARAM: usize> HashToField<F>
    for LegacyFieldHasher<H, SEC_PARAM>
{
    fn new(dst: &[u8]) -> Self {
        Self(DefaultFieldHasher::new_given_z_pad_len::<F>(
            dst,
            get_len_per_elem::<F, SEC_PARAM>(),
        ))
    }

    fn hash_to_field<const N: usize>(&self, message: &[u8]) -> [F; N] {
        self.0.hash::<F, N>(message)
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
            field_hashers::{DefaultFieldHasher, HashToField, LegacyFieldHasher},
            PrimeField,
        },
        bls12_381,
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

    /// `HashToField::new` with SHA-384 and SHA-512 takes their 128-byte block as the Z_pad, on the
    /// RFC 9380 appendix J.8.1 DST, with the expected values from a Python `expand_message_xmd`
    /// over `hashlib`.
    #[test]
    fn test_hash_to_field_sha384_sha512_block_size() {
        fn check<H: digest::FixedOutputReset + Default + Clone + digest::core_api::BlockSizeUser>(
            vectors: [(&str, [&str; 2]); 2],
        ) where
            DefaultFieldHasher<H, 128>: HashToField<Fq>,
        {
            let hasher = <DefaultFieldHasher<H, 128> as HashToField<Fq>>::new(
                b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_",
            );
            for (msg, want) in vectors {
                let got: [Fq; 2] = hasher.hash_to_field(msg.as_bytes());
                let want = want.map(|u| Fq::from_be_bytes_mod_order(&hex::decode(u).unwrap()));
                assert_eq!(got, want, "msg = {msg:?}");
            }
        }
        check::<sha2::Sha384>([
            (
                "",
                [
                    "77584cd08349aa6ebb6d7e511bb58d4a4c30eb07ba9bcb64e19cb62d17352a54",
                    "1f4e3b062a26c9a11a718b00a11283cc01b39105a3280f5d0de093267080cc7b",
                ],
            ),
            (
                "abc",
                [
                    "c054081108c0f44dbb71b0447f567421fbedaabaf1a7497e184491aa6e1a9407",
                    "cd1fd5df9b7c3745875338b48b8e50b9b6ca29cddc58d47816d3637498bfefe7",
                ],
            ),
        ]);
        check::<sha2::Sha512>([
            (
                "",
                [
                    "e5fdfdd81349a5327255d12eeb74571654f4016e7de187df8d35ccb1f8bc9d20",
                    "97d526d6060326f4e83062167c4b0a3895d550e56948e6b65ad01c2494854d2b",
                ],
            ),
            (
                "abc",
                [
                    "d91abdd89ac56a16422606f5aa2cd8e4f672a3c840275785686740a5b32339b2",
                    "e95477246b7afcaea17388114adf19a31fc1d3037224d15cd6d17096f06e7d74",
                ],
            ),
        ]);
    }

    /// `LegacyFieldHasher` on the RFC 9380 appendix J.8.1 inputs, with the expected values computed
    /// by `expand_message_xmd` with a 48-byte Z_pad.
    #[test]
    fn test_legacy_hash_to_field_secp256k1() {
        let hasher = <LegacyFieldHasher<Sha256, 128> as HashToField<Fq>>::new(
            b"QUUX-V01-CS02-with-secp256k1_XMD:SHA-256_SSWU_RO_",
        );
        let vectors: [(&str, [&str; 2]); 3] = [
            (
                "",
                [
                    "09c84b120e693ffc6c7f0fce162aef5f996c50beac4b102685a4ed9888f5c322",
                    "b92d8673fe1d1ba210d06b178160b65e64bfb8e95bd162cb9c62ae2750d2c0e2",
                ],
            ),
            (
                "abc",
                [
                    "0d55eddbbd9b8435ad371665d6942b51d340934eb461555a61737e9a730383b5",
                    "550a31478c70fe98b1c778a91398425a0ff9ca88d41d003346c9f46b6fe04998",
                ],
            ),
            (
                "abcdef0123456789",
                [
                    "8984b558c9895cf7fa8c55bb4602ef054cf7d6efe2659805d01b73ec49667174",
                    "425cfa137baa113ab5ea91d03b9d9bbda9a48e172064277f0644dcd52eff1877",
                ],
            ),
        ];
        for (msg, want) in vectors {
            let got: [Fq; 2] = hasher.hash_to_field(msg.as_bytes());
            let want = want.map(|u| Fq::from_be_bytes_mod_order(&hex::decode(u).unwrap()));
            assert_eq!(got, want, "msg = {msg:?}");
        }
    }

    /// `LegacyFieldHasher` equals `DefaultFieldHasher` when the element length is SHA-256's 64-byte
    /// block, as for BLS12-381's base field.
    #[test]
    fn test_legacy_hash_to_field_matches_default_for_64_byte_elements() {
        let dst = b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_";
        let default = <DefaultFieldHasher<Sha256, 128> as HashToField<bls12_381::Fq>>::new(dst);
        let legacy = <LegacyFieldHasher<Sha256, 128> as HashToField<bls12_381::Fq>>::new(dst);
        for msg in ["", "abc", "abcdef0123456789"] {
            let want: [bls12_381::Fq; 2] = default.hash_to_field(msg.as_bytes());
            let got: [bls12_381::Fq; 2] = legacy.hash_to_field(msg.as_bytes());
            assert_eq!(got, want, "msg = {msg:?}");
        }
    }
}
