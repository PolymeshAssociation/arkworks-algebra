import argparse

def find_cube_root(p):
    """Find a non-trivial cube root of unity in F_p."""
    if p % 3 != 1:
        return None
    # Fermat's little theorem: a^(p-1) = 1 mod p
    # So (a^((p-1)/3))^3 = 1 mod p
    exp = (p - 1) // 3
    for i in range(2, 1000):
        root = pow(i, exp, p)
        if root != 1:
            return root
    return None

def find_square_root_minus_one(p):
    """Find a square root of -1 in F_p."""
    if p % 4 != 1:
        return None
    exp = (p - 1) // 4
    for i in range(2, 1000):
        root = pow(i, exp, p)
        if pow(root, 2, p) == p - 1:
            return root
    return None

def gauss_lattice_reduction(u, v):
    """Find shortest basis vectors for the lattice spanned by u and v."""
    while True:
        if u[0]**2 + u[1]**2 > v[0]**2 + v[1]**2:
            u, v = v, u
        dot_uv = v[0]*u[0] + v[1]*u[1]
        dot_uu = u[0]*u[0] + u[1]*u[1]
        m = round(dot_uv / dot_uu)
        if m == 0:
            return u, v
        v = [v[0] - m*u[0], v[1] - m*u[1]]

def limbs_le(x, n):
    assert x >> (64 * n) == 0, "value does not fit in the requested number of limbs"
    # out is 64-bit limbs of x in little-endian
    out = [(x >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(n)]
    return out

def compute_fast_decomp(r, a12, a22, n12, n22, limbs):
    # 2 spare limbs to handle error from division
    M = 64 * (limbs + 2)
    # rounding trick: add 1/2 to numerator since python's integer division is equivalent to float
    g1 = (2**M * a22 + r // 2) // r  # round(2^M * |n22| / r)
    g2 = (2**M * a12 + r // 2) // r  # round(2^M * |n12| / r)
    
    fmt = lambda g: ",\n            ".join(f"0x{w:016x}" for w in limbs_le(g, limbs + 1))
    
    print("  const FAST_DECOMP: Option<ark_ec::scalar_mul::glv::GLVFastDecomp<Self::ScalarField>> = Some(ark_ec::scalar_mul::glv::GLVFastDecomp {")
    print(f"      g1: &[\n            {fmt(g1)},\n      ],")
    print(f"      g2: &[\n            {fmt(g2)},\n      ],")
    print(f'      a12: ark_ff::MontFp!("{a12}"),')
    print(f'      a22: ark_ff::MontFp!("{a22}"),')
    print(f"      negate_k2: {str(n12 * n22 < 0).lower()},")
    print("  });\n")

def main():
    parser = argparse.ArgumentParser(description="Find GLV parameters for a given curve.")
    parser.add_argument("--p", type=int, required=True, help="Base field modulus p")
    parser.add_argument("--r", type=int, required=True, help="Scalar field modulus r")
    parser.add_argument("--A", type=int, required=True, help="Curve equation coefficient A")
    parser.add_argument("--B", type=int, required=True, help="Curve equation coefficient B")
    args = parser.parse_args()

    p, r, A, B = args.p, args.r, args.A, args.B

    supported = False
    beta = None
    lam = None
    
    if A == 0 and B != 0:
        # y^2 = x^3 + B, supports GLV if p = 1 (mod 3)
        if p % 3 == 1 and r % 3 == 1:
            supported = True
            beta = find_cube_root(p)
            lam_candidate = find_cube_root(r)
            
            print(f"Curve supports GLV endomorphism (x, y) -> (beta * x, y).")
            # Usually we need to match beta and lambda such that phi(P) = lambda P
            lam = lam_candidate
            lam2 = (-1 - lam) % r # Since lam^2 + lam + 1 = 0 mod r
            
            print(f"BETA (for ENDO_COEFFS) = {beta}")
            print(f"Or possibly BETA^2 = {pow(beta, 2, p)}")
            print(f"LAMBDA = {lam} or {lam2}")
            print(f"NOTE: You must verify which lambda corresponds to which beta by checking phi(P) == lambda * P on a random point.")
            
    elif B == 0 and A != 0:
        if p % 4 == 1 and r % 4 == 1:
            supported = True
            beta = find_square_root_minus_one(p) # This will be used to multiply y
            lam = find_square_root_minus_one(r)
            print(f"Curve supports GLV endomorphism (x, y) -> (-x, beta * y).")
            lam2 = r - lam
            print(f"BETA (for y coordinate) = {beta}")
            print(f"LAMBDA = {lam} or {lam2}")
            print(f"NOTE: Arkworks GLVConfig usually only multiplies the x-coordinate by ENDO_COEFFS[0]. For B=0, you need an endomorphism over both x and y.")
    else:
        print("Curve does not support GLV (requires A=0 or B=0 with appropriate field modulus).")
        return

    if supported and lam is not None:
        print("\nCalculating SCALAR_DECOMP_COEFFS...")
        # Compute lattice basis for lam
        u = [r, 0]
        v = [-lam, 1]
        v1, v2 = gauss_lattice_reduction(u, v)
        
        # Ensure det is positive r
        det = v1[0]*v2[1] - v1[1]*v2[0]
        if det < 0:
            v2 = [-v2[0], -v2[1]]
            
        # Verify the lattice basis
        n11, n12, n21, n22 = v1[0], v1[1], v2[0], v2[1]
        assert (n11 + n12 * lam) % r == 0, "row 1 not in GLV lattice"
        assert (n21 + n22 * lam) % r == 0, "row 2 not in GLV lattice"
        assert abs(n11 * n22 - n12 * n21) == r, "determinant != r"
        
        print(f"For LAMBDA = {lam}:")
        print(f"  const SCALAR_DECOMP_COEFFS: [(bool, <Self::ScalarField as ark_ff::PrimeField>::BigInt); 4] = [")
        for x in [n11, n12, n21, n22]:
            sign = "true" if x >= 0 else "false"
            print(f'      ({sign}, ark_ff::BigInt!("{abs(x)}")),')
        print(f"  ];\n")
        
        # Calculate limbs (number of 64-bit words needed for scalar field)
        limbs = (r.bit_length() + 63) // 64
        print("Calculating FAST_DECOMP...")
        compute_fast_decomp(r, abs(n12), abs(n22), n12, n22, limbs)

if __name__ == "__main__":
    main()
