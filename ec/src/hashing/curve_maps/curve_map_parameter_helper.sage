# Arguments:
# - F, a field object, e.g., F = GF(2^521 - 1)
# - A and B, the coefficients of the curve y^2 = x^3 + A * x + B
def find_z_sswu(F, A, B):
    R.<xx> = F[]
    # Polynomial ring over F
    g = xx^3 + F(A) * xx + F(B)
    # y^2 = g(x) = x^3 + A * x + B
    ctr = F.gen()
    while True:
        for Z_cand in (F(ctr), F(-ctr)):
            # Criterion 1: Z is non-square in F.
            if is_square(Z_cand):
                continue
            # Criterion 2: Z != -1 in F.
            if Z_cand == F(-1):
                continue
            # Criterion 3: g(x) - Z is irreducible over F.
            if not (g - Z_cand).is_irreducible():
                continue
            # Criterion 4: g(B / (Z * A)) is square in F.
            if is_square(g(B / (Z_cand * A))):
                return Z_cand
        ctr += 1

# Finds the smallest z in term of non-zero bit
# in sage representation for constructing
# elligator2 map for a curve defined over field F.
# Argument:
# - F, a field object, e.g., F = GF(2^255 - 19)
def find_z_ell2(F):
    ctr = F.gen()
    while True:
        for Z_cand in (F(ctr), F(-ctr)):
            # Z must be a non-square in F.
            if is_square(Z_cand):
                continue
            return Z_cand
        ctr += 1

# Finds Z for the Shallue-van de Woestijne map, from RFC 9380 appendix H.1
# <https://www.rfc-editor.org/rfc/rfc9380.html#appendix-H.1>.
# Arguments:
# - F, a field object, e.g., F = GF(2^521 - 1)
# - A and B, the coefficients of the curve y^2 = x^3 + A * x + B
# - init_ctr, the first candidate. RFC 9380 starts at 1 and falls back to F.gen().
def find_z_svdw(F, A, B, init_ctr=1):
    g = lambda x: F(x)^3 + F(A) * F(x) + F(B)
    h = lambda Z: -(F(3) * Z^2 + F(4) * A) / (F(4) * g(Z))
    ctr = init_ctr
    while True:
        for Z_cand in (F(ctr), F(-ctr)):
            # Criterion 1: g(Z) != 0 in F.
            if g(Z_cand) == 0:
                continue
            # Criterion 2: -(3 * Z^2 + 4 * A) / (4 * g(Z)) != 0 in F.
            if h(Z_cand) == 0:
                continue
            # Criterion 3: -(3 * Z^2 + 4 * A) / (4 * g(Z)) is square in F.
            if not is_square(h(Z_cand)):
                continue
            # Criterion 4: At least one of g(Z) and g(-Z / 2) is square in F.
            if is_square(g(Z_cand)) or is_square(g(-Z_cand / F(2))):
                return Z_cand
        ctr += 1

# sgn0 of RFC 9380 section 4.1 for F = GF(p) or GF(p^2).
def sgn0(x):
    c = list(x.polynomial()) + [0, 0] if x.parent().degree() > 1 else [x, 0]
    return (ZZ(c[0]) % 2 == 1) or (c[0] == 0 and ZZ(c[1]) % 2 == 1)

# Returns (Z, C1, C2, C3, C4) of `SVDWConfig` for y^2 = x^3 + A * x + B over F,
# as in RFC 9380 appendix F.1 <https://www.rfc-editor.org/rfc/rfc9380.html#appendix-F.1>.
# BN254 (F = GF(p), A = 0, B = 3 for G1; F = GF(p^2) with u^2 = -1, A = 0,
# B = 3 / (u + 9) for G2) gives Z = 1 for both.
def svdw_constants(F, A, B, init_ctr=1):
    A, B = F(A), F(B)
    g = lambda x: x^3 + A * x + B
    Z = find_z_svdw(F, A, B, init_ctr)
    c3 = sqrt(-g(Z) * (3 * Z^2 + 4 * A))
    if sgn0(c3):
        c3 = -c3
    return (Z, g(Z), -Z / 2, c3, -4 * g(Z) / (3 * Z^2 + 4 * A))
