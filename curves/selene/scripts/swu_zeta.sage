# From sage script specified [in IETF draft here](https://www.rfc-editor.org/rfc/rfc9380.html#section-6.6.2)

# Arguments:
# - F, a field object, e.g., F = GF(2^521 - 1)
# - A and B, the coefficients of the curve y^2 = x^3 + A * x + B
def find_z_sswu(F, A, B):
    R.<xx> = F[]                       # Polynomial ring over F
    g = xx^3 + F(A) * xx + F(B)        # y^2 = g(x) = x^3 + A * x + B
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

p = 57896044618658097711785492504343953926623305935230693509004809574567321395027
F_p = GF(p)
A = -3
B = 25675911719867737339625140396204798989996478324626569376465022644547366285284
zeta = find_z_sswu(F_p, A, B)