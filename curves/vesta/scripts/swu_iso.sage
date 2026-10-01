# Checks the constants in `src/curves/swu_iso.rs` against Sage.
# Run from the crate directory: `sage scripts/swu_iso.sage`.
#
# E: y^2 = x^3 + 5 is Vesta. E_iso: y^2 = x^3 + A*x + B is the 3-isogenous curve used by the
# simplified SWU map. A, B, ZETA and the isogeny constants `iso` are from pasta_curves
# `src/curves.rs` (https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs):
# `IsoEq` lines 1070-1087, `Eq::ISOGENY_CONSTANTS` 1191-1270, `Eq::Z` 1273-1278. The isogeny is
#   x' = (iso0*x^3 + iso1*x^2 + iso2*x + iso3) / (x^2 + iso4*x + iso5)
#   y' = y * (iso6*x^3 + iso7*x^2 + iso8*x + iso9) / (x^3 + iso10*x^2 + iso11*x + iso12)

import re

p = 28948022309329048855892746252171976963363056481941647379679742748393362948097
r = 28948022309329048855892746252171976963363056481941560715954676764349967630337
E_B = 5

A = 17413348858408915339762682399132325137863850198379221683097628341577494210225
B = 1265
ZETA = -13
iso = [
    25731575386070265649682441113041757300767161317281464337493104665238544842753,
    13377367003779316331268047403600734872799183885837485433911493934102207511749,
    11064082577423419940183149293632076317553812518550871517841037420579891210813,
    22515128462811482443472135973911537638171266152621281295306466582083726737451,
    4604213796697651557841441623718706001740429044770779386484474413346415813353,
    9250006497141849826017568406346290940322373181457057184910582871723433210981,
    8577191795356755216560813704347252433589053772427154779164368221746181614251,
    21162694656554182593580396827886355918081120183889566406795618341247785229923,
    11620280474556824258112134491145636201000922752744881519070727793732904824884,
    13937936667454727226911322269564285204582212380194126516142098360337545123123,
    21380331849711001764708535561664047484292171808126992769566582994216305194078,
    27750019491425549478052705219038872820967119544371171554731748615170299632943,
    28948022309329048855892746252171976963363056481941647379679742748393362947557,
]

Fp = GF(p)
E = EllipticCurve(Fp, [0, E_B])
E_iso = EllipticCurve(Fp, [A, B])

# Isogenous curves have the same order, so E_iso has prime order r and cofactor 1.
assert r.is_prime()
assert E.order() == r
assert E_iso.order() == r
assert not Fp(ZETA).is_square()

# `IsogenyMap` coefficients, lowest degree first.
x_num = [iso[3], iso[2], iso[1], iso[0]]
x_den = [iso[5], iso[4], 1]
y_num = [iso[9], iso[8], iso[7], iso[6]]
y_den = [iso[12], iso[11], iso[10], 1]

# The rational map above equals a degree-3 isogeny E_iso -> E computed by Sage, composed with one
# of the automorphisms of E.
found = False
for phi in E_iso.isogenies_prime_degree(3):
    if phi.codomain().j_invariant() != E.j_invariant():
        continue
    for to_E in phi.codomain().isomorphisms(E):
        psi = to_E * phi
        fx, fy = psi.rational_maps()
        X, Y = fx.parent().gens()
        poly = lambda c: sum(Fp(ci) * X^i for i, ci in enumerate(c))
        if fx == poly(x_num) / poly(x_den) and fy == Y * poly(y_num) / poly(y_den):
            found = True
assert found, "isogeny constants do not match a 3-isogeny from E_iso to E"

# The denominators vanish only at the x-coordinate x_K of the kernel points {O, K, -K}, and K is not
# an F_p-point. The leading coefficients are the scaling (x, y) -> (l^2 x, l^3 y) of an isomorphism.
Rx.<x> = Fp[]
x_K = -Fp(iso[4]) / 2
assert Rx(x_den) == (x - x_K)^2
assert Rx(y_den) == (x - x_K)^3
assert not (x_K^3 + A * x_K + B).is_square()
assert Fp(iso[0])^3 == Fp(iso[6])^2

# Generator of E_iso: smallest x with x^3 + A*x + B a square, and the even y.
gx = Fp(0)
while not (gx^3 + A * gx + B).is_square():
    gx += 1
gy = (gx^3 + A * gx + B).sqrt()
if Integer(gy) % 2 == 1:
    gy = -gy
G = E_iso(gx, gy)
assert r * G == E_iso(0)

# The square-root constants: with p - 1 = T * 2^32, T odd, `ZETA_TRACE_POWER = ZETA^((T - 1) / 2)`,
# and `ZETA_OVER_ROOT_SQRT` is either square root of ZETA / g for the two-adic root of unity
# g = 5^T, 5 being the field's multiplicative generator.
T = (p - 1) >> 32
assert T % 2 == 1
g = Fp(5)^T
assert g^(2^31) == -1
zeta_trace_power = Fp(ZETA)^((T - 1) / 2)

# Compare with `src/curves/swu_iso.rs`. Its `MontFp!` literals in file order are COEFF_A, COEFF_B,
# the generator x and y, ZETA_TRACE_POWER, ZETA_OVER_ROOT_SQRT, ZETA, then the isogeny map
# coefficients.
src = open("src/curves/swu_iso.rs").read()
lits = [Fp(Integer(v)) for v in re.findall(r'MontFp!\("(-?\d+)"\)', src)]
expected = [A, B, gx, gy, zeta_trace_power, None, ZETA] + x_num + x_den + y_num + y_den
assert len(lits) == len(expected), "unexpected number of MontFp! literals"
assert lits[5]^2 * g == Fp(ZETA), "ZETA_OVER_ROOT_SQRT does not square to ZETA / g"
assert [l for l, e in zip(lits, expected) if e is not None] == [
    Fp(e) for e in expected if e is not None
], "swu_iso.rs constants do not match"

print("E_iso order == r, ZETA is a non-square")
print("isogeny matches Sage; denominators are (x - x_K)^2 and (x - x_K)^3 with K not in E_iso(F_p)")
print("generator: x = %d, y = %d" % (gx, gy))
print("x_map_numerator:   %s" % x_num)
print("x_map_denominator: %s" % x_den)
print("y_map_numerator:   %s" % y_num)
print("y_map_denominator: %s" % y_den)
print("ZETA_TRACE_POWER = %d" % zeta_trace_power)
print("swu_iso.rs matches")
