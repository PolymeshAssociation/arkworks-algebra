# Checks the constants in `src/curves/swu_iso.rs` against Sage.
# Run from the crate directory: `sage scripts/swu_iso.sage`.
#
# E: y^2 = x^3 + 5 is Pallas. E_iso: y^2 = x^3 + A*x + B is the 3-isogenous curve used by the
# simplified SWU map. A, B, ZETA and the isogeny constants `iso` are from pasta_curves
# `src/curves.rs` (https://github.com/zcash/pasta_curves/blob/41e8149b028ff14569801782c665228af8e17af4/src/curves.rs):
# `IsoEp` lines 1052-1069, `Ep::ISOGENY_CONSTANTS` 1091-1170, `Ep::Z` 1173-1178. The isogeny is
#   x' = (iso0*x^3 + iso1*x^2 + iso2*x + iso3) / (x^2 + iso4*x + iso5)
#   y' = y * (iso6*x^3 + iso7*x^2 + iso8*x + iso9) / (x^3 + iso10*x^2 + iso11*x + iso12)

import re

p = 28948022309329048855892746252171976963363056481941560715954676764349967630337
r = 28948022309329048855892746252171976963363056481941647379679742748393362948097
E_B = 5

A = 10949663248450308183708987909873589833737836120165333298109615750520499732811
B = 1265
ZETA = -13
iso = [
    6432893846517566412420610278260439325191790329320346825767705947633326140075,
    23989696149150192365340222745168215001509815558210986772351135915822265203574,
    10492611921771203378452795982353351666191589197598957448093274638589204800759,
    12865787693035132824841220556520878650383580658640693651535411895266652280192,
    13271109177048389296812780941310096270046944650307955939477485891950613419807,
    22768321103861051515190775253992702316905399997697804654926324362758820947460,
    11793638718615538422771118843477472096184948937087302513907460903994431256804,
    11994848074575096182670111372584107500754907779105493386175567957911132601787,
    28823569610051396102362669851238297121581474897215657071023781420043761726004,
    1072148974419594402070101713043406554198631721553391137627950991272221023311,
    5432652610908059517272798285879155923388888734491153551238890455750936314542,
    10408918692925056833786833257634153023990087029210292532869619559576527581706,
    28948022309329048855892746252171976963363056481941560715954676764349967629797,
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

# Compare with `src/curves/swu_iso.rs`. Its `MontFp!` literals in file order are COEFF_A, COEFF_B,
# the generator x and y, ZETA, then the isogeny map coefficients.
src = open("src/curves/swu_iso.rs").read()
lits = [Fp(Integer(v)) for v in re.findall(r'MontFp!\("(-?\d+)"\)', src)]
expected = [A, B, gx, gy, ZETA] + x_num + x_den + y_num + y_den
assert len(lits) == len(expected), "unexpected number of MontFp! literals"
assert lits == [Fp(v) for v in expected], "swu_iso.rs constants do not match"

print("E_iso order == r, ZETA is a non-square")
print("isogeny matches Sage; denominators are (x - x_K)^2 and (x - x_K)^3 with K not in E_iso(F_p)")
print("generator: x = %d, y = %d" % (gx, gy))
print("x_map_numerator:   %s" % x_num)
print("x_map_denominator: %s" % x_den)
print("y_map_numerator:   %s" % y_num)
print("y_map_denominator: %s" % y_den)
print("swu_iso.rs matches")
