#!/usr/bin/env python3
"""Check a BN curve's `GtGlsParams` and print its `adj0_div_r` constants.

`bn::gls4_digits` writes `k = \\sum_i k_i p^i (mod r)` by Babai rounding against a reduced basis of
the lattice `{v : \\sum_i v_i lambda^i = 0 (mod r)}`, `lambda = p mod r`. The rounding
`beta_j = round(k * adj0[j] / r)` is done as `round(k * g_j / 2^M)` with `M = 64 * (N + 2)` and a
precomputed `g_j = round(2^M * |adj0[j]| / r)`, so the division by `r` becomes a shift, as in
`glv_fast_decomp.py`. This script checks that every basis row lies in the lattice, that
`det(basis) = r`, and that `adj0` is the first row of `adj(basis)`, then prints `adj0_div_r`.

Edit `CURVES` below (the basis and `adj0` mirror the curve's `GT_GLS`) and run:

    python3 scripts/bn_gls_decomp.py
"""

from fractions import Fraction

CURVES = {
    "BN254": dict(
        x=4965661367192848881,
        limbs=4,
        basis=[
            [9931322734385697762, 4965661367192848882, -4965661367192848881, 4965661367192848881],
            [4965661367192848882, 4965661367192848881, 4965661367192848881, -9931322734385697762],
            [-4965661367192848881, 4965661367192848881, -4965661367192848881, -9931322734385697763],
            [9931322734385697763, -4965661367192848881, -4965661367192848882, -4965661367192848881],
        ],
        adj0=[
            (True, [0x113C366715DEDAF5, 0xD7ADF45CF590C4C8, 0x1DF623EF8AF183E3, 0x0]),
            (True, [0x620AAA6F726909F1, 0x46FB76A5E4491EC5, 0x1DF623EF8AF183E4, 0x0]),
            (False, [0xD8378506DD96F60E, 0x46FB76A5E4491EC4, 0x1DF623EF8AF183E4, 0x0]),
            (True, [0x934DF252932DEC1D, 0x46FB76A5E4491EC4, 0x1DF623EF8AF183E4, 0x0]),
        ],
    ),
}


def from_limbs(limbs):
    return sum(w << (64 * i) for i, w in enumerate(limbs))


def limbs_le(x, n):
    assert x >> (64 * n) == 0, "value does not fit in the requested number of limbs"
    return [(x >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(n)]


def inverse(m):
    n = len(m)
    a = [[Fraction(v) for v in row] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        p = next(i for i in range(c, n) if a[i][c] != 0)
        a[c], a[p] = a[p], a[c]
        a[c] = [v / a[c][c] for v in a[c]]
        for i in range(n):
            if i != c and a[i][c] != 0:
                a[i] = [vi - a[i][c] * vc for vi, vc in zip(a[i], a[c])]
    return [row[n:] for row in a]


def det(m):
    if len(m) == 1:
        return m[0][0]
    return sum((-1) ** j * m[0][j] * det([row[:j] + row[j + 1:] for row in m[1:]]) for j in range(len(m)))


for name, c in CURVES.items():
    x, N, basis = c["x"], c["limbs"], c["basis"]
    r = 36 * x**4 + 36 * x**3 + 18 * x**2 + 6 * x + 1
    p = 36 * x**4 + 36 * x**3 + 24 * x**2 + 6 * x + 1
    lam = p % r
    for j, row in enumerate(basis):
        assert sum(v * lam**i for i, v in enumerate(row)) % r == 0, f"{name}: row {j} not in lattice"
    d = det(basis)
    assert d == r, f"{name}: det != r"
    inv = inverse(basis)
    adj0 = [inv[0][j] * d for j in range(4)]
    stored = [(1 if s else -1) * from_limbs(l) for s, l in c["adj0"]]
    assert all(a.denominator == 1 for a in adj0) and [int(a) for a in adj0] == stored, f"{name}: adj0"
    for i in range(4):
        bound = sum(abs(basis[j][i]) for j in range(4))
        assert bound // 2 + 1 < 2**64, f"{name}: digit {i} may exceed 64 bits"

    M = 64 * (N + 2)
    fmt = lambda g: ", ".join(f"0x{w:016x}" for w in limbs_le(g, N + 1))
    print(f"// {name}")
    print("        adj0_div_r: [")
    for a in stored:
        print(f"            [{fmt((2**M * abs(a) + r // 2) // r)}],")
    print("        ],")
