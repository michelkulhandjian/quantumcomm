"""
Verify the closed-form Hamada-style rank formula for the
Singer-circulant point-hyperplane incidence matrix A of PG(d, p)
over F_p (with p prime, q = p):

   rank_{F_p}(A) = binom(p + d - 1, d) + 1

Test on:  (p, d) in { (3,2), (3,3), (3,5), (5,2), (5,3), (7,2), (7,3) }.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from math import comb


def build_field(p, m, poly_coeffs):
    n_field = p**m
    field_size = n_field - 1
    assert poly_coeffs[-1] == 1
    poly = poly_coeffs[:-1]

    def mul_x(a):
        new = [0] * m
        new[0] = (-poly[0] * a[m-1]) % p
        for k in range(1, m):
            new[k] = (a[k-1] - poly[k] * a[m-1]) % p
        return tuple(new)

    one = tuple(1 if k == 0 else 0 for k in range(m))
    cur = one
    seen = {one: 0}
    alpha_powers = [one]
    for k in range(1, field_size):
        cur = mul_x(cur)
        if cur == one or cur in seen:
            return None
        seen[cur] = k
        alpha_powers.append(cur)
    if mul_x(cur) != one:
        return None
    return alpha_powers


def find_primitive(p, m):
    """Return first primitive monic polynomial of degree m over F_p."""
    import itertools
    for coeffs in itertools.product(range(p), repeat=m):
        if coeffs[0] == 0:
            continue
        poly = list(coeffs) + [1]
        ap = build_field(p, m, poly)
        if ap is not None:
            return poly, ap
    return None, None


def trace_table(p, m, alpha_powers):
    field_size = p**m - 1
    traces = []
    for i in range(field_size):
        s = [0] * m
        for k in range(m):
            ap = alpha_powers[(i * p**k) % field_size]
            for r in range(m):
                s[r] = (s[r] + ap[r]) % p
        traces.append(s[0])
    return traces


def rank_mod_p(M, p):
    M = (M.copy() % p).astype(int)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        inv = pow(int(M[r, c]) % p, -1, p)
        M[r] = (M[r] * inv) % p
        for i in range(rows):
            if i != r and M[i, c] % p != 0:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
        if r == rows:
            break
    return r


def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M


def test(p, d):
    m = d + 1
    print(f"\n--- (p, d) = ({p}, {d}) ---")
    print(f"  Searching primitive polynomial of degree {m} over F_{p}...")
    poly, alpha_powers = find_primitive(p, m)
    if poly is None:
        print("  FAILED to find primitive polynomial.")
        return
    field_size = p**m - 1
    traces = trace_table(p, m, alpha_powers)
    n = field_size // (p - 1)
    tau_H = np.array([1 if traces[i % field_size] == 0 else 0
                      for i in range(n)], dtype=int)
    A = circ(tau_H) % p
    r = rank_mod_p(A, p)
    pred = comb(p + d - 1, d) + 1
    match = (r == pred)
    print(f"  n = theta_{d}({p}) = {n}")
    print(f"  rank_F{p}(A) = {r}")
    print(f"  predicted    = C({p}+{d}-1, {d}) + 1 = C({p+d-1}, {d}) + 1 = {pred}")
    print(f"  match: {'YES' if match else 'NO'}")
    return r, pred, match


print("Verifying Hamada-Smith closed-form: rank_{F_p}(A) = C(p+d-1, d) + 1")
print("for q = p prime in PG(d, p)")
print("=" * 60)

cases = [
    (3, 2), (3, 3), (3, 5),
    (5, 2), (5, 3),
    (7, 2), (7, 3),
    (11, 2),
]
results = []
for p, d in cases:
    r = test(p, d)
    if r is not None:
        results.append((p, d) + r)

print("\n" + "=" * 60)
print("Summary:")
print(f"{'(p,d)':<10} {'computed':<12} {'predicted':<12} {'match':<8}")
print("-" * 50)
for p, d, r, pred, match in results:
    print(f"({p},{d}){'':<5} {r:<12} {pred:<12} {'YES' if match else 'NO'}")
