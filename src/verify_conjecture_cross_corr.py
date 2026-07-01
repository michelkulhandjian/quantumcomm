"""
Numerical verification of Conjecture 6.X" (cross-correlation modular constancy)
for (q, d) = (5, 3) and (7, 3).

Conjecture:  For q odd and d odd >= 3, the cross-correlation
   nu_i = |E cap (i - Q)|       (i in G = Z/nZ,   n = theta_d(q))
satisfies  nu_i  ==  c  (mod q)   for all i,  for some constant c in F_q.

Equivalently:  M_Q . tau_H  ==  c . 1   (mod q).

If true, Lemma 6.X applies and  d_min(Q(q,d)) <= 2  whenever n is even.

Script:
  - Build K = F_{q^{d+1}} from a primitive polynomial.
  - Build tau_H, tau_Q on K^*/F^*.
  - Compute the cyclic cross-correlation.
  - Check unique values, and unique values mod q.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import itertools


def build_field(p, m, poly_coeffs):
    """F_{p^m} = F_p[x] / <poly>.  poly_coeffs = [c_0, ..., c_m] (monic).
    Returns None if poly is not primitive (i.e. x doesn't have full
    order p^m - 1 in (F_p[x]/<poly>)^*)."""
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
        if cur == one:
            return None  # x has order k < field_size: not primitive
        if cur in seen:
            return None  # cycle returned earlier than expected
        seen[cur] = k
        alpha_powers.append(cur)
    # Final mul should land back on one
    if mul_x(cur) != one:
        return None
    return alpha_powers


def trace_table(p, m, alpha_powers):
    """Returns trace[i] = Tr_{K/F_p}(alpha^i) for i = 0, ..., field_size - 1."""
    field_size = p**m - 1
    traces = []
    for i in range(field_size):
        s = [0] * m
        for k in range(m):
            ap = alpha_powers[(i * p**k) % field_size]
            for r in range(m):
                s[r] = (s[r] + ap[r]) % p
        # s should be in F_p (only first coord non-zero)
        assert all(s[k] == 0 for k in range(1, m)), \
            f"Tr(alpha^{i}) = {s} not in F_p"
        traces.append(s[0])
    return traces


def cross_correlate(tau_H, tau_Q):
    """nu[k] = sum_j tau_H[(k+j) mod n] * tau_Q[j] = |E cap (k - Q^*)|.
    Equivalent to  M_Q^T  applied to tau_H,  but we'll do it directly."""
    n = len(tau_H)
    nu = np.zeros(n, dtype=int)
    for k in range(n):
        s = 0
        for j in range(n):
            if tau_H[(k + j) % n] and tau_Q[j]:
                s += 1
        nu[k] = s
    return nu


def find_primitive(p, m):
    """Search for a primitive polynomial of degree m over F_p."""
    # Iterate over monic polynomials c_0 + c_1 x + ... + c_{m-1} x^{m-1} + x^m
    for coeffs in itertools.product(range(p), repeat=m):
        poly_coeffs = list(coeffs) + [1]
        # Quick check: c_0 != 0 (otherwise x divides poly)
        if poly_coeffs[0] == 0:
            continue
        ap = build_field(p, m, poly_coeffs)
        if ap is not None:
            return poly_coeffs, ap
    return None, None


def probe(q, d):
    print(f"\n=== Probe (q, d) = ({q}, {d}) ===")
    m = d + 1
    print(f"Searching for primitive polynomial of F_{q}^{m} = F_{q**m}...")
    poly, alpha_powers = find_primitive(q, m)
    if poly is None:
        print("  NO primitive polynomial found (very unlikely).")
        return None
    print(f"  Primitive polynomial: f(x) = {poly}  (coeffs c_0..c_m of monic f)")
    field_size = q**m - 1
    print(f"  |K^*| = {field_size}, alpha^|K^*| = 1 confirmed.")

    traces = trace_table(q, m, alpha_powers)
    n_E = sum(1 for t in traces if t == 0)
    print(f"  |E| in K^* = {n_E}  (expect q^d - 1 = {q**d - 1})")

    n_cosets = field_size // (q - 1)
    print(f"  n = |K^*/F^*| = {n_cosets}  (expect theta_d(q) = {(q**m - 1)//(q-1)})")

    # tau_H on cosets:  index i in [0, n_cosets), tau_H[i] = 1 iff Tr(alpha^i) = 0
    tau_H = np.array([1 if traces[i % field_size] == 0 else 0
                      for i in range(n_cosets)], dtype=int)
    # formal Q (q odd):  Q = { z : Tr(z^2) = 0 }
    xi = 2 if q % 2 == 1 else q + 1
    tau_Q = np.array([1 if traces[(xi * i) % field_size] == 0 else 0
                      for i in range(n_cosets)], dtype=int)
    print(f"  weight(tau_H) = {int(tau_H.sum())}  (expect theta_{{d-1}}(q) = "
          f"{(q**d - 1)//(q-1)})")
    print(f"  weight(tau_Q) = {int(tau_Q.sum())}")

    # Cross-correlation
    print("  Computing cross-correlation E * Q^{-1} ...")
    nu = cross_correlate(tau_H, tau_Q)
    nu_unique = sorted(set(int(x) for x in nu))
    print(f"  nu unique values: {nu_unique}")
    nu_mod_q = sorted(set(int(x) % q for x in nu))
    print(f"  nu unique values mod {q}: {nu_mod_q}")

    if len(nu_mod_q) == 1:
        print(f"  *** Conjecture 6.X\" HOLDS: nu == {nu_mod_q[0]} (mod {q}). ***")
        print(f"  *** Lemma 6.X applies => d_min <= 2 if n is even. ***")
        if n_cosets % 2 == 0:
            print(f"  *** n = {n_cosets} is EVEN => d_min(Q({q},{d})) <= 2 GUARANTEED. ***")
        else:
            print(f"  *** n = {n_cosets} is ODD => no 2-torsion obstruction. ***")
    else:
        print(f"  *** Conjecture 6.X\" FAILS: nu takes "
              f"{len(nu_mod_q)} different values mod {q}. ***")
    return nu, nu_mod_q


# --- (5, 3): n = (5^4 - 1)/4 = 156, EVEN  ---
probe(5, 3)

# --- (7, 3): n = (7^4 - 1)/6 = 400, EVEN ---
probe(7, 3)

# --- additional cross-checks ---
# (3, 5): n = (3^6 - 1)/2 = 364, EVEN
probe(3, 5)

# (5, 5): n = (5^6 - 1)/4 = 3906, EVEN
# Skip -- too large for naive O(n^2) cross-correlation in pure Python.
# probe(5, 5)
