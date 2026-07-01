"""
Singer-Frobenius symmetry-amplified ISD for Q(7,2) at weights 6, 7, 8.

The Singer cyclic group (Z/57Z) and Frobenius (x -> 7x mod 57) together
generate a group of order n * ord(q mod n) = 57 * 3 = 171 acting on
position indices.  Witnesses come in orbits of size <= 171.  We canonicalise
each subset by taking the lex-min image over all 171 transformations.

Speed-up: ~171x over full enumeration.

Estimated work:
  w=6: C(57,6)/171 = 36M/171 ~ 212K subsets, ~3min.
  w=7: 264M/171 ~ 1.5M subsets, ~25min.
  w=8: 1.65B/171 ~ 9.6M subsets, ~3hr.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools, time
import numpy as np
from math import comb


def build_field(p, m, poly_coeffs):
    fs = p**m - 1
    poly = poly_coeffs[:-1]
    def mul_x(a):
        new = [0] * m
        new[0] = (-poly[0] * a[m-1]) % p
        for k in range(1, m):
            new[k] = (a[k-1] - poly[k] * a[m-1]) % p
        return tuple(new)
    one = tuple(1 if k == 0 else 0 for k in range(m))
    cur = one; seen = {one: 0}; aps = [one]
    for kk in range(1, fs):
        cur = mul_x(cur)
        if cur == one or cur in seen: return None
        seen[cur] = kk; aps.append(cur)
    if mul_x(cur) != one: return None
    return aps

def find_primitive(p, m):
    for c in itertools.product(range(p), repeat=m):
        if c[0] == 0: continue
        poly = list(c) + [1]
        ap = build_field(p, m, poly)
        if ap is not None: return poly, ap
    return None, None

def trace_table(p, m, aps):
    fs = p**m - 1
    out = []
    for i in range(fs):
        s = [0]*m
        for k in range(m):
            ap = aps[(i*p**k) % fs]
            for r in range(m): s[r] = (s[r] + ap[r]) % p
        out.append(s[0])
    return out

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n): M[i, j] = c[(i - j) % n]
    return M

def right_kernel_basis(M, p):
    M2 = (M.copy().astype(int) % p)
    rows, cols = M2.shape
    pivot_col = []; pivots = set(); r = 0
    for c in range(cols):
        pr = None
        for i in range(r, rows):
            if M2[i, c] % p != 0: pr = i; break
        if pr is None: continue
        if pr != r: M2[[r, pr]] = M2[[pr, r]]
        inv = pow(int(M2[r, c]) % p, -1, p)
        M2[r] = (M2[r] * inv) % p
        for i in range(rows):
            if i != r and M2[i, c] % p != 0:
                M2[i] = (M2[i] - M2[i, c] * M2[r]) % p
        pivot_col.append(c); pivots.add(c); r += 1
    free = [c for c in range(cols) if c not in pivots]
    K = np.zeros((cols, len(free)), dtype=int)
    for k, fc in enumerate(free):
        v = np.zeros(cols, dtype=int); v[fc] = 1
        for ri, pc in enumerate(pivot_col):
            coef = int(M2[ri, fc]) % p
            if coef: v[pc] = (-coef) % p
        K[:, k] = v
    return K


# Q(7,2) setup
q, d = 7, 2
poly, aps = find_primitive(q, d+1)
fs = q**(d+1) - 1
n = fs // (q-1)
traces = trace_table(q, d+1, aps)
tau_H = np.array([1 if traces[i % fs] == 0 else 0 for i in range(n)], dtype=int)
xi = 2 if q%2==1 else q+1
tau_Q = np.array([1 if traces[(xi*i) % fs] == 0 else 0 for i in range(n)], dtype=int)
A = circ(tau_H) % q
M_Q = circ(tau_Q) % q
H_x = (M_Q @ A) % q
H = np.hstack([A, H_x])
M_cent = np.hstack([H_x, (-A) % q]) % q
K_S = right_kernel_basis(H, q)

print(f"Q({q},{d}): n={n}, primitive_poly={poly}")

# Compute Frobenius order
def order(g, n):
    o = 1; cur = g % n
    while cur != 1:
        cur = (cur * g) % n; o += 1
    return o
ord_q = order(q, n)
print(f"ord_{{Z/{n}Z}}({q}) = {ord_q}")

# Singer-Frobenius orbit-canonical form:
# transformations are (shift_s, frob_k):  i -> q^k * (i + s)  mod n
# We want lex-min over all (s, k).
def canonical_form(S, n, q, ord_q):
    """Return canonical representative of S under Singer-Frobenius group."""
    best = None
    for k in range(ord_q):
        qk = pow(q, k, n)
        for s in range(n):
            transformed = tuple(sorted((qk * (i + s)) % n for i in S))
            if best is None or transformed < best:
                best = transformed
    return best


def is_canonical(S, n, q, ord_q):
    """Check if S equals its canonical form."""
    cf = canonical_form(S, n, q, ord_q)
    return tuple(sorted(S)) == cf


def search_weight_singer_frob(q, n, M_cent, K_S, w, time_budget):
    """Iterate over all C(n, w) subsets, but only process canonical ones."""
    n_total = comb(n, w)
    print(f"  [w={w}]  total {n_total:,} subsets, "
          f"expect ~{n_total // 171:,} canonical orbits")
    t0 = time.time()
    n_canonical = 0
    n_kernel_nonempty = 0

    for S in itertools.combinations(range(n), w):
        if not is_canonical(S, n, q, 3):
            continue
        n_canonical += 1
        cols = list(S) + [p + n for p in S]
        M_S = M_cent[:, cols]
        K_loc = right_kernel_basis(M_S, q)
        d_loc = K_loc.shape[1]
        if d_loc > 0:
            n_kernel_nonempty += 1
            # Enumerate kernel for non-stab logical
            for coefs in itertools.product(range(q), repeat=d_loc):
                if all(c==0 for c in coefs): continue
                v_loc = np.zeros(2*w, dtype=int)
                for k in range(d_loc):
                    if coefs[k]:
                        v_loc = (v_loc + coefs[k] * K_loc[:, k]) % q
                ab = np.zeros(2*n, dtype=int)
                for k_idx, pos in enumerate(S):
                    ab[pos] = v_loc[k_idx]
                    ab[pos + n] = v_loc[k_idx + w]
                a_part = ab[:n]; b_part = ab[n:]
                pw = int(((a_part!=0)|(b_part!=0)).sum())
                if pw != w: continue
                stab = (ab @ K_S) % q
                if not (stab==0).all():
                    elapsed = time.time() - t0
                    supp = [i for i in range(n) if a_part[i] or b_part[i]]
                    print(f"  *** weight-{w} witness in {elapsed:.1f}s")
                    print(f"     support: {supp}")
                    print(f"     a (Z): {a_part[supp]}")
                    print(f"     b (X): {b_part[supp]}")
                    return w, ab

        if n_canonical % 5000 == 0:
            elapsed = time.time() - t0
            if elapsed > time_budget:
                print(f"  [w={w}]  TIME BUDGET EXPIRED ({elapsed:.0f}s) at "
                      f"{n_canonical} canonical, {n_kernel_nonempty} kernel nonempty")
                return None, None

    elapsed = time.time() - t0
    print(f"  [w={w}]  no witness in {elapsed:.0f}s "
          f"({n_canonical} canonical orbits, {n_kernel_nonempty} kernel nonempty)")
    return None, None


# Search w = 7 and w = 8 with much longer budgets
for w, budget in [(7, 4000), (8, 16000)]:    # 1h + 4.5h
    print(f"\n--- Q(7,2) Singer-Frobenius search weight {w} (budget {budget}s) ---")
    result, ab = search_weight_singer_frob(q, n, M_cent, K_S, w, time_budget=budget)
    if result is not None:
        print(f"\n  >>> d_min(Q(7,2)) <= {result}")
        break
