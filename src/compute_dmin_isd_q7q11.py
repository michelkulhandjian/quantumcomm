"""
ISD search for Q(7,2) and Q(11,2) with Singer-cyclic-symmetry pruning.

By Singer cyclic symmetry, witnesses come in cyclic orbits.  We only
iterate over subsets whose minimum element is 0, capturing one
representative of each orbit (most orbits have size n, giving ~n/w
speedup).

Time budgets:
  Q(7,2)  n=57:  weight 6 -> C(56,5) = 3.8M, weight 7 -> C(56,6) = 32M.
  Q(11,2) n=133: weight 4 -> C(132,3) = 374K, weight 5 -> C(132,4) = 12M.

Goal:
  - Q(7,2): pin down d_min in [6, 8] via weight 6 and 7 search.
  - Q(11,2): verify d_min >= 5 via weights 1..4 exhaustive.
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

def rank_mod_p(M, p):
    M = (M.copy() % p).astype(int)
    rows, cols = M.shape; r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] % p != 0: pivot = i; break
        if pivot is None: continue
        if pivot != r: M[[r, pivot]] = M[[pivot, r]]
        inv = pow(int(M[r, c]) % p, -1, p)
        M[r] = (M[r] * inv) % p
        for i in range(rows):
            if i != r and M[i, c] % p != 0:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
        if r == rows: break
    return r


def build_Q(q, d):
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
    return n, H, M_cent, K_S, poly


def isd_search_canonical(q, n, M_cent, K_S, w, time_budget=None):
    """Search subsets {0, s2, s3, ..., sw} with 0 < s2 < ... < sw < n
    (i.e., subsets where 0 is the smallest element).  By cyclic symmetry,
    each Singer-orbit of weight-w subsets is represented at least once.
    """
    n_subsets = comb(n-1, w-1)
    print(f"  [w={w}]  searching {n_subsets:,} canonical subsets (smallest-elem-0 form)")
    t0 = time.time()
    n_kernel_nonempty = 0
    n_processed = 0

    for tail in itertools.combinations(range(1, n), w-1):
        positions = (0,) + tail
        cols = list(positions) + [p+n for p in positions]
        M_S = M_cent[:, cols]
        K_loc = right_kernel_basis(M_S, q)
        d_loc = K_loc.shape[1]
        n_processed += 1

        if d_loc > 0:
            n_kernel_nonempty += 1
            for coefs in itertools.product(range(q), repeat=d_loc):
                if all(c==0 for c in coefs): continue
                v_loc = np.zeros(2*w, dtype=int)
                for k in range(d_loc):
                    if coefs[k]:
                        v_loc = (v_loc + coefs[k] * K_loc[:, k]) % q
                ab = np.zeros(2*n, dtype=int)
                for k_idx, pos in enumerate(positions):
                    ab[pos] = v_loc[k_idx]
                    ab[pos+n] = v_loc[k_idx+w]
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

        # Time budget check
        if time_budget and (n_processed % 5000 == 0):
            elapsed = time.time() - t0
            if elapsed > time_budget:
                pct = 100 * n_processed / n_subsets
                print(f"  [w={w}]  TIME BUDGET EXPIRED ({elapsed:.0f}s) at "
                      f"{pct:.1f}%, {n_kernel_nonempty} kernel nonempty so far")
                return None, None

    elapsed = time.time() - t0
    print(f"  [w={w}]  no witness in {elapsed:.0f}s "
          f"({n_kernel_nonempty} kernel nonempty)")
    return None, None


def search_dmin(q, d, weights, time_per_w):
    n, H, M_cent, K_S, poly = build_Q(q, d)
    rH = rank_mod_p(H, q)
    print(f"\n=== Q({q},{d}): n={n}, primitive_poly={poly}, "
          f"rank(H)={rH}, k={n-rH} ===")
    for w in weights:
        result, ab = isd_search_canonical(q, n, M_cent, K_S, w, time_per_w)
        if result is not None:
            print(f"  >>> d_min(Q({q},{d})) <= {result}")
            return result
    return None


# Q(7, 2): search w = 6, 7 with 30 min budget each
print("="*60)
print("Q(7, 2)")
print("="*60)
search_dmin(7, 2, [6, 7], time_per_w=1800)

# Q(11, 2): search w = 1, 2, 3, 4 with 10 min budget each
print("\n" + "="*60)
print("Q(11, 2)")
print("="*60)
search_dmin(11, 2, [1, 2, 3, 4], time_per_w=600)
