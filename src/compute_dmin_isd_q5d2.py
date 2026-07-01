"""
Smart d_min search for Q(5,2) via support-restriction (information-set-style).

Naive brute force at weight 4: 10.4B candidates -> 5h.

Smart approach: for each w-subset S of n qudit positions, build the
restriction of the centralizer-defining matrix M_cent to the 2|S|
columns indexed by S in (a,b)-representation.  Solutions to the
centralizer constraint with support in S form
   ker(M_cent_S)  in  F_q^{2|S|}.
For random S, this kernel is empty (rank deficient is rare).  When
non-empty, enumerate the (small) kernel and check non-stabilizer +
Pauli weight = |S|.

Complexity per subset: O((n-k) * |S|^2) for kernel + O(q^|kernel|) for
enumeration.  Total: C(n, w) * O(...).

For (q, d) = (5, 2), n = 31:
  w = 4:  C(31, 4) = 31465 subsets;  per-subset work tiny if kernel empty.
  w = 5:  C(31, 5) = 169911;  still fast.
  w = 6:  C(31, 6) = 736281;  fast.

Search up to weight 6 (line-indicator-like upper bound q+1 = 6).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools
import time
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
    cur = one
    seen = {one: 0}
    aps = [one]
    for kk in range(1, fs):
        cur = mul_x(cur)
        if cur == one or cur in seen:
            return None
        seen[cur] = kk
        aps.append(cur)
    if mul_x(cur) != one:
        return None
    return aps


def find_primitive(p, m):
    for coeffs in itertools.product(range(p), repeat=m):
        if coeffs[0] == 0:
            continue
        poly = list(coeffs) + [1]
        ap = build_field(p, m, poly)
        if ap is not None:
            return poly, ap
    return None, None


def trace_table(p, m, aps):
    fs = p**m - 1
    out = []
    for i in range(fs):
        s = [0] * m
        for k in range(m):
            ap = aps[(i * p**k) % fs]
            for r in range(m):
                s[r] = (s[r] + ap[r]) % p
        out.append(s[0])
    return out


def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M


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


def right_kernel_basis(M, p):
    """Return basis of right-kernel of M as columns of a matrix."""
    M2 = (M.copy().astype(int) % p)
    rows, cols = M2.shape
    pivot_col = []
    pivots = set()
    r = 0
    for c in range(cols):
        pivot_row = None
        for i in range(r, rows):
            if M2[i, c] % p != 0:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            M2[[r, pivot_row]] = M2[[pivot_row, r]]
        inv = pow(int(M2[r, c]) % p, -1, p)
        M2[r] = (M2[r] * inv) % p
        for i in range(rows):
            if i != r and M2[i, c] % p != 0:
                M2[i] = (M2[i] - M2[i, c] * M2[r]) % p
        pivot_col.append(c)
        pivots.add(c)
        r += 1
    free_cols = [c for c in range(cols) if c not in pivots]
    K = np.zeros((cols, len(free_cols)), dtype=int)
    for k, fc in enumerate(free_cols):
        v = np.zeros(cols, dtype=int)
        v[fc] = 1
        for ri, pc in enumerate(pivot_col):
            coef = int(M2[ri, fc]) % p
            if coef:
                v[pc] = (-coef) % p
        K[:, k] = v
    return K


def build_Q(q, d, primitive_poly):
    m = d + 1
    aps = build_field(q, m, primitive_poly)
    fs = q**m - 1
    n = fs // (q - 1)
    traces = trace_table(q, m, aps)
    tau_H = np.array([1 if traces[i % fs] == 0 else 0 for i in range(n)], dtype=int)
    xi = 2 if q % 2 == 1 else q + 1
    tau_Q = np.array([1 if traces[(xi * i) % fs] == 0 else 0 for i in range(n)], dtype=int)
    A = circ(tau_H) % q
    M_Q = circ(tau_Q) % q
    H_z = A
    H_x = (M_Q @ A) % q
    H = np.hstack([H_z, H_x])
    M_cent = np.hstack([H_x, (-H_z) % q]) % q
    K_S = right_kernel_basis(H, q)
    return n, H, M_cent, K_S


def isd_search_weight_w(q, n, H, M_cent, K_S, w):
    """Find any Pauli of weight exactly w in centralizer \\ stabilizer."""
    n_centralizer_hits = 0
    n_subsets_processed = 0
    n_subsets_total = comb(n, w)
    t0 = time.time()
    print(f"  [w={w}]  searching {n_subsets_total:,} subsets")

    for positions in itertools.combinations(range(n), w):
        positions = list(positions)
        # Restrict M_cent to columns indexed by positions and positions+n:
        cols = positions + [p + n for p in positions]   # (a then b coords)
        M_cent_S = M_cent[:, cols]   # (n-k) x 2w

        # Find right-kernel of M_cent_S in F_q^{2w}
        K_local = right_kernel_basis(M_cent_S, q)   # 2w x dim_kernel

        n_subsets_processed += 1
        if K_local.shape[1] == 0:
            continue   # no centralizer member with support in this subset

        # Enumerate non-zero elements of the kernel:
        d_loc = K_local.shape[1]
        # We can iterate over q^d_loc vectors, but most will have full
        # support (Pauli weight = w).  We need them to NOT be in the
        # stabilizer, so check K_S.
        for coefs in itertools.product(range(q), repeat=d_loc):
            if all(c == 0 for c in coefs):
                continue
            v_local = np.zeros(2*w, dtype=int)
            for k in range(d_loc):
                if coefs[k]:
                    v_local = (v_local + coefs[k] * K_local[:, k]) % q
            # Embed back into F_q^{2n}:
            ab = np.zeros(2*n, dtype=int)
            for k_idx, pos in enumerate(positions):
                ab[pos] = v_local[k_idx]                # a coord
                ab[pos + n] = v_local[k_idx + w]        # b coord

            # Pauli weight: # positions where (a_i, b_i) != (0,0)
            a_part = ab[:n]
            b_part = ab[n:]
            pw = int(((a_part != 0) | (b_part != 0)).sum())
            if pw != w:
                continue   # only count weight-exactly-w hits

            n_centralizer_hits += 1

            # Stabilizer membership test
            stab_check = (ab @ K_S) % q
            if not (stab_check == 0).all():
                # Found a non-stabilizer Pauli of weight w!
                t = time.time() - t0
                print(f"  *** weight-{w} non-stabilizer found in {t:.1f}s")
                supp = [i for i in range(n) if a_part[i] or b_part[i]]
                print(f"     support: {supp}")
                print(f"     a (Z): {a_part[supp]}")
                print(f"     b (X): {b_part[supp]}")
                return w, ab

    t = time.time() - t0
    print(f"  [w={w}]  no weight-{w} non-stabilizer logical found in {t:.1f}s")
    print(f"     ({n_centralizer_hits} centralizer hits at this exact weight, "
          f"all in stabilizer)")
    return None, None


def search_dmin(q, d, primitive_poly, max_weight=8):
    n, H, M_cent, K_S = build_Q(q, d, primitive_poly)
    rH = rank_mod_p(H, q)
    print(f"\n=== Q({q},{d}): n={n}, rank(H)={rH}, k={n-rH} ===")
    for w in range(1, max_weight + 1):
        print(f"\n  searching weight {w} ...")
        result, ab = isd_search_weight_w(q, n, H, M_cent, K_S, w)
        if result is not None:
            print(f"\n  >>> d_min(Q({q},{d})) <= {result} (witness found)")
            # Sanity check: confirm not weight < w (we already searched lower)
            return result
    print(f"\n  >>> No logical of weight <= {max_weight}, d_min > {max_weight}")
    return None


# ============== Q(5,2): n=31, k=15 ============================
poly5, _ = find_primitive(5, 3)
print(f"primitive poly F_5^3: {poly5}")
search_dmin(5, 2, poly5, max_weight=6)

# ============== Q(7,2): n=57, k=28 ============================
# C(57, 6) = 36M subsets; per-subset work small; total ~few minutes
print("\n" + "="*60)
poly7, _ = find_primitive(7, 3)
print(f"primitive poly F_7^3: {poly7}")
search_dmin(7, 2, poly7, max_weight=6)
