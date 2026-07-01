"""
Compute d_min for Q(q, d) instances where q is EVEN or d is EVEN
(beyond the flagship d=2 family).

The 2-torsion ceiling (Corollary cor:dmin-le-2-proven) does not apply in
these regimes, so non-trivial distances are possible.

Targets:
   Q(2, 2) = [[7, ?, ?]]_2     - qubit code on Fano plane PG(2,2),
                                 xi = q+1 = 3 in formal Q.
   Q(2, 3) = [[15, ?, ?]]_2    - qubit code on PG(3,2).
   Q(2, 4) = [[31, ?, ?]]_2    - qubit code on PG(4,2).
   Q(4, 2) = [[21, ?, ?]]_4    - F_4 qudit code on PG(2,4).

For each: build A, M_Q, H, compute rank(H) (giving k), and exhaustive
ISD-style search for d_min via support-restriction up to a budget.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools
import numpy as np


def build_field(p, m, poly):
    fs = p**m - 1
    pp = poly[:-1]
    def mul_x(a):
        new = [0]*m
        new[0] = (-pp[0]*a[m-1]) % p
        for k in range(1, m):
            new[k] = (a[k-1] - pp[k]*a[m-1]) % p
        return tuple(new)
    one = tuple(1 if k==0 else 0 for k in range(m))
    cur=one; seen={one:0}; aps=[one]
    for kk in range(1, fs):
        cur = mul_x(cur)
        if cur==one or cur in seen: return None
        seen[cur]=kk; aps.append(cur)
    if mul_x(cur) != one: return None
    return aps

def find_primitive(p, m):
    for c in itertools.product(range(p), repeat=m):
        if c[0]==0: continue
        poly = list(c)+[1]
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
            for r in range(m): s[r] = (s[r]+ap[r]) % p
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


def f_rank(M, p):
    """F_p rank of integer matrix."""
    M2 = (M.copy().astype(int) % p)
    rows, cols = M2.shape
    r = 0
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
        r += 1
    return r


def build_QqdH(q, d):
    """Build A, M_Q, H = (A | M_Q A) for Q(q, d).  Works for q prime power."""
    # We need a primitive element of F_{q^{d+1}}.  q must be a prime power.
    if q == 2:
        p, t = 2, 1
    elif q == 3:
        p, t = 3, 1
    elif q == 4:
        p, t = 2, 2
    elif q == 5:
        p, t = 5, 1
    elif q == 7:
        p, t = 7, 1
    elif q == 8:
        p, t = 2, 3
    elif q == 9:
        p, t = 3, 2
    else:
        raise NotImplementedError(f"q = {q} not supported")

    m_total = t * (d + 1)
    poly, aps = find_primitive(p, m_total)
    if poly is None:
        raise RuntimeError(f"No primitive poly for F_{p^{m_total}}")

    fs_total = p**m_total - 1
    n_q = q**(d + 1) - 1
    n = (q**(d+1) - 1) // (q - 1)

    # Trace from F_{p^{m_total}} = F_{q^{d+1}} -> F_p directly is the
    # F_p trace; we want trace to F_q, which is partial.  Easier: project
    # K^*/F^* with index step (q-1).  In F_{p^{t(d+1)}}, F_q = F_{p^t}
    # is subfield of index t*(d+1)/t = d+1.
    #
    # The F_q trace is Tr_{K/F_q}(z) = z + z^q + z^{q^2} + ... + z^{q^d}.
    # In our table we have aps indexed by F_p exponents, fs_total = p^{t(d+1)} - 1.
    # alpha^i in K corresponds to alpha^i in F_{p^{m_total}}; alpha^{i*q} = alpha^{i*p^t}.
    # We compute Tr_{K/F_q}(alpha^i) = sum_{k=0}^{d} alpha^{i*p^{tk}}.
    # The result is in F_q = F_{p^t}, which we represent as elements of F_{p^t}.
    # An element is in F_q iff its position in aps satisfies certain constraints.

    # Simpler: just compute the F_p trace via trace_table, then mod out.
    # Actually we need F_q trace.  Let's compute Tr_{K/F_q} directly:
    #
    #   Tr_q(alpha^i) = sum_{k=0}^d alpha^{i*q^k}
    #                 = sum_{k=0}^d alpha^{i*p^{tk}}
    # which is some element of F_q ⊂ K.

    def Tr_q(i):
        """F_q-trace of alpha^i, returned as a tuple in F_p^{m_total}."""
        s = [0]*m_total
        for k in range(d+1):
            j = (i * pow(p, t*k, fs_total)) % fs_total
            ap = aps[j]
            for r in range(m_total):
                s[r] = (s[r] + ap[r]) % p
        return tuple(s)

    # Compute trace at each i = 0, ..., n_q - 1, and check if it equals 0
    # (additive identity in F_p^{m_total}).
    zero = tuple(0 for _ in range(m_total))
    # tau_H(i) = 1 if Tr_q(alpha^i) = 0
    tau_H_full = [1 if Tr_q(i) == zero else 0 for i in range(n_q)]

    # Project to K^*/F^*: the F_q-coset of alpha^i is determined by i mod n.
    # Indeed, alpha^j and alpha^{j + n*(q-1)k} = alpha^j * (alpha^{n(q-1)})^k differ
    # by a power of alpha^{(q^{d+1}-1)/(q-1) * (q-1)} = alpha^{q^{d+1}-1} = 1.  Hmm.
    # Actually: K* has order q^{d+1} - 1 = n*(q-1).  F* has order q-1.  Quotient has
    # order n.  So alpha^i and alpha^j map to same K^*/F^* coset iff i ≡ j (mod n).
    # Actually alpha^i / alpha^j = alpha^{i-j} ∈ F* iff alpha^{(i-j)} ∈ F_q iff
    # (i-j) is divisible by n_q / (q-1) = n.

    tau_H = np.array([tau_H_full[i] for i in range(n)], dtype=int)
    # Verify weight is theta_{d-1}(q) = (q^d - 1)/(q-1)
    expected_wt = (q**d - 1) // (q - 1)
    print(f"  weight(tau_H) = {int(tau_H.sum())}  (expected theta_{d-1}({q}) = {expected_wt})")

    # Now M_Q with formal-quadric:  Q = {z : Tr_q(z^xi) = 0}/F^*, xi = q+1 if q even, 2 if q odd.
    xi = q + 1 if q % 2 == 0 else 2
    tau_Q_full = [1 if Tr_q((xi * i) % fs_total) == zero else 0 for i in range(n_q)]
    tau_Q = np.array([tau_Q_full[i] for i in range(n)], dtype=int)

    A = circ(tau_H) % q if q in (2, 3, 5, 7) else circ(tau_H) % p  # need F_q semantics; for q prime, q==p
    M_Q = circ(tau_Q) % q if q in (2, 3, 5, 7) else circ(tau_Q) % p

    # WARNING: for q non-prime (q=4, 8, 9), F_q != Z/qZ.  The matrix entries are
    # 0/1 from the indicator, so they live in F_q's prime subfield F_p naturally.
    # For now restrict to q prime to keep things simple.

    if q not in (2, 3, 5, 7, 11, 13):
        print(f"  WARNING: q = {q} is non-prime; using F_p={p} arithmetic, may not be F_q semantics.")

    A = A % p
    M_Q = M_Q % p
    H_x = (M_Q @ A) % p
    H = np.hstack([A, H_x])

    return tau_H, tau_Q, A, M_Q, H, p, n


def test_symplectic(H, p):
    """Check H_z H_x^T = H_x H_z^T (mod p)."""
    n = H.shape[1] // 2
    H_z = H[:, :n]
    H_x = H[:, n:]
    S = (H_z @ H_x.T - H_x @ H_z.T) % p
    return np.array_equal(S, np.zeros_like(S))


def search_dmin(H, p, max_w=None, time_budget=300):
    """Exhaustive ISD search for d_min."""
    import time
    n = H.shape[1] // 2
    H_z = H[:, :n]; H_x = H[:, n:]
    M_cent = np.hstack([H_x, (-H_z) % p]) % p
    K_S = right_kernel_basis(H, p)

    if max_w is None:
        max_w = n

    t0 = time.time()
    for w in range(1, max_w + 1):
        n_combos = 0
        n_kernel = 0
        for S in itertools.combinations(range(n), w):
            n_combos += 1
            cols = list(S) + [pp + n for pp in S]
            M_S = M_cent[:, cols]
            K_loc = right_kernel_basis(M_S, p)
            d_loc = K_loc.shape[1]
            if d_loc > 0:
                n_kernel += 1
                # Enumerate kernel for non-stab logical with full support
                for coefs in itertools.product(range(p), repeat=d_loc):
                    if all(c == 0 for c in coefs): continue
                    v_loc = np.zeros(2*w, dtype=int)
                    for k in range(d_loc):
                        if coefs[k]:
                            v_loc = (v_loc + coefs[k] * K_loc[:, k]) % p
                    ab = np.zeros(2*n, dtype=int)
                    for k_idx, pos in enumerate(S):
                        ab[pos] = v_loc[k_idx]
                        ab[pos + n] = v_loc[k_idx + w]
                    a_part = ab[:n]; b_part = ab[n:]
                    pw = int(((a_part != 0) | (b_part != 0)).sum())
                    if pw != w: continue
                    stab = (ab @ K_S) % p
                    if not (stab == 0).all():
                        elapsed = time.time() - t0
                        supp = [i for i in range(n) if a_part[i] or b_part[i]]
                        print(f"  *** weight-{w} witness in {elapsed:.2f}s, support={supp}")
                        print(f"      a={a_part[supp].tolist()}, b={b_part[supp].tolist()}")
                        return w
            if time.time() - t0 > time_budget:
                print(f"  TIME BUDGET EXPIRED at w={w}, processed {n_combos} of "
                      f"C({n},{w}) subsets, {n_kernel} kernel-nonempty.")
                return None
        elapsed = time.time() - t0
        print(f"  w={w}: {n_combos} subsets, {n_kernel} kernel-nonempty, "
              f"NO WITNESS, elapsed={elapsed:.1f}s")
    return None


# --- Main ---
for (q, d) in [(2, 2), (2, 3), (2, 4), (3, 4)]:
    print(f"\n{'='*60}")
    print(f"=== Q({q}, {d}) ===")
    print('='*60)
    n = (q**(d+1) - 1) // (q - 1)
    print(f"n = {n}")
    try:
        tau_H, tau_Q, A, M_Q, H, p, n_chk = build_QqdH(q, d)
    except Exception as e:
        print(f"  ERROR building H: {e}")
        continue

    rank_H = f_rank(H, p)
    rank_A = f_rank(A, p)
    k = n - rank_H
    print(f"  rank(A) = {rank_A}, rank(H) = {rank_H}, k = n - rank(H) = {k}")
    print(f"  symplectic-isotropic? {test_symplectic(H, p)}")

    n_parity = "EVEN" if n % 2 == 0 else "ODD"
    print(f"  n parity: {n_parity}  ({'might hit 2-torsion ceiling d_min<=2' if n%2==0 else 'no 2-torsion ceiling'})")

    # Choose budget based on n
    if n <= 7:
        budget = 60; max_w = n
    elif n <= 15:
        budget = 300; max_w = 8
    elif n <= 31:
        budget = 600; max_w = 5
    else:
        budget = 600; max_w = 4

    dmin = search_dmin(H, p, max_w=max_w, time_budget=budget)
    if dmin is not None:
        print(f"  >>> d_min(Q({q},{d})) = {dmin}")
    else:
        print(f"  >>> d_min(Q({q},{d})) >= some lower bound (search incomplete)")
