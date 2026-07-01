"""
Construct an explicit Clifford gate U_M whose conjugation maps the
trivial Z-only CSS stabiliser H_0 = (A | 0) of the LDS-spreading
construction to the non-CSS stabiliser H = (A | M_Q A) of Q(q, 2).

Recipe: a Clifford gate U_M corresponds to the symplectic matrix
    S_M = [ I   M ;
            0   I ]
acting on Pauli vectors (z, x) in F_q^{2n} by (z, x) -> (z, z M + x).
Symplecticity over F_q: requires M = M^T.

Conjugation maps stabiliser generator (z, 0) to (z, z M).  For the
i-th row a_i of A (which has shape n x 1 here viewed as F_q^n):
   (a_i, 0)  ->  (a_i, a_i M)
which equals the i-th row of (A | A M).  We want this to equal
(A | M_Q A), i.e. A M = M_Q A.

KEY LEMMA: such a symmetric M exists iff in every reciprocal CRT pair
(f, f^*) of X^n - 1, NOT BOTH have tau_H,f != 0.  By
Theorem 6.tau-zero-count, exactly ONE of every pair has tau_H,f = 0,
so the obstruction NEVER arises and M can always be chosen.

The construction: solve A M = M_Q A by CRT decomposition.

  - On CRT factor f where tau_H,f != 0: m_f = tau_Q,f (forced).
  - On CRT factor f where tau_H,f = 0: m_f is free; choose so that the
    palindromic constraint m(X) = m(X^{-1}) is satisfied (which forces
    m_f to be the "reciprocal image" of m_{f*}).

Verification at (q, d) = (3, 2):
  1. Compute A, M_Q over F_3 with n=13.
  2. Compute m(X) by CRT.
  3. Construct M = circ(m).
  4. Verify M = M^T.
  5. Verify A M = M_Q A.
  6. Verify (A | A M) = (A | M_Q A) (i.e. the Clifford U_M maps the
     trivial Z-only CSS code to Q(3, 2) ).
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


def crt_orbits(n, p):
    visited = [False]*n
    orbits = []
    for i in range(n):
        if visited[i]: continue
        orbit = []
        cur = i
        while not visited[cur]:
            visited[cur] = True
            orbit.append(cur)
            cur = (cur*p) % n
        orbits.append(tuple(orbit))
    return orbits


def construct_M(q, d):
    """Construct symmetric M satisfying A M = M_Q A over F_q[X]/(X^n - 1)."""
    p = q
    poly, aps = find_primitive(q, d+1)
    fs = q**(d+1) - 1
    n = fs // (q-1)
    traces = trace_table(q, d+1, aps)

    tau_H = np.array([1 if traces[i % fs] == 0 else 0 for i in range(n)], dtype=int)
    tau_Q = np.array([1 if traces[(2*i) % fs] == 0 else 0 for i in range(n)], dtype=int)

    A = circ(tau_H) % q
    M_Q = circ(tau_Q) % q

    # We seek m in F_p[X]/(X^n - 1) such that
    #   tau_H * m  =  tau_Q * tau_H   in F_p[X]/(X^n - 1)
    # and m is palindromic (m_i = m_{-i mod n}).
    #
    # Polynomial-multiplication shortcut:  m * tau_H  =  tau_Q * tau_H.
    # Pick m so that on every CRT factor f,  m_f * tau_H,f = tau_Q,f * tau_H,f.
    # Where tau_H,f != 0, we are forced to m_f = tau_Q,f.  Where tau_H,f = 0,
    # m_f is free.  We choose the free m_f to make m palindromic.
    #
    # Computational shortcut: just attempt
    #   m_naive = tau_Q  (everywhere).
    # That trivially satisfies tau_H * m = tau_Q * tau_H, since the cyclic
    # algebra is commutative.  CHECK PALINDROMICITY of tau_Q.  If tau_Q is not
    # palindromic, we add a correction d supported on the zero-set of tau_H
    # to symmetrise, exploiting the freedom on those factors.

    # Step 1: try m = tau_Q.
    m_naive = tau_Q.copy()
    is_pal = all(m_naive[i] == m_naive[(-i) % n] for i in range(n))
    print(f"  tau_Q palindromic ?  {is_pal}")
    if is_pal:
        return tau_H, tau_Q, A, M_Q, m_naive

    # Step 2: tau_Q not palindromic; correct by adding a polynomial d
    # supported on the zero-set of tau_H, chosen so that m = tau_Q + d is
    # palindromic.
    #
    # Concretely: define
    #   m_target_i = (tau_Q_i + tau_Q_{-i mod n}) / 2   over F_p
    # but ONLY at indices i where the symmetrisation is "free" (i.e. i is
    # in a zero-CRT-factor support).  At indices where tau_H is forced, we
    # cannot change m_i.
    #
    # Easier approach: solve linearly.  Let z(X) be a polynomial supported
    # on the zero-orbits of tau_H (i.e. z is in the cyclic code of factors
    # where tau_H_f = 0).  We need z palindromic and tau_Q + z palindromic.
    #
    # Equivalently: z_i + tau_Q_i = z_{-i} + tau_Q_{-i}, i.e.
    # z_i - z_{-i} = tau_Q_{-i} - tau_Q_i, for all i.
    #
    # Restricting z to live in the cyclic-code generated by zero-CRT
    # factors (a subspace W of F_p^n), we look for z in W satisfying the
    # palindrome difference.
    #
    # Use linear algebra over F_p.

    # Compute zero CRT factors of tau_H over F_p[X]/(X^n - 1).
    orbits = crt_orbits(n, p)
    # For each orbit, evaluate tau_H at omega^k where omega is a primitive
    # nth root of unity in F_{p^|orbit|}.  Use the same construction as
    # verify_bch_count.py.

    # Build a primitive root in F_{p^L} where L = lcm of orbit sizes.
    from math import gcd
    L = 1
    for orb in orbits:
        L = L * len(orb) // gcd(L, len(orb))
    if L == 1:
        return None  # trivial

    poly_K, aps_K = find_primitive(p, L)
    fsK = p**L - 1
    while fsK % n != 0:
        L += 1
        poly_K, aps_K = find_primitive(p, L)
        fsK = p**L - 1
    omega_exp = fsK // n

    # Determine which orbits have tau_H_f = 0
    zero_orbits = []
    for orb in orbits:
        k = orb[0]
        s = [0]*L
        for i in range(n):
            if tau_H[i] == 0: continue
            j = (i*k*omega_exp) % fsK
            for r in range(L):
                s[r] = (s[r] + aps_K[j][r]) % p
        if all(x == 0 for x in s):
            zero_orbits.append(orb)
    print(f"  Zero-orbits of tau_H: {zero_orbits}")
    print(f"  Total |F| = {len(orbits)}, # zero = {len(zero_orbits)} (= (|F|-1)/2 = {(len(orbits)-1)//2})")

    # Build the cyclic code W = polynomials supported only on zero-orbits.
    # An element w in W has w_g = 0 for g not in zero_orbits.  Equivalently,
    # as F_p^n vector, w is in the row-span of polynomials f(X) where f is
    # the product of factors in zero_orbits.
    #
    # Easier: W = {w in F_p^n : tau_H * w = 0 in F_p[X]/(X^n-1)}, by CRT.
    # Verify: w in W iff w_f = 0 for all f with tau_H_f != 0, i.e. w is
    # supported only on zero-orbits.  But we want "tau_H_f = 0 OR w_f = 0",
    # which is tau_H * w = 0.

    # So W = ker of multiplication-by-tau_H map on F_p^n.
    A_mat = circ(tau_H) % p
    # tau_H * w = 0 means A_mat @ w = 0 (as polynomial multiplication).
    # Wait, multiplication by tau_H IS the matrix A acting on coefficient
    # vectors. So W = ker(A) as left null space.

    # Compute null space of A
    def null_space(M, p):
        Mc = M.copy() % p
        rows, cols = Mc.shape
        pivots = []; r = 0
        for c in range(cols):
            pr = None
            for i in range(r, rows):
                if Mc[i, c] % p != 0: pr = i; break
            if pr is None: continue
            if pr != r: Mc[[r, pr]] = Mc[[pr, r]]
            inv = pow(int(Mc[r, c]), -1, p)
            Mc[r] = (Mc[r] * inv) % p
            for i in range(rows):
                if i != r and Mc[i, c] % p != 0:
                    Mc[i] = (Mc[i] - Mc[i, c] * Mc[r]) % p
            pivots.append(c); r += 1
        free = [c for c in range(cols) if c not in pivots]
        K = np.zeros((cols, len(free)), dtype=int)
        for k, fc in enumerate(free):
            v = np.zeros(cols, dtype=int); v[fc] = 1
            for ri, pc in enumerate(pivots):
                coef = int(Mc[ri, fc]) % p
                if coef: v[pc] = (-coef) % p
            K[:, k] = v
        return K

    W = null_space(A_mat, p)
    print(f"  dim(W) = {W.shape[1]} (expected = sum of zero-orbit sizes = {sum(len(o) for o in zero_orbits)})")

    # Now solve: find z in W (column space of W) such that
    #   (tau_Q + z) is palindromic, i.e.
    #   (tau_Q_i + z_i) = (tau_Q_{-i mod n} + z_{-i mod n})   for all i.
    # Equivalently: z_i - z_{-i} = tau_Q_{-i} - tau_Q_i.

    # Build the "antisymmetrise" operator: anti(z)_i = z_i - z_{-i mod n}.
    # We want anti(z) = anti(tau_Q^{-})... hmm let's just do it by linear algebra.

    # Build target: t_i = tau_Q_{-i mod n} - tau_Q_i.
    t = np.array([(int(tau_Q[(-i) % n]) - int(tau_Q[i])) % p for i in range(n)], dtype=int)

    # Solve for c (coefficients in W) such that  W @ c  has  anti(W @ c) = t.
    # Equivalently: anti_op @ (W @ c) = t.  anti_op is n x n matrix.
    anti_op = np.zeros((n, n), dtype=int)
    for i in range(n):
        anti_op[i, i] += 1
        anti_op[i, (-i) % n] -= 1
    anti_op = anti_op % p

    # Compose: M_full = anti_op @ W   (n x dim(W))
    M_full = (anti_op @ W) % p

    # Solve M_full @ c = t over F_p.  Use Gaussian elimination on augmented matrix.
    aug = np.hstack([M_full, t.reshape(-1, 1)]) % p
    rows_aug, cols_aug = aug.shape
    cols_main = M_full.shape[1]
    pivots_aug = []; r = 0
    for c in range(cols_main):
        pr = None
        for i in range(r, rows_aug):
            if aug[i, c] % p != 0: pr = i; break
        if pr is None: continue
        if pr != r: aug[[r, pr]] = aug[[pr, r]]
        inv = pow(int(aug[r, c]), -1, p)
        aug[r] = (aug[r] * inv) % p
        for i in range(rows_aug):
            if i != r and aug[i, c] % p != 0:
                aug[i] = (aug[i] - aug[i, c] * aug[r]) % p
        pivots_aug.append(c); r += 1

    # Check consistency: for any non-pivot row, last column must be 0
    consistent = True
    for i in range(r, rows_aug):
        if aug[i, -1] % p != 0:
            consistent = False
            break
    print(f"  Linear system consistent? {consistent}")
    if not consistent:
        print(f"  ERROR: cannot symmetrise.  Construction fails at (q,d) = ({q},{d}).")
        return None

    # Extract solution c
    c_sol = np.zeros(cols_main, dtype=int)
    for ri, pc in enumerate(pivots_aug):
        c_sol[pc] = int(aug[ri, -1]) % p

    z = (W @ c_sol) % p
    m = (tau_Q + z) % p

    # Verify palindromic
    is_pal_2 = all(m[i] == m[(-i) % n] for i in range(n))
    print(f"  m = tau_Q + z palindromic? {is_pal_2}")

    # Verify A m = M_Q A as circulants.  Equivalently tau_H * m = tau_Q * tau_H
    # in F_p[X]/(X^n-1).
    M_circ = circ(m) % p
    LHS = (A @ M_circ) % p
    RHS = (M_Q @ A) % p
    print(f"  A M = M_Q A?  {np.array_equal(LHS, RHS)}")

    # Verify M = M^T
    print(f"  M symmetric? {np.array_equal(M_circ, M_circ.T)}")

    return tau_H, tau_Q, A, M_Q, m


# --- Main ---
for (q, d) in [(3, 2), (5, 2), (7, 2)]:
    print(f"\n=== Constructing Clifford for Q({q}, {d}) ===")
    res = construct_M(q, d)
    if res is None:
        continue
    tau_H, tau_Q, A, M_Q, m = res
    print(f"  m = {m.tolist()}")
    n = len(m)
    M_circ = circ(m) % q
    H = np.hstack([A, (M_Q @ A) % q])
    H_via_M = np.hstack([A, (A @ M_circ) % q])
    print(f"  H = (A | M_Q A) and (A | A M) coincide? {np.array_equal(H, H_via_M)}")
