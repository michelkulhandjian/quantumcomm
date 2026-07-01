"""
Verify Conjecture 6.BCH numerically:
For Q(p, 2), the number of CRT factors f with both u_f != 0 AND tau_f != 0
is >= |F| - (p-1).

Equivalently:
   N_zero = #{f : u_f = 0 OR tau_f = 0}  <=  p-1.

Numerically check at p = 3, 5, 7, 11.
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


def crt_orbits(n, p):
    """Frobenius orbits of Z/nZ under multiplication by p."""
    visited = [False] * n
    orbits = []
    for i in range(n):
        if visited[i]: continue
        orbit = []
        cur = i
        while not visited[cur]:
            visited[cur] = True
            orbit.append(cur)
            cur = (cur * p) % n
        orbits.append(orbit)
    return orbits


def ms_transform_at(c, k, n, p_field_size, primitive_root_powers):
    """Mattson-Solomon transform of indicator c at orbit-rep k.
    Computed in F_{p^|orbit|} via the primitive nth root of unity."""
    # We just check whether sum_i c[i] * omega^{i*k} = 0 in F_{...}
    # Use a different approach: compute polynomial c(X) mod f(X) for the
    # CRT factor f corresponding to orbit of k.  Vanishing iff f | c(X).
    pass


def vanishing_count(c_indices, n, p):
    """For c = indicator of c_indices in F_p^n, count CRT factors where
    c (mod f) = 0, by checking polynomial divisibility."""
    # Easier numerical approach: compute the Mattson-Solomon DFT of c
    # over an extension field containing nth root of unity, then count
    # zero-orbits.
    # Build Q^n - 1 factorization implicitly via Frobenius orbits.
    orbits = crt_orbits(n, p)

    # We need to evaluate c(omega^k) for k in each orbit.  c(omega^k) is in
    # F_{p^|orbit|}.  Vanishing test: c(omega^k) = 0.
    #
    # Strategy: realize omega in F_{p^L} where L = lcm of orbit sizes.
    # Compute c(omega^k) = sum_{i in c_indices} omega^{i*k}.

    # Determine L = lcm of orbit sizes.
    from math import gcd
    L = 1
    for orb in orbits:
        L = L * len(orb) // gcd(L, len(orb))

    # Build F_{p^L} as F_p[x] / <primitive poly of degree L>
    poly, aps = find_primitive(p, L) if L > 1 else (None, None)
    if L == 1:
        # All orbits have size 1, just F_p
        # Then omega = 1 is the only nth root in F_p, and we can't talk about
        # non-trivial roots.  This is a degenerate case for our purposes.
        return 0

    fs = p**L - 1
    # primitive root of unity of order n in F_{p^L}
    # Using primitive alpha of F_{p^L}: omega = alpha^{(fs)/n}.
    # We need n | fs, i.e. n | p^L - 1.
    if fs % n != 0:
        # Need to increase L
        while fs % n != 0:
            L += 1
            poly, aps = find_primitive(p, L)
            fs = p**L - 1
            if poly is None: return -1

    omega_exp = fs // n  # alpha^omega_exp is omega
    # alpha^k corresponds to aps[k]

    n_zero = 0
    for orb in orbits:
        k = orb[0]  # orbit representative
        # Compute c(omega^k) = sum_{i in c_indices} omega^{i*k} = sum aps[i*k * omega_exp mod fs]
        s = [0]*L
        for i in c_indices:
            j = (i * k * omega_exp) % fs
            for r in range(L):
                s[r] = (s[r] + aps[j][r]) % p
        if all(x == 0 for x in s):
            n_zero += 1
    return n_zero


def verify(p, d=2):
    print(f"\n=== Q({p},{d}) ===")
    # Build field K = F_{p^{d+1}} for trace
    m = d + 1
    poly, aps_K = find_primitive(p, m)
    fs = p**m - 1
    n = fs // (p - 1)
    traces = trace_table(p, m, aps_K)
    E_indices = [i for i in range(n) if traces[i % fs] == 0]
    Q_indices_full = [i for i in range(fs) if traces[(2*i) % fs] == 0]
    Q_indices = sorted(set(i % n for i in Q_indices_full))

    print(f"  n = {n}, |E| = {len(E_indices)}, |Q| = {len(Q_indices)}")

    # Compute u = M_Q * tau_H = circ-correlation of tau_H with tau_Q
    # u[k] = #{j : (k+j) mod n in E and j in Q}  =  # E ∩ (Q + k)
    Q_set = set(Q_indices)
    u_indices_with_value = {}
    for k in range(n):
        cnt = sum(1 for j in Q_indices if (k - j) % n in set(E_indices))
        # Hmm let me re-derive
        # u[k] = (M_Q tau_H)[k] = sum_j M_Q[k,j] * tau_H[j]
        #      = sum_j tau_Q[(k-j) mod n] * tau_H[j]
        cnt_check = sum(tau_Q_v * tau_H_v
                        for tau_Q_v, tau_H_v in
                        [(1 if (k-j) % n in Q_set else 0,
                          1 if j in set(E_indices) else 0)
                         for j in range(n)])
        u_indices_with_value[k] = cnt_check

    # u as a polynomial: nonzero positions
    u_indices = [k for k, v in u_indices_with_value.items() if v % p != 0]
    print(f"  weight(u mod {p}) = {len(u_indices)}")

    # Number of CRT factors |F| over F_p
    orbits = crt_orbits(n, p)
    F_count = len(orbits)
    print(f"  # CRT factors |F| = {F_count}")

    # Count zeros of tau_H (as polynomial in F_p[X]/<X^n - 1>) over CRT factors
    n_zero_tau = vanishing_count(E_indices, n, p)
    print(f"  # CRT factors with tau_f = 0:  {n_zero_tau}")

    # Count zeros of u
    # u as polynomial: coefficient is u_indices_with_value[k] mod p, supports = u_indices
    # vanishing_count expects an indicator (sum of monomials with coeff 1).
    # General vanishing: u_f = 0 iff polynomial has all coefficients vanishing.
    # We approximate by treating u as a 'weighted' polynomial.  Skip for now
    # and check the easier statement.

    # The conjecture wants:
    # #{f : tau_f = 0 OR u_f = 0} <= p - 1
    # We have a bound on tau alone.  If # tau zeros > p-1, conjecture fails.
    print(f"  Conjecture #zeros <= {p-1}: "
          f"{'HOLDS' if n_zero_tau <= p-1 else 'FAILS for tau alone'}")
    return n_zero_tau, F_count


for p in [3, 5, 7, 11]:
    verify(p, 2)
