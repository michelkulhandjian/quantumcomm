"""
Compute  d_min  for the  V_1  family  Q(q, d)  at several (q, d).

Test the conjecture:
   If  n = theta_d(q) = (q^{d+1} - 1)/(q - 1)  is EVEN,
   then  d_min <= 2  with witness  Z_0 . Z_{n/2}^{q-1}
   (a Pauli supported on the 2-torsion of  K*/F*).

We test:
   (q, d) = (3, 2)   n = 13   ODD     => prediction d_min > 2
   (q, d) = (3, 3)   n = 40   EVEN    => prediction d_min <= 2  (verified)
   (q, d) = (5, 2)   n = 31   ODD     => prediction d_min > 2
"""
import itertools, time, sys
from math import factorial
import numpy as np

# Force unbuffered output:
sys.stdout.reconfigure(line_buffering=True)


# --------- Generic GF(p) primitive-element machinery ---------------
def build_field(p, m, poly_coeffs):
    """Build F_{p^m} = F_p[x] / <poly>, with poly_coeffs = [c_0, c_1, ..., c_m]
    representing  poly(x) = c_0 + c_1 x + ... + c_m x^m.
    Returns:   alpha_powers:  list of m-tuples for alpha^0, ..., alpha^{p^m - 1}
               trace[i]:      Tr(alpha^i) for i = 0, ..., p^m - 2  (in F_p)
    """
    n_field = p**m
    field_size = n_field - 1   # multiplicative order
    # require monic with poly[m] = 1  (caller ensures)
    assert poly_coeffs[-1] == 1
    poly = poly_coeffs[:-1]    # x^m = - poly[0] - poly[1] x - ... - poly[m-1] x^{m-1}
    # i.e. x^m = sum_k (- poly[k]) x^k

    def mul_x(a):
        # multiply polynomial a by x mod f, where a is m-tuple
        new = [0] * m
        new[0] = (-poly[0] * a[m-1]) % p
        for k in range(1, m):
            new[k] = (a[k-1] - poly[k] * a[m-1]) % p
        return tuple(new)

    one = tuple(1 if k == 0 else 0 for k in range(m))
    alpha_powers = [one]
    cur = one
    for _ in range(1, field_size):
        cur = mul_x(cur)
        alpha_powers.append(cur)
    # Sanity: alpha^(field_size) should be 1 again (loop closes)
    final = mul_x(cur)
    assert final == one, "polynomial not primitive"

    def trace(i):
        # Tr(alpha^i) = sum_{k=0}^{m-1} alpha^{i p^k}, projected to F_p
        s = [0] * m
        for k in range(m):
            ap = alpha_powers[(i * p**k) % field_size]
            for r in range(m):
                s[r] = (s[r] + ap[r]) % p
        # The result should be in F_p (only constant coordinate non-zero)
        # since Tr maps to F_p.
        return s[0]

    traces = [trace(i) for i in range(field_size)]
    return alpha_powers, traces


# --------- Generic helpers -----------------------------------------
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

def right_kernel(M, p):
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


# --------- Build the Q(q, d) data for given (q, d) --------------------
def build_Q_qd(q, d, primitive_poly):
    """Return  tau_H, tau_Q, A, M_Q, H, M_cent, K_S  for the Q(q,d) code."""
    # field is F_{q^{d+1}}  (since K = F_{q^{d+1}})
    m = d + 1
    alpha_powers, traces = build_field(q, m, primitive_poly)
    field_size = q**m - 1
    n = (q**m - 1) // (q - 1)

    tau_H = np.array([1 if traces[i % field_size] == 0 else 0
                      for i in range(n)], dtype=int)
    # Formal Q:  Tr(alpha^{xi * i}) = 0,  xi = 2 if q odd else q+1
    xi = 2 if (q % 2 == 1) else (q + 1)
    tau_Q = np.array([1 if traces[(xi * i) % field_size] == 0 else 0
                      for i in range(n)], dtype=int)

    A = circ(tau_H) % q
    M_Q = circ(tau_Q) % q
    H_z = A
    H_x = (M_Q @ A) % q
    H = np.hstack([H_z, H_x])
    M_cent = np.hstack([H_x, (-H_z) % q]) % q
    K_S = right_kernel(H, q)
    return {
        "n": n, "q": q, "d": d,
        "tau_H": tau_H, "tau_Q": tau_Q,
        "A": A, "M_Q": M_Q, "H_z": H_z, "H_x": H_x, "H": H,
        "M_cent": M_cent, "K_S": K_S,
    }


def search_dmin(data, weights=(1, 2, 3, 4), batch_size=100_000, verbose=True):
    n = data["n"]; q = data["q"]; d = data["d"]
    M_cent = data["M_cent"]; K_S = data["K_S"]
    if verbose:
        print(f"\n--- Searching d_min for Q({q},{d}), n={n} ---")
        rH = rank_mod_p(data["H"], q)
        print(f"    rank_F{q}(H) = {rH},  k = n - rank = {n - rH}")

    nz_pairs = [(a, b) for a in range(q) for b in range(q)
                if (a, b) != (0, 0)]
    AB_count = len(nz_pairs)

    found_pauli = None
    for w in weights:
        n_pos = 1
        for k in range(w):
            n_pos *= (n - k)
        n_pos //= factorial(w)
        n_total = n_pos * (AB_count ** w)
        if verbose:
            print(f"    [w={w}]  {n_pos} x {AB_count}^{w} = {n_total:,} candidates")

        t0 = time.time()
        a_buf = np.zeros((batch_size, n), dtype=np.int8)
        b_buf = np.zeros((batch_size, n), dtype=np.int8)
        idx = 0
        n_in_centralizer = 0

        for positions in itertools.combinations(range(n), w):
            pos_list = list(positions)
            for assignment in itertools.product(nz_pairs, repeat=w):
                a_buf[idx, :] = 0
                b_buf[idx, :] = 0
                for k in range(w):
                    pos = pos_list[k]
                    a_buf[idx, pos] = assignment[k][0]
                    b_buf[idx, pos] = assignment[k][1]
                idx += 1
                if idx == batch_size:
                    AB = np.hstack([a_buf[:idx].astype(int),
                                    b_buf[:idx].astype(int)])
                    cent = (AB @ M_cent.T) % q
                    in_cent = (cent == 0).all(axis=1)
                    n_in_centralizer += int(in_cent.sum())
                    if in_cent.any():
                        cands = AB[in_cent]
                        stab = (cands @ K_S) % q
                        in_stab = (stab == 0).all(axis=1)
                        non_triv = cands[~in_stab]
                        if non_triv.shape[0] > 0:
                            v = non_triv[0]
                            a, b = v[:n], v[n:]
                            actual = int(((a != 0) | (b != 0)).sum())
                            t = time.time() - t0
                            if verbose:
                                print(f"    *** d_min <= {actual}  in {t:.1f}s")
                                # Print the witness
                                supp = [i for i in range(n)
                                        if (a[i] != 0 or b[i] != 0)]
                                print(f"        support: {supp}")
                                print(f"        a (Z): {a[supp]}")
                                print(f"        b (X): {b[supp]}")
                            found_pauli = (actual, a, b, supp)
                            return found_pauli
                    idx = 0

        # Flush last partial batch
        if idx > 0:
            AB = np.hstack([a_buf[:idx].astype(int),
                            b_buf[:idx].astype(int)])
            cent = (AB @ M_cent.T) % q
            in_cent = (cent == 0).all(axis=1)
            n_in_centralizer += int(in_cent.sum())
            if in_cent.any():
                cands = AB[in_cent]
                stab = (cands @ K_S) % q
                in_stab = (stab == 0).all(axis=1)
                non_triv = cands[~in_stab]
                if non_triv.shape[0] > 0:
                    v = non_triv[0]
                    a, b = v[:n], v[n:]
                    actual = int(((a != 0) | (b != 0)).sum())
                    t = time.time() - t0
                    if verbose:
                        print(f"    *** d_min <= {actual}  in {t:.1f}s")
                        supp = [i for i in range(n) if (a[i] != 0 or b[i] != 0)]
                        print(f"        support: {supp}")
                        print(f"        a (Z): {a[supp]}")
                        print(f"        b (X): {b[supp]}")
                    return (actual, a, b, supp)
        t = time.time() - t0
        if verbose:
            print(f"    [w={w}]  no logical found.  "
                  f"{n_in_centralizer} in centralizer.  {t:.1f}s")
    return None


# ---- (q, d) = (3, 2):  n = 13, odd, no 2-torsion ----
# K = F_27 = F_3[x] / (x^3 + 2x + 1)  i.e., poly = [1, 2, 0, 1]
data32 = build_Q_qd(3, 2, [1, 2, 0, 1])
print(f"Q(3,2): n = {data32['n']}, tau_H weight = {int(data32['tau_H'].sum())}")
res32 = search_dmin(data32, weights=(1, 2, 3, 4), batch_size=50_000)

# ---- (q, d) = (3, 3):  n = 40, even, has 2-torsion at {0, 20} ----
# K = F_81 = F_3[x] / (x^4 + x + 2)  i.e., poly = [2, 1, 0, 0, 1]
data33 = build_Q_qd(3, 3, [2, 1, 0, 0, 1])
print(f"\nQ(3,3): n = {data33['n']}, tau_H weight = {int(data33['tau_H'].sum())}")
res33 = search_dmin(data33, weights=(1, 2, 3), batch_size=200_000)

# ---- (q, d) = (5, 2):  n = 31, odd, no 2-torsion ----
# K = F_125 = F_5[x] / (x^3 + 3x + 3)  -- need to verify this is primitive
# Actually let's pick x^3 + x + 1 over F_5? Not sure.  Let's try different.
# x^3 + 3x + 3 over F_5:  test irreducibility: f(0)=3, f(1)=2, f(2)=11=1, f(3)=33=3, f(4)=51=1.  No roots.
# Verifying primitivity is harder; just try and the build_field will catch.
try:
    data52 = build_Q_qd(5, 2, [3, 3, 0, 1])
    print(f"\nQ(5,2): n = {data52['n']}, tau_H weight = {int(data52['tau_H'].sum())}")
    res52 = search_dmin(data52, weights=(1, 2, 3, 4), batch_size=50_000)
except AssertionError as e:
    print(f"\n[skip Q(5,2)] {e}; would need a primitive polynomial of degree 3 over F_5.")
