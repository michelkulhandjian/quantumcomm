"""
Tighten  d_min  for the V_1 stabilizer code  Q(3,3) = [[40, 29, d]]_3.

A Pauli string is  (a, b) in F_3^40 x F_3^40,  weight = #{i : (a_i,b_i)!=(0,0)}.

  Centralizer C  = { (a,b) : a H_x^T - b H_z^T = 0 mod 3 }
                 = right-kernel of  M_cent_T = (H_x^T ; -H_z^T)  acting on (a,b)
  Stabilizer  S  = rowspan(H) over F_3,  with H = (H_z | H_x).

We seek min  w  such that there exists  (a,b)  of weight w  in  C \ S.
Lower bound from Prop. 6.3: d_min >= 3.
Upper bound (already computed): d_min <= 6.

Strategy:  exhaustively enumerate weight-w Pauli strings for w = 3, 4
(and abandon w = 5 unless needed -- 21B candidates).  Vectorise with numpy.
"""

import itertools
import time
import numpy as np

P = 3
N = 40

A = np.load("q3d3_A.npy").astype(int) % P
tau_Qo = np.load("q3d3_tau_Qo.npy")

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M

M_Q = circ(tau_Qo).astype(int) % P
H_z = A
H_x = (M_Q @ A) % P
H = np.hstack([H_z, H_x])    # 40 x 80, rank 11

# Centralizer condition: a H_x^T - b H_z^T = 0  i.e.  (a, b) . [H_x; -H_z]^T = 0
# Equivalently kernel of  M_cent = (H_x | -H_z)   acting on (a,b) row-wise.
M_cent = np.hstack([H_x, (-H_z) % P]) % P    # 40 x 80
# We will compute  ab @ M_cent.T  (shape (B, 40))  and check zero.

# --------- Stabilizer membership test  via right-kernel of H ----------
# Basis K of right-null-space of H over F_3; v in rowspan(H) iff v.K = 0.

def right_kernel_F3(M, p=P):
    """Basis of right-null-space of M over F_p, returned as a (n x (n-r)) array."""
    M2 = (M.copy().astype(int) % p)
    rows, cols = M2.shape
    pivot_col = []
    pivots = set()
    r = 0
    for c in range(cols):
        # find pivot in column c at or below row r
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
        if r == rows:
            break
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

K_S = right_kernel_F3(H, P)
print(f"Right-null-space of H has shape {K_S.shape}  (expect (80, 69))")
assert K_S.shape == (80, 80 - 11)

# Stabilizer membership: v in rowspan(H) iff v @ K_S == 0 mod 3.
def in_stabilizer_batch(V):
    return (V @ K_S % P).any(axis=1) == False  # True if all zero

# ---------- Generate weight-w Pauli strings as (a,b) in F_3^80 -----
PAULI_NONZERO_AB = [(a, b) for a in range(P) for b in range(P)
                    if (a, b) != (0, 0)]    # 8 entries
assert len(PAULI_NONZERO_AB) == 8

def search_weight(w, batch_size=200_000, verbose=True):
    """Return (found, weight, pauli_ab) where 'found' is True iff a
    weight-w Pauli is in C \\ S, and pauli_ab is one such (a,b)."""
    if verbose:
        n_combos = 1
        for k in range(w):
            n_combos *= (N - k)
        from math import factorial
        n_combos //= factorial(w)
        n_pauli = n_combos * (8 ** w)
        print(f"  [w={w}]  {n_combos} position choices x 8^{w} = {n_pauli:,}  candidates")

    t0 = time.time()
    n_checked = 0
    n_in_centralizer = 0
    pauli_buf_a = np.zeros((batch_size, N), dtype=np.int8)
    pauli_buf_b = np.zeros((batch_size, N), dtype=np.int8)

    # iterate over all ((position-set), (assignment)) pairs:
    pos_iter = itertools.combinations(range(N), w)
    asg_iter = itertools.product(PAULI_NONZERO_AB, repeat=w)

    buf_idx = 0
    for positions in pos_iter:
        positions = list(positions)
        for assignment in asg_iter:
            pauli_buf_a[buf_idx, :] = 0
            pauli_buf_b[buf_idx, :] = 0
            for k in range(w):
                pos = positions[k]
                ab = assignment[k]
                pauli_buf_a[buf_idx, pos] = ab[0]
                pauli_buf_b[buf_idx, pos] = ab[1]
            buf_idx += 1
            if buf_idx == batch_size:
                # Flush batch
                A_buf = pauli_buf_a[:buf_idx].astype(int)
                B_buf = pauli_buf_b[:buf_idx].astype(int)
                AB_buf = np.hstack([A_buf, B_buf])    # (batch, 80)
                # Centralizer test:  AB . M_cent.T  shape (batch, 40)
                cent_check = (AB_buf @ M_cent.T) % P
                in_cent = (cent_check == 0).all(axis=1)
                n_in_centralizer += int(in_cent.sum())
                if in_cent.any():
                    # Stabilizer test on those passing centralizer:
                    candidates = AB_buf[in_cent]
                    stab_check = (candidates @ K_S) % P
                    in_stab = (stab_check == 0).all(axis=1)
                    non_trivial = candidates[~in_stab]
                    if non_trivial.shape[0] > 0:
                        v = non_trivial[0]
                        elapsed = time.time() - t0
                        a = v[:N]
                        b = v[N:]
                        actual_w = int(((a != 0) | (b != 0)).sum())
                        print(f"  *** FOUND non-trivial logical of weight {actual_w} in {elapsed:.1f}s")
                        return True, actual_w, (a, b)
                n_checked += buf_idx
                buf_idx = 0
        # Reset assignment iterator for next position
        asg_iter = itertools.product(PAULI_NONZERO_AB, repeat=w)
    # Flush final partial batch
    if buf_idx > 0:
        A_buf = pauli_buf_a[:buf_idx].astype(int)
        B_buf = pauli_buf_b[:buf_idx].astype(int)
        AB_buf = np.hstack([A_buf, B_buf])
        cent_check = (AB_buf @ M_cent.T) % P
        in_cent = (cent_check == 0).all(axis=1)
        n_in_centralizer += int(in_cent.sum())
        if in_cent.any():
            candidates = AB_buf[in_cent]
            stab_check = (candidates @ K_S) % P
            in_stab = (stab_check == 0).all(axis=1)
            non_trivial = candidates[~in_stab]
            if non_trivial.shape[0] > 0:
                v = non_trivial[0]
                a = v[:N]
                b = v[N:]
                actual_w = int(((a != 0) | (b != 0)).sum())
                print(f"  *** FOUND non-trivial logical of weight {actual_w}")
                return True, actual_w, (a, b)
        n_checked += buf_idx
    elapsed = time.time() - t0
    print(f"  [w={w}]  no non-trivial logical found.  "
          f"checked {n_checked:,}, "
          f"{n_in_centralizer} in centralizer, all in stabilizer.  "
          f"{elapsed:.1f}s")
    return False, None, None


for w in [1, 2, 3]:
    print(f"\n=== Searching weight {w} ===")
    found, ww, pauli = search_weight(w, batch_size=200_000)
    if found:
        print(f"\nMinimum distance found: d_min <= {ww}")
        print(f"Witness: a = {pauli[0]}")
        print(f"         b = {pauli[1]}")
        break
else:
    print("\nNo logical of weight <= 3 found.  Continuing to weight 4...")
