"""
Verify the LDS-columns-as-centralizer-elements conjecture at (q,d)=(3,2).

The 2021 LDS spreading matrix C is 13x15, with entries in {-1, 0, +1}.
Columns 1-13 come from sign-flipped Singer incidence I_13.
Columns 14-15 are the formal-quadric vector tau_Q and its cyclic shift.

Conjecture:  For each column c_j of C, interpreting (a,b) = (c_j, 0) as a
Z-only Pauli string in F_3^13, the resulting Pauli string is in the
centralizer of Q(3,2), i.e. it commutes with every row of
H = (A | M_Q A) over F_3.

Furthermore, we test whether each column is in the stabilizer rowspan
(it should NOT be -- if it is, the column would be a redundant copy of
a stabilizer rather than a logical operator).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np


P = 3
N = 13

# --- Build Q(3, 2) data ----------------------------------------
def mul_x(a, p=P):
    """Multiply by x in F_3[x] / <x^3 + 2x + 1>: x^3 = -2x - 1 = x + 2."""
    a0, a1, a2 = a
    # a * x = a0 x + a1 x^2 + a2 x^3 = a0 x + a1 x^2 + a2 (x + 2)
    #       = 2 a2 + (a0 + a2) x + a1 x^2
    return ((2 * a2) % p, (a0 + a2) % p, a1 % p)

ONE = (1, 0, 0)
alpha_powers = [ONE]
cur = ONE
for _ in range(1, 26):
    cur = mul_x(cur)
    alpha_powers.append(cur)
assert mul_x(cur) == ONE, "primitivity"

def trace(i):
    z1 = alpha_powers[i % 26]
    z3 = alpha_powers[(3 * i) % 26]
    z9 = alpha_powers[(9 * i) % 26]
    return (z1[0] + z3[0] + z9[0]) % P

traces = [trace(i) for i in range(26)]
tau_H = np.array([1 if traces[i] == 0 else 0 for i in range(N)], dtype=int)
tau_Q = np.array([1 if traces[(2 * i) % 26] == 0 else 0 for i in range(N)], dtype=int)

print(f"tau_H = {tau_H.tolist()}    weight = {int(tau_H.sum())}")
print(f"tau_Q = {tau_Q.tolist()}    weight = {int(tau_Q.sum())}")

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M

A = circ(tau_H) % P
M_Q = circ(tau_Q) % P
H_z = A
H_x = (M_Q @ A) % P
H = np.hstack([H_z, H_x])
M_cent = np.hstack([H_x, (-H_z) % P]) % P

# Right kernel of H over F_3 -> stabilizer-membership test
def right_kernel_F3(M, p=P):
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

K_S = right_kernel_F3(H)
print(f"\nrank(H) = {80 - K_S.shape[1] if False else (2*N - K_S.shape[1])}, "
      f"K_S shape = {K_S.shape}")

# --- The 13 x 15 LDS matrix from [LDS2021] -------------------
# Read row-by-row.  Entries are -1, 0, +1.
# Note: column 0 = first column = first user's signature.

C_LDS = np.array([
#   col: 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    [   1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0],  # row 0
    [   1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1],  # row 1
    [   0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0],  # row 2
    [   1,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],  # row 3
    [   0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],  # row 4
    [   0,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0],  # row 5
    [   0,  0,  0,  1,  0,  1,  1,  0,  0,  0,  1,  0,  0,  0,  0],  # row 6
    [   0,  0,  0,  0, -1,  0,  1, -1,  0,  0,  0,  1,  0,  1,  0],  # row 7
    [   0,  0,  0,  0,  0, -1,  0,  1, -1,  0,  0,  0, -1,  1,  1],  # row 8
    [  -1,  0,  0,  0,  0,  0,  1,  0,  1, -1,  0,  0,  0,  0,  1],  # row 9
    [   0,  1,  0,  0,  0,  0,  0,  1,  0,  1,  1,  0,  0,  0,  0],  # row 10
    [   0,  0,  1,  0,  0,  0,  0,  0,  1,  0, -1, -1,  0,  1,  0],  # row 11
    [   0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0, -1,  1,  0,  1],  # row 12
], dtype=int)

assert C_LDS.shape == (13, 15), f"expected (13,15), got {C_LDS.shape}"

# Convert -1 to 2 in F_3
C_F3 = C_LDS % P

# --- For each column c_j, test centralizer + non-stabilizer ------
print("\n" + "="*78)
print(f"{'col j':<7} {'support':<28} {'in centralizer?':<17} {'in stabilizer?':<15}")
print("-"*78)

n_cent = 0
n_logical = 0
for j in range(15):
    a = C_F3[:, j].copy()    # Z-component (from C_LDS, mapping -1 -> 2)
    b = np.zeros(N, dtype=int)
    ab = np.concatenate([a, b])

    # Centralizer test:  ab . M_cent.T mod 3 == 0
    sym = (ab @ M_cent.T) % P
    is_cent = bool((sym == 0).all())
    if is_cent:
        n_cent += 1

    # Stabilizer test:  ab . K_S mod 3 == 0
    stab = (ab @ K_S) % P
    is_stab = bool((stab == 0).all())
    if is_cent and not is_stab:
        n_logical += 1

    supp = [(i, int(a[i])) for i in range(N) if a[i] != 0]
    supp_str = ",".join(f"{i}^{v}" for i, v in supp)
    print(f"{j:<7} {supp_str:<28} {'YES' if is_cent else 'no':<17} "
          f"{'YES' if is_stab else 'no':<15}")

print("-"*78)
print(f"\nSummary:  {n_cent}/15 columns are in the centralizer.")
print(f"          {n_logical}/15 columns are non-trivial logical operators "
      f"(centralizer minus stabilizer).")
if n_cent == 15:
    print(f"\n*** Conjecture (LDS-columns-as-centralizer) HOLDS at (q,d)=(3,2). ***")
else:
    print(f"\n*** Conjecture FAILS:  {15 - n_cent} columns NOT in centralizer. ***")

# --- Bonus: check that the rank of the LDS-columns subspace is k=6 ---
# (the logical-Z subspace dimension for Q(3,2) = [[13, 6, 3]]_3 is 6).
ldscols_F3 = C_F3.T   # 15 x 13 (each row = one column of C_LDS, Z-only)
# Stack with stabilizer rows to see how many ADDITIONAL dimensions
# the LDS columns add beyond the stabilizer rowspan (Z-part).
H_z_only = H_z   # 13 x 13 over F_3
combined = np.vstack([H_z_only, ldscols_F3])  # 28 x 13
def rank_F3(M):
    return 13 - right_kernel_F3(M).shape[1]    # rank = n - nullity
rank_combined = rank_F3(combined)
rank_stab_z = rank_F3(H_z_only)
print(f"\nrank_F3(H_z) = {rank_stab_z}  (Z-stabilizer dim)")
print(f"rank_F3([H_z; LDS_cols]) = {rank_combined}  "
      f"(span of stab + LDS Z-only logicals)")
print(f"LDS columns add {rank_combined - rank_stab_z} new Z-only logical generators "
      f"(target = k = 6).")
