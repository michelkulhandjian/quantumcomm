"""
Worked example for the D6 paper.  Case (q, d) = (3, 3).

F_81 = F_3[x] / <x^4 + x + 2>  (verify f primitive)
alpha = x  (image of indeterminate)
Singer indexing of PG(3, 3) by K^* / F^* of order 40.

Outputs:
  1. Verification that x has multiplicative order 80 in F_81.
  2. Trace table  Tr(alpha^i)  for i = 0, ..., 79.
  3. Singer incidence first column tau_H of length 40, where
     (tau_H)_i = 1  iff  Tr(alpha^i) = 0.
  4. Full 40 x 40 circulant incidence matrix A.
  5. Verification A A^T = (k - lambda) I + lambda J = 9 I + 4 J,
     i.e. the (40, 13, 4) difference-set property.
  6. (Conjecture probe) rank over F_3 of A.
"""
import numpy as np

P = 3
# f(x) = x^4 + x + 2 over F_3.
# x^4 = -x - 2 = 2x + 1 (mod 3).
F_DESC = "x^4 + x + 2"

def mul_x(a):
    a0, a1, a2, a3 = a
    return (a3 % P, (a0 + 2 * a3) % P, a1 % P, a2 % P)

def add(a, b):
    return tuple((ai + bi) % P for ai, bi in zip(a, b))

def scalar_mul(c, a):
    return tuple((c * ai) % P for ai in a)

def to_str(a):
    parts = []
    for i, c in enumerate(a):
        if c == 0:
            continue
        if i == 0:
            parts.append(f"{c}")
        elif i == 1:
            parts.append(f"{c}x" if c != 1 else "x")
        else:
            parts.append(f"{c}x^{i}" if c != 1 else f"x^{i}")
    return " + ".join(parts) if parts else "0"

# ---- 1. Build powers of alpha ------------------------------------------
ONE = (1, 0, 0, 0)
ZERO = (0, 0, 0, 0)

alpha_powers = [ONE]
cur = ONE
for i in range(1, 81):
    cur = mul_x(cur)
    alpha_powers.append(cur)

# Sanity 1: alpha^80 == 1
print(f"alpha^80 = {alpha_powers[80]}   (expect (1,0,0,0))")
assert alpha_powers[80] == ONE, "alpha^80 != 1; f is not the order-80 polynomial."

# Sanity 2: alpha^k != 1 for 1 <= k <= 79  =>  x is primitive
prim_check = all(alpha_powers[k] != ONE for k in range(1, 80))
print(f"f(x) = {F_DESC}  is primitive over F_3:  {prim_check}")
assert prim_check, "x is not primitive!"

# alpha^40 should equal -1 = 2 in F_3
print(f"alpha^40 = {alpha_powers[40]}   (expect (2,0,0,0))")
assert alpha_powers[40] == (2, 0, 0, 0)

# ---- 2. Trace table -----------------------------------------------------
# Tr(z) = z + z^3 + z^9 + z^27   in  F_81 / F_3
# For z = alpha^i nonzero, Frobenius(alpha^i) = alpha^(3i),
# so Tr(alpha^i) = alpha^i + alpha^(3i mod 80) + alpha^(9i mod 80) + alpha^(27i mod 80).

def trace_alpha(i):
    if i % 80 == 0 and i != 0:
        i = i % 80
    z1 = alpha_powers[i % 80]
    z3 = alpha_powers[(3 * i) % 80]
    z9 = alpha_powers[(9 * i) % 80]
    z27 = alpha_powers[(27 * i) % 80]
    s = add(add(z1, z3), add(z9, z27))
    assert s[1] == 0 and s[2] == 0 and s[3] == 0, \
        f"Tr(alpha^{i}) = {s} is not in F_3; bug in arithmetic."
    return s[0]

# i = 0 means alpha^0 = 1; Tr(1) = 4 mod 3 = 1
traces_full = [trace_alpha(i) for i in range(80)]
print("\nTrace table for i = 0, ..., 79  (Tr(alpha^i) in F_3):")
for row in range(0, 80, 20):
    print(" ", " ".join(f"{traces_full[i]}" for i in range(row, row + 20)))

# Number of trace-zero elements in K^*: should be q^d - 1 = 26.
zeros_full = [i for i in range(80) if traces_full[i] == 0]
print(f"\n#{{i in [0,79] : Tr(alpha^i) = 0}} = {len(zeros_full)}   (expect q^d - 1 = 26)")
assert len(zeros_full) == 26

# ---- 3. Singer first column on the 40 cosets ----------------------------
# Cosets of F_3^* in F_81^* : {alpha^i, alpha^{i+40}} for i = 0,..,39.
# Tr(alpha^{i+40}) = Tr(alpha^i * alpha^40) = Tr(alpha^i * 2)
#                  = 2 Tr(alpha^i)  mod 3   (Tr is F_3-linear).
# So Tr(alpha^i) = 0  <=>  Tr(alpha^{i+40}) = 0 :  trace-zero support is
# well-defined on cosets.  Build the indicator vector:

tau_H = np.array([1 if traces_full[i] == 0 else 0 for i in range(40)], dtype=int)
print(f"\nSinger first column tau_H (length 40):")
print(" ", "".join(str(b) for b in tau_H))
ones_count = int(tau_H.sum())
print(f"#{{ones in tau_H}} = {ones_count}   (expect theta_{{d-1}}(q) = (q^d - 1)/(q-1) = 13)")
assert ones_count == 13, f"Wrong column weight {ones_count}, expected 13."

# Sanity-check on cosets: tau on i+40 should match tau on i (it does by
# construction since traces_full[i] = 0 iff traces_full[i+40] = 0).
for i in range(40):
    assert (traces_full[i] == 0) == (traces_full[i + 40] == 0)

# ---- 4. Build the 40 x 40 circulant incidence matrix --------------------
# Convention: column j is cyclic down-shift of column 0 by j positions.
# A[i, j] = tau_H[ (i - j) mod 40 ].

n = 40
A = np.zeros((n, n), dtype=int)
for j in range(n):
    for i in range(n):
        A[i, j] = tau_H[(i - j) % n]

print(f"\nIncidence matrix A is {n} x {n}, density {A.sum()/(n*n):.4f}")

# ---- 5. Difference-set identity: A A^T = (k - lambda) I + lambda J -----
# Singer DS in PG(d,q): n = theta_d(q),  k = theta_{d-1}(q),
#                       lambda = theta_{d-2}(q).
# (q,d)=(3,3):  n = 40,  k = 13,  lambda = (3^2 - 1)/(3-1) = 4.
k_param = 13
lam = 4
expected = (k_param - lam) * np.eye(n, dtype=int) + lam * np.ones((n, n), dtype=int)
AAT = A @ A.T
match = np.array_equal(AAT, expected)
print(f"A A^T == {k_param - lam} I + {lam} J  :  {match}")
if not match:
    diff = AAT - expected
    print("Diff sample row 0:", diff[0])
    print("AAT diag:", np.unique(np.diag(AAT)))
    print("AAT off-diag values:", np.unique(AAT - np.diag(np.diag(AAT))))

# ---- 6. Rank of A over F_3 ---------------------------------------------
# (For the Gandhi 2018 dimension formula k_quantum = n - rank_p(A).)
def rank_mod_p(M, p=3):
    M = M.copy() % p
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        # find pivot
        pivot = None
        for i in range(r, rows):
            if M[i, c] % p != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        # normalize pivot row
        inv = pow(int(M[r, c]) % p, -1, p)
        M[r] = (M[r] * inv) % p
        # eliminate
        for i in range(rows):
            if i != r and M[i, c] % p != 0:
                M[i] = (M[i] - M[i, c] * M[r]) % p
        r += 1
        if r == rows:
            break
    return r

rA = rank_mod_p(A, P)
print(f"\nrank_F3(A) = {rA}")
print(f"Conjectured quantum dimension  k = n - rank_F3(A) - c = 40 - {rA} - c.")
print("(c will be computed from the symplectic-deficit matrix S in step (c).)")

# ---- 7. Probe the "formal Q_o" at d = 3 (anticipating step (c)) --------
# In the classical Waterloo construction for q odd:
#       Q_o = { z in F_{q^{d+1}}^* : Tr(z^2) = 0 }
# When d is even, |Q_o| / |F_q^*| = (q^d - 1)/(q - 1) = theta_{d-1}(q),
# matching hyperplane cardinality (this is what makes Waterloo work).
# When d is odd, the same formula fails the cardinality match -- this is
# the heart of the Waterloo theorem.  Let's see what we actually get.

# Tr(alpha^{2i}) = 0  iff  i s.t. 2i mod 80 is in zeros_full.
formal_Qo_full = [i for i in range(80) if traces_full[(2 * i) % 80] == 0]
print(f"\nFormal Q_o = {{ alpha^i : Tr(alpha^(2i)) = 0 }}")
print(f"  |Q_o| in F_81^*           = {len(formal_Qo_full)}   (expect q^d - 1 = 26 for a hyperplane)")
print(f"  |Q_o| / (cosets of F_3^*) = {len(set(i % 40 for i in formal_Qo_full))}"
      f"   (expect 13 for a hyperplane in PG(d-1,q))")

# Compare: does formal Q_o equal the trace-zero hyperplane E?
E_full = set(zeros_full)
Qo_full = set(formal_Qo_full)
print(f"  Q_o == E (full)?  {Qo_full == E_full}")
print(f"  |Q_o cap E| = {len(Qo_full & E_full)},   |Q_o minus E| = {len(Qo_full - E_full)},"
      f"   |E minus Q_o| = {len(E_full - Qo_full)}")

# Indicator vector tau_Qo on cosets:
tau_Qo = np.array([1 if i in set(j % 40 for j in formal_Qo_full) else 0
                   for i in range(40)], dtype=int)
print(f"\nFormal tau_Q_o (cosets, length 40):")
print(" ", "".join(str(b) for b in tau_Qo))
print(f"Weight = {int(tau_Qo.sum())}   (expect 13 if it WERE a hyperplane-cardinality quadric)")

# ---- 8. Save for step (c) ---------------------------------------------
np.save("q3d3_tau_H.npy", tau_H)
np.save("q3d3_A.npy", A)
np.save("q3d3_traces_full.npy", np.array(traces_full, dtype=int))
np.save("q3d3_tau_Qo.npy", tau_Qo)
print("\nSaved: q3d3_tau_H.npy, q3d3_A.npy, q3d3_traces_full.npy, q3d3_tau_Qo.npy")
