"""
Approach (ii) DONE RIGHT.  Following Arasu-Dillon-Jungnickel-Pott 1995
Section 4 verbatim:

   For d = 2f even, q odd:
     Q_o = { z in K* : Tr(z^2) = 0 } / F*    (cardinality theta_{d-1}(q))
     E   = { z in K* : Tr(z)   = 0 } / F*    (the "trace-zero hyperplane")
     E . Q^(-1)  =  a . G  +  q^{f-1} . (A - B)              in ZG
   where (A, B) is the Waterloo decomposition, and the right-hand-side
   coefficient extraction gives  w_split = (E . Q^(-1)  -  a)  /  q^{f-1}
   with a = (q^{f-1} - 1)/(q-1)  =  theta_{f-2}(q)   (= 0 when f=1, etc.)

Result 3.3 / Section 4 also gives explicit values
  a = (q^d - q^f)/2,  b = q^{d-1},  c = (q^d + q^f)/2 - q^{d-1}
(for q odd; q even uses Q_e and slightly different multiplicities).

For us, the cyclic cross-correlation of indicator vectors gives
EQinv directly:
   EQinv[k]  =  sum_j  tau_E[(k + j) mod n] * tau_Q[j]
            =  | { j : tau_E[k+j]=1 AND tau_Q[j]=1 } |
            =  | (E + k) cap Q |       (set sum in Z/nZ)

Then for d EVEN we expect EQinv to take only TWO values, one on
position 0 (where g=0 and (E + 0) cap Q = E cap Q has special size),
and another value on positions in D (the complement of E).  In fact
Arasu et al. show EQinv on D takes values  (a + q^{f-1})  and
(a - q^{f-1}), corresponding to A and B respectively.  These are the
elements of the signed split.

For d ODD the structure breaks.  We compute and inspect.
"""

import numpy as np

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M

def cyclic_cross_corr(a, b):
    """Returns vector v with v[k] = sum_j a[(k+j) mod n] * b[j]."""
    n = len(a)
    v = np.zeros(n, dtype=int)
    for k in range(n):
        v[k] = sum(int(a[(k + j) % n]) * int(b[j]) for j in range(n))
    return v

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

def waterloo_split(tau_E, tau_Q, q, d):
    """Compute the signed split w from EQinv following Arasu et al.
    Returns the integer vector w such that, for d even,
    EQinv = a + q^{f-1} * w_indicator  with  w in {-1, 0, +1}.
    For d odd, w is non-canonical."""
    EQinv = cyclic_cross_corr(tau_E, tau_Q)
    n = len(tau_E)
    print(f"EQinv unique values: {sorted(set(int(v) for v in EQinv))}")

    # support on E: positions where tau_E = 1
    # support on D: positions where tau_E = 0
    EQ_on_E = [int(EQinv[i]) for i in range(n) if tau_E[i] == 1]
    EQ_on_D = [int(EQinv[i]) for i in range(n) if tau_E[i] == 0]
    print(f"EQinv on E  unique = {sorted(set(EQ_on_E))}")
    print(f"EQinv on D  unique = {sorted(set(EQ_on_D))}")

    return EQinv

def probe(name, q, d, tau_H, tau_Q):
    print(f"\n=================  {name}  (q={q}, d={d})  =================")
    n = len(tau_H)
    print(f"n = {n},  |E| = {int(tau_H.sum())},  |Q| = {int(tau_Q.sum())}")

    EQinv = waterloo_split(tau_H, tau_Q, q, d)

    # If d is even, locate the two values on D and turn them into a signed
    # vector.  If d is odd, do the analogous and inspect the spread.
    EQinv_on_D = np.array([int(EQinv[i]) if tau_H[i] == 0 else 0
                           for i in range(n)])

    # Center on the *mean* over D, which should equal "a" classically.
    D_indices = [i for i in range(n) if tau_H[i] == 0]
    D_vals = [int(EQinv[i]) for i in D_indices]
    a_est = (max(D_vals) + min(D_vals)) / 2  # midpoint
    radius = (max(D_vals) - min(D_vals)) / 2
    print(f"On D:  midpoint a = {a_est},  half-spread = {radius}")
    print(f"       q^{{f-1}} predicted (d=2f) = {q**(d//2 - 1) if d % 2 == 0 else 'n/a (d odd)'}")

    # Build sign vector on D from EQinv values
    if radius > 0:
        w_unscaled = np.where(tau_H == 1, 0, EQinv - a_est)
        # rescale: signs only
        sgn = np.sign(w_unscaled).astype(int)
    else:
        sgn = np.zeros(n, dtype=int)
    print(f"sgn vector on D (signs only): {sgn}")
    print(f"  |+1| count = {int((sgn == +1).sum())},  |-1| count = {int((sgn == -1).sum())},"
          f"   sum-of-squares = {int(((sgn != 0).sum()))}  (expect q^d = {q**d})")

    W = circ(sgn)
    WWT = W @ W.T

    diag_unique = np.unique(np.diag(WWT))
    off_diag = WWT - np.diag(np.diag(WWT))
    off_unique = np.unique(off_diag)
    print(f"diag(W W^T) = {diag_unique}   (target q^d = {q**d})")
    print(f"off-diag(W W^T) unique = {off_unique}")

    Delta = WWT - (q**d) * np.eye(n, dtype=int)
    print(f"Frobenius |Delta| = {np.linalg.norm(Delta.astype(float)):.3f}")
    print(f"max|Delta|     = {int(np.abs(Delta).max())}")
    print(f"rank_Q(Delta)  = {int(np.linalg.matrix_rank(Delta.astype(float)))}")
    for p in [2, 3, 5, 7, 11]:
        if p == q:
            continue
        r = rank_mod_p(Delta, p)
        print(f"rank_F{p}(Delta) = {r}")

    return sgn, W, Delta


# ---- (q,d) = (3,2): the user's 2021 case, classical Waterloo regime ----
P3 = 3
def mul_x_27(a):
    a0, a1, a2 = a
    return ((2 * a2) % P3, (a0 + a2) % P3, a1 % P3)
ap27 = [(1,0,0)]
for _ in range(1, 27):
    ap27.append(mul_x_27(ap27[-1]))
def tr27(i):
    z1 = ap27[i % 26]; z3 = ap27[(3*i) % 26]; z9 = ap27[(9*i) % 26]
    return sum(z1[k] + z3[k] + z9[k] for k in [0]) % 3 if False \
           else (z1[0] + z3[0] + z9[0]) % 3
trc = [tr27(i) for i in range(26)]
tau_H_32 = np.array([1 if trc[i] == 0 else 0 for i in range(13)], dtype=int)
tau_Qo_32 = np.array([1 if trc[(2*i) % 26] == 0 else 0 for i in range(13)], dtype=int)
probe("(q,d)=(3,2)", 3, 2, tau_H_32, tau_Qo_32)


# ---- (q,d) = (3,3): focal d-odd case ----------------------------------
tau_H_33 = np.load("q3d3_tau_H.npy")
tau_Qo_33 = np.load("q3d3_tau_Qo.npy")
probe("(q,d)=(3,3)", 3, 3, tau_H_33, tau_Qo_33)


# ---- (q,d) = (2,3): cross-check ---------------------------------------
def mul_x_16(a):
    a0, a1, a2, a3 = a
    return (a3 % 2, (a0+a3) % 2, a1 % 2, a2 % 2)
ap16 = [(1,0,0,0)]
for _ in range(1, 16):
    ap16.append(mul_x_16(ap16[-1]))
def tr16(i):
    z1 = ap16[i % 15]; z2 = ap16[(2*i) % 15]; z4 = ap16[(4*i) % 15]; z8 = ap16[(8*i) % 15]
    return (z1[0]+z2[0]+z4[0]+z8[0]) % 2
trc16 = [tr16(i) for i in range(15)]
tau_H_23 = np.array([1 if trc16[i] == 0 else 0 for i in range(15)], dtype=int)
tau_Qe_23 = np.array([1 if trc16[(3*i) % 15] == 0 else 0 for i in range(15)], dtype=int)
probe("(q,d)=(2,3)", 2, 3, tau_H_23, tau_Qe_23)
