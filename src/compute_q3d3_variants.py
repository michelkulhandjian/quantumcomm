"""
Probe the symplectic deficit for several candidate constructions at
(q, d) = (3, 3).  Goal: find a recipe whose symplectic-deficit rank
exposes the Waterloo defect (i.e., is non-zero for d odd).

Variants tried:
   V1 : H = (A | M_Q A)        over F_3   (Gandhi original) -> c=0
   V2 : H = (A | M_Q)          over F_3
   V3 : H = (A | M_Q^T A)      over F_3
   V4 : H = (A | M_Q A)        over F_2
   V5 : H = (A | M_Q)          over F_2
   V6 : H = (A | A^T)          over F_3
   V7 : H = (W_+ | W_-)  signed split based on formal Q_o     over F_3
   V8 : Twisted product H = (A circ(tau_E) | A circ(tau_Q))   over F_3
"""
import numpy as np

P_LIST = [2, 3]
N = 40

tau_H = np.load("q3d3_tau_H.npy")
tau_Qo = np.load("q3d3_tau_Qo.npy")
A_int = np.load("q3d3_A.npy")

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

def deficit(Hz, Hx, p):
    S = (Hx @ Hz.T - Hz @ Hx.T) % p
    return rank_mod_p(S, p)

# Build base matrices
A = A_int % 1                       # placeholder so A is int array
A = A_int.astype(int)
M_Q = circ(tau_Qo)

# Signed-split W based on formal Q_o:
# tau_E = tau_H (the trace-zero hyperplane indicator on cosets)
# tau_W: +1 if i in support(tau_H) AND in support(tau_Qo)
#        -1 if i in support(tau_H) AND NOT in support(tau_Qo)
#        0  otherwise
tau_W = np.array([
    +1 if (tau_H[i] == 1 and tau_Qo[i] == 1) else
    -1 if (tau_H[i] == 1 and tau_Qo[i] == 0) else
    0
    for i in range(N)], dtype=int)
W = circ(tau_W)
W_plus  = circ(np.where(tau_W ==  1, 1, 0))   # support of +1
W_minus = circ(np.where(tau_W == -1, 1, 0))   # support of -1

# Twisted variant using indicator of E and Q
tau_E = tau_H.copy()
tau_Q = tau_Qo.copy()
M_E = circ(tau_E)
M_QQ = circ(tau_Q)

print(f"{'variant':<60} {'p':<3} {'rank S':<8} {'c=rank/2':<8}")
print("-" * 90)

# V1: H = (A | M_Q A)
for p in P_LIST:
    Hz, Hx = A % p, (M_Q @ A) % p
    rS = deficit(Hz, Hx, p)
    print(f"V1  H = (A | M_Q A)                                          {p:<3} {rS:<8} {rS//2:<8}")

# V2: H = (A | M_Q)
for p in P_LIST:
    Hz, Hx = A % p, M_Q % p
    rS = deficit(Hz, Hx, p)
    print(f"V2  H = (A | M_Q)                                            {p:<3} {rS:<8} {rS//2:<8}")

# V3: H = (A | M_Q^T A)
for p in P_LIST:
    Hz, Hx = A % p, (M_Q.T @ A) % p
    rS = deficit(Hz, Hx, p)
    print(f"V3  H = (A | M_Q^T A)                                        {p:<3} {rS:<8} {rS//2:<8}")

# V4 already covered by V1 over p=2.
# V5 already covered by V2 over p=2.

# V6: H = (A | A^T)
for p in P_LIST:
    Hz, Hx = A % p, A.T % p
    rS = deficit(Hz, Hx, p)
    print(f"V6  H = (A | A^T)                                            {p:<3} {rS:<8} {rS//2:<8}")

# V7: H = (W_+ | W_-)  signed split (Waterloo-attempt PCM)
for p in P_LIST:
    Hz, Hx = W_plus % p, W_minus % p
    rS = deficit(Hz, Hx, p)
    print(f"V7  H = (W_+ | W_-)  signed split based on formal Q_o        {p:<3} {rS:<8} {rS//2:<8}")

# V8: Twisted H = (A M_E | A M_Q)
for p in P_LIST:
    Hz, Hx = (A @ M_E) % p, (A @ M_QQ) % p
    rS = deficit(Hz, Hx, p)
    print(f"V8  H = (A M_E | A M_Q)                                      {p:<3} {rS:<8} {rS//2:<8}")

# V9: H = (M_E | M_Q)   pure circulant pair, no A multiplication
for p in P_LIST:
    Hz, Hx = M_E % p, M_QQ % p
    rS = deficit(Hz, Hx, p)
    print(f"V9  H = (M_E | M_Q)                                          {p:<3} {rS:<8} {rS//2:<8}")

# V10: H = (M_E | M_Q^T)
for p in P_LIST:
    Hz, Hx = M_E % p, M_QQ.T % p
    rS = deficit(Hz, Hx, p)
    print(f"V10 H = (M_E | M_Q^T)                                        {p:<3} {rS:<8} {rS//2:<8}")

print()
print("Reminder: for d=3 we want a recipe with non-zero rank S over the")
print("relevant field, AND zero rank when d is even.  We will replicate the")
print("best variant at (q,d)=(3,2) (your 2021 case) for the parity check.")
