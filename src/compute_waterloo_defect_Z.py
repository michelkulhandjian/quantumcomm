"""
Approach (ii) -- the hard way.  Probe the classical Waterloo defect
   Delta := W W^T - q^d * I        (over Z)
where W = W_+ - W_- is the signed circulant indexed by the
formal-quadric split  +1 on E cap Q,  -1 on E setminus Q,  0 elsewhere.

For (q,d) with d EVEN, Theorem 4.1 of Arasu-Dillon-Jungnickel-Pott 1995
guarantees that with the *correct* signed split derived from a
non-degenerate quadric of cardinality theta_{d-1}(q), one has
Delta = 0 in Z.  Our formal-Q is non-degenerate at d even but the
sign assignment may differ from Arasu et al.; nevertheless we can
verify Delta has small structure in this regime.

For d ODD, no Waterloo decomposition exists; Delta is non-zero, and
its structure / rank gives us a candidate "Waterloo deficit" that we
will then attempt to lift into an EAQECC ebit count.

We compute Delta for (q, d) in {(3, 2), (3, 3), (2, 3)} and examine:
  - Frobenius norm
  - rank over Q (or any field of characteristic coprime to q*p)
  - rank over F_2, F_3, F_5, F_7
  - eigenvalue distribution (over R)
"""
import numpy as np

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

def rank_Z(M):
    """Rank over Q via numpy."""
    return int(np.linalg.matrix_rank(M.astype(float)))


def waterloo_probe(name, q, d, tau_H, tau_Q):
    """Compute the classical Waterloo defect Delta = W W^T - q^d I over Z."""
    n = len(tau_H)
    print(f"\n=================  {name}  (q={q}, d={d})  =================")
    print(f"n = {n},  |E_cosets| = {int(tau_H.sum())},  |Q_cosets| = {int(tau_Q.sum())}")

    # signed indicator: +1 on E cap Q, -1 on E setminus Q, 0 otherwise
    sgn = np.array([
        +1 if (tau_H[i] == 1 and tau_Q[i] == 1) else
        -1 if (tau_H[i] == 1 and tau_Q[i] == 0) else
        0
        for i in range(n)], dtype=int)
    a_count = int((sgn == +1).sum())
    b_count = int((sgn == -1).sum())
    print(f"|E cap Q| = {a_count}  (+1's),   |E minus Q| = {b_count}  (-1's),   "
          f"sum of squares = {a_count + b_count} (expect = |E_cosets|)")

    W = circ(sgn)
    WWT = W @ W.T
    target = (q ** d) * np.eye(n, dtype=int)
    Delta = WWT - target

    diag_unique = np.unique(np.diag(WWT))
    off_diag = WWT - np.diag(np.diag(WWT))
    off_unique = np.unique(off_diag)
    print(f"diag(W W^T)     unique values: {diag_unique}   (target q^d = {q**d})")
    print(f"off-diag(W W^T) unique values: {off_unique}")

    print(f"|Delta|_F  = {np.linalg.norm(Delta.astype(float)):.3f}")
    print(f"max |Delta_ij| = {int(np.abs(Delta).max())}")
    print(f"rank_Q(Delta)  = {rank_Z(Delta)}")
    for p in [2, 3, 5, 7]:
        if p == q:
            continue  # skip the natural field where things vanish
        rp = rank_mod_p(Delta, p)
        print(f"rank_F{p}(Delta) = {rp}")

    return W, Delta


# ---------- (q,d) = (3,2): user's 2021 paper case ---------------------
# tau_H, tau_Qo from F_27 = F_3[x]/(x^3+2x+1)
P3 = 3
def mul_x_27(a):
    a0, a1, a2 = a
    return ((2 * a2) % P3, (a0 + a2) % P3, a1 % P3)

ONE3 = (1, 0, 0)
ap27 = [ONE3]
cur = ONE3
for _ in range(1, 27):
    cur = mul_x_27(cur)
    ap27.append(cur)

def tr27(i):
    z1 = ap27[i % 26]
    z3 = ap27[(3*i) % 26]
    z9 = ap27[(9*i) % 26]
    s = tuple((a + b + c) % 3 for a, b, c in zip(z1, z3, z9))
    return s[0]

trc = [tr27(i) for i in range(26)]
tau_H_32 = np.array([1 if trc[i] == 0 else 0 for i in range(13)], dtype=int)
tau_Qo_32 = np.array([1 if trc[(2*i) % 26] == 0 else 0 for i in range(13)], dtype=int)

W32, D32 = waterloo_probe("(q,d)=(3,2)  Waterloo regime", 3, 2,
                          tau_H_32, tau_Qo_32)


# ---------- (q,d) = (3,3): the focal case -----------------------------
tau_H_33 = np.load("q3d3_tau_H.npy")
tau_Qo_33 = np.load("q3d3_tau_Qo.npy")
W33, D33 = waterloo_probe("(q,d)=(3,3)  d odd, Waterloo defect", 3, 3,
                          tau_H_33, tau_Qo_33)


# ---------- (q,d) = (2,3): cross-check ---------------------------------
P2 = 2
def mul_x_16(a):
    a0, a1, a2, a3 = a
    return (a3 % P2, (a0 + a3) % P2, a1 % P2, a2 % P2)

ONE2 = (1, 0, 0, 0)
ap16 = [ONE2]
cur = ONE2
for _ in range(1, 16):
    cur = mul_x_16(cur)
    ap16.append(cur)

def tr16(i):
    z1 = ap16[i % 15]
    z2 = ap16[(2*i) % 15]
    z4 = ap16[(4*i) % 15]
    z8 = ap16[(8*i) % 15]
    s = tuple((a+b+c+d) % 2 for a, b, c, d in zip(z1, z2, z4, z8))
    return s[0]

trc16 = [tr16(i) for i in range(15)]
tau_H_23 = np.array([1 if trc16[i] == 0 else 0 for i in range(15)], dtype=int)
# formal Q_e at q=2: Tr(z^{q+1}) = Tr(z^3) = 0
tau_Qe_23 = np.array([1 if trc16[(3*i) % 15] == 0 else 0 for i in range(15)], dtype=int)

W23, D23 = waterloo_probe("(q,d)=(2,3)  d odd, q=2", 2, 3,
                          tau_H_23, tau_Qe_23)
