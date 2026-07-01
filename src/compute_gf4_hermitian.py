"""
Approach (b): GF(4) Hermitian / CRSS angle.

CRSS framework (Calderbank-Rains-Shor-Sloane):
  Encode Pauli operators as GF(4) elements:
       I -> 0,    X -> 1,    Z -> omega,    Y -> omega^2 = bar{omega}.
  Conjugation in GF(4) is x -> x^2 (Frobenius), which is also bar{x}:
       0 <-> 0,    1 <-> 1,    omega <-> bar{omega}.
  A vector u in GF(4)^n encodes an n-qubit Pauli string.
  Two Pauli strings commute iff Tr(<u, v_bar>) = 0 in F_2,
  where the Hermitian inner product is <u, v_bar> = sum u_i bar{v_i}.

We build the GF(4) PCM
       H_GF4[i, j] = omega * A[i, j]  +  1 * (M_Q A)[i, j]
i.e. a Z-stabilizer wherever A=1, an X-stabilizer wherever (M_Q A)=1,
and Y wherever both, in row i.  Compute the Hermitian Gram matrix
     G_H[i, k] = sum_j H[i, j] * H[k, j]^2    in GF(4).
The EAQECC ebit count is
     c_Hermitian = rank_GF4(G_H)
(NB: not divided by 2; in the Hermitian framework the rank is the c).

The trace inner product gives c_F2_symplectic = rank_GF4 / 2 in some
conventions; we report both for clarity.
"""
import numpy as np


# ---------- GF(4) arithmetic ------------------------------------------
# elements:  0 -> 0,  1 -> 1,  2 -> omega,  3 -> bar{omega}
# representation: bit_low = "1 part",  bit_high = "omega part"
#   0 = 00,  1 = 01,  omega = 10,  omega_bar = 11
# addition is XOR.
# multiplication: omega * omega = omega + 1 = bar{omega}, etc.

def gf4_add(a, b):
    return a ^ b

def gf4_mul(a, b):
    # Use the recipe omega^2 = omega + 1
    # (a0 + a1*omega) * (b0 + b1*omega)
    a0, a1 = a & 1, (a >> 1) & 1
    b0, b1 = b & 1, (b >> 1) & 1
    # = a0 b0 + (a0 b1 + a1 b0) omega + a1 b1 omega^2
    # omega^2 = omega + 1, so
    # = (a0 b0 + a1 b1) + (a0 b1 + a1 b0 + a1 b1) omega
    c0 = (a0 * b0 + a1 * b1) % 2
    c1 = (a0 * b1 + a1 * b0 + a1 * b1) % 2
    return c0 | (c1 << 1)

def gf4_conj(a):
    # conjugation in GF(4) is the squaring (Frobenius)
    # 0 -> 0,  1 -> 1,  omega -> omega^2 = omega_bar,  omega_bar -> omega
    # equivalently:  swap omega <-> omega_bar, fix 0 and 1
    return gf4_mul(a, a)

# Sanity:
assert gf4_mul(2, 2) == 3  # omega * omega = omega_bar
assert gf4_mul(3, 3) == 2  # omega_bar^2 = omega
assert gf4_mul(2, 3) == 1  # omega * omega_bar = 1
assert gf4_conj(2) == 3
assert gf4_conj(3) == 2

def gf4_matmul(M1, M2):
    """Matrix multiplication over GF(4)."""
    n, p = M1.shape
    p2, m = M2.shape
    assert p == p2
    out = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            acc = 0
            for k in range(p):
                acc = gf4_add(acc, gf4_mul(int(M1[i, k]), int(M2[k, j])))
            out[i, j] = acc
    return out

def gf4_conj_T(M):
    """Hermitian conjugate transpose (entrywise conjugation, then transpose)."""
    n, m = M.shape
    out = np.zeros((m, n), dtype=int)
    for i in range(n):
        for j in range(m):
            out[j, i] = gf4_conj(int(M[i, j]))
    return out

def gf4_rank(M):
    """Rank of M over GF(4) by Gaussian elimination."""
    M = M.copy().astype(int)
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        # multiply row r by inverse of pivot
        # inverse in GF(4): 1->1, omega->omega_bar, omega_bar->omega
        inv = M[r, c]  # since x * x^2 = x^3 = 1, x^{-1} = x^2 = conj
        inv = gf4_conj(int(inv))
        for j in range(cols):
            M[r, j] = gf4_mul(int(M[r, j]), inv)
        # eliminate
        for i in range(rows):
            if i != r and M[i, c] != 0:
                coef = int(M[i, c])
                for j in range(cols):
                    sub = gf4_mul(coef, int(M[r, j]))
                    M[i, j] = gf4_add(int(M[i, j]), sub)
        r += 1
        if r == rows:
            break
    return r


def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M


def crss_probe(name, q, d, tau_H, tau_Q):
    print(f"\n=================  {name}  (q={q}, d={d})  =================")
    n = len(tau_H)
    A = circ(tau_H)
    M_Q = circ(tau_Q)
    M_Q_A = (M_Q.astype(int) @ A.astype(int)) % 2  # mod 2 for GF(4) embed

    # Build H_GF4[i, j] = omega * A[i, j] + 1 * (M_Q A)[i, j]
    # (operating component-wise in GF(4), with A and M_Q_A as F_2 matrices)
    OMEGA = 2
    H_GF4 = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            zpart = OMEGA if A[i, j] % 2 else 0
            xpart = 1 if M_Q_A[i, j] else 0
            H_GF4[i, j] = gf4_add(zpart, xpart)

    # Hermitian Gram: G_H = H * (H)^*    (where * is conj-transpose)
    H_star = gf4_conj_T(H_GF4)
    G_H = gf4_matmul(H_GF4, H_star)

    # Stats
    nonzero = int((G_H != 0).sum())
    diag_unique = sorted(set(int(G_H[i, i]) for i in range(n)))
    off_unique = sorted(set(int(G_H[i, j]) for i in range(n) for j in range(n) if i != j))
    print(f"|G_H| nonzero = {nonzero} / {n*n}")
    print(f"diag(G_H) unique values: {diag_unique}")
    print(f"off-diag(G_H) unique values: {off_unique}")

    rank_GH = gf4_rank(G_H)
    print(f"rank_GF4(G_H) = {rank_GH}   (this is the EAQECC ebit count c_H)")
    print(f"(For comparison: V_1 over F_2 gave c = 4 at (3,3),"
          f" c = 6 at (3,2), c = 0 at (2,3).)")

    # Hermitian self-orthogonality? (G_H = 0)
    print(f"H is Hermitian self-orthogonal: {nonzero == 0}")

    return rank_GH


# ---------- (q,d) = (3,2): user's 2021 case --------------------------
P3 = 3
def mul_x_27(a):
    a0, a1, a2 = a
    return ((2 * a2) % P3, (a0 + a2) % P3, a1 % P3)
ap27 = [(1,0,0)]
for _ in range(1, 27):
    ap27.append(mul_x_27(ap27[-1]))
def tr27(i):
    z1 = ap27[i % 26]; z3 = ap27[(3*i) % 26]; z9 = ap27[(9*i) % 26]
    return (z1[0] + z3[0] + z9[0]) % 3
trc = [tr27(i) for i in range(26)]
tau_H_32 = np.array([1 if trc[i] == 0 else 0 for i in range(13)], dtype=int)
tau_Qo_32 = np.array([1 if trc[(2*i) % 26] == 0 else 0 for i in range(13)], dtype=int)
crss_probe("(q,d)=(3,2)", 3, 2, tau_H_32, tau_Qo_32)


# ---------- (q,d) = (3,3): focal case ---------------------------------
tau_H_33 = np.load("q3d3_tau_H.npy")
tau_Qo_33 = np.load("q3d3_tau_Qo.npy")
crss_probe("(q,d)=(3,3)", 3, 3, tau_H_33, tau_Qo_33)


# ---------- (q,d) = (2,3): cross-check --------------------------------
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
crss_probe("(q,d)=(2,3)", 2, 3, tau_H_23, tau_Qe_23)
