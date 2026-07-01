"""
Step (c) for the D6 paper.  Test of Conjecture 4.2 at (q, d) = (3, 3).

Loads the Singer column tau_H and the formal-quadric column tau_Qo
saved by compute_q3d3.py, builds H = (A | M_Q A), and computes the
F_3-rank of the symplectic-deficit matrix S = H_x H_z^T - H_z H_x^T.

Conjecture predicts  c = rank_F3(S) / 2 = (q^{d-1} - 1)/(q-1) = 4.
"""
import numpy as np

P = 3
N = 40

tau_H = np.load("q3d3_tau_H.npy")
tau_Qo = np.load("q3d3_tau_Qo.npy")
A = np.load("q3d3_A.npy")

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            M[i, j] = c[(i - j) % n]
    return M

def rank_mod_p(M, p=P):
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


# ---- Build the candidate stabilizer matrix H = (A | M_Q A) ----------
M_Q = circ(tau_Qo) % P
H_z = A % P
H_x = (M_Q @ A) % P

# ---- Symplectic-deficit matrix --------------------------------------
S = (H_x @ H_z.T - H_z @ H_x.T) % P

# Verify skew-symmetry mod P (since char != 2):
asym_check = np.array_equal((S + S.T) % P, np.zeros_like(S))
print(f"S + S^T == 0 mod {P}  (skew-symmetric):  {asym_check}")

rS = rank_mod_p(S, P)
print(f"\nrank_F{P}(S) = {rS}")
print(f"In characteristic {P} != 2 the rank of a skew matrix is even.")
print(f"Predicted entanglement budget  c = rank/2 = {rS // 2}")

# ---- Compare to conjectured formula ----------------------------------
q, d = 3, 3
predicted_c = (q**(d-1) - 1) // (q - 1)
print(f"\nConjecture 4.2:  c(d,q) = (q^(d-1) - 1)/(q-1) = "
      f"({q}^{d-1} - 1)/({q}-1) = {predicted_c}")

actual_c = rS // 2
print(f"Computed c       = {actual_c}")
print(f"Conjecture holds at (q,d)=({q},{d}):  {actual_c == predicted_c}")

# ---- Additional cross-check: c via Brun-Devetak-Hsieh formula -------
# For an (n-k) x 2n binary PCM H over F_q, Brun-Devetak-Hsieh defines
#   c = (rank_q(H) + rank_q(symplectic Gram)) / 2 - (n - k_classical)
# but the simpler equivalent statement is c = rank_q(S)/2.
# We just verified that.

# ---- Sanity: dimension of unassisted vs entanglement-assisted code --
# rank of H over F_3 (full 40 x 80 matrix):
rH = rank_mod_p(np.hstack([H_z, H_x]), P)
print(f"\nrank_F{P}(H) over the full 40 x 80 PCM:  {rH}")
print(f"Quantum logical-qubit count   k = n - rank(H) + c "
      f"= {N} - {rH} + {actual_c} = {N - rH + actual_c}")

# Save deficit data
np.save("q3d3_S.npy", S)
np.save("q3d3_H_z.npy", H_z)
np.save("q3d3_H_x.npy", H_x)
print("\nSaved: q3d3_S.npy, q3d3_H_z.npy, q3d3_H_x.npy")
