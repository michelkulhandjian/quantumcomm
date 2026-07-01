"""
Sanity check: tau_H must be invariant under the Frobenius map
   i  ->  q*i mod n        (here  q=3,  n=40)
because  Tr(alpha^{q*i}) = Tr(alpha^i)  (squaring is identity on F_q).

If saved tau_H is NOT Frobenius-invariant, there's a bug somewhere.
"""
import numpy as np

P = 3
N = 40

tau_H = np.load("q3d3_tau_H.npy")
traces_full = np.load("q3d3_traces_full.npy")

print(f"Saved tau_H positions of 1: "
      f"{[i for i in range(N) if tau_H[i] == 1]}")
print(f"Saved tau_H weight = {int(tau_H.sum())}\n")

# Check: tau_H[i] should equal tau_H[3*i mod 40] for ALL i.
bad = []
for i in range(N):
    j = (3 * i) % N
    if tau_H[i] != tau_H[j]:
        bad.append((i, j, int(tau_H[i]), int(tau_H[j])))
if bad:
    print(f"FAILED Frobenius invariance check.  "
          f"First few mismatches: {bad[:5]}")
else:
    print("Frobenius invariance: OK")

# Direct check from traces_full:
# For i in [0, 39]:  tau_H[i] = 1 iff traces_full[i] = 0.
# But also:  traces_full[i] = traces_full[3*i mod 80]  (Frobenius on Z/80Z).
print("\nDirect check from traces_full:")
print(f"  traces_full[27] = {int(traces_full[27])}   "
      f"(expect 0 if tau_H[27]=1, since 27 ≡ 3·9 mod 80)")
print(f"  traces_full[9]  = {int(traces_full[9])}    "
      f"(expect 0 if tau_H[9]=1)")
print(f"  traces_full[25] = {int(traces_full[25])}   "
      f"(expect 0 if tau_H[25]=1, since 25 ≡ 3·5 mod 80? No: 3·5=15. 25 = 5*5)")
print()

# Recompute tau_H from traces_full and compare:
tau_H_recomputed = np.array([1 if int(traces_full[i]) == 0 else 0
                             for i in range(N)], dtype=int)
print(f"tau_H from saved npy:        {tau_H}")
print(f"tau_H recomputed from traces:{tau_H_recomputed}")
print(f"Match: {np.array_equal(tau_H, tau_H_recomputed)}")

# If they do NOT match, the saved file was generated under different
# convention.  Inspect cyclotomic cosets:
def cyclo_cosets(n, q):
    seen = set()
    cosets = []
    for i in range(n):
        if i in seen:
            continue
        c = []
        cur = i
        while cur not in seen:
            seen.add(cur)
            c.append(cur)
            cur = (q * cur) % n
        cosets.append(c)
    return cosets

cosets = cyclo_cosets(N, P)
print(f"\nCyclotomic cosets of Z/{N}Z under x -> {P}*x:")
for c in cosets:
    in_tau = [i for i in c if tau_H[i] == 1]
    if 0 < len(in_tau) < len(c):
        flag = "  <-- MIXED"
    else:
        flag = ""
    print(f"  {c}    in tau_H: {in_tau}{flag}")
