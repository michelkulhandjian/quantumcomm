"""
Investigate the q=2 distance ceiling.

Hypothesis:  for all  Q(2, d),  the cross-correlation  u = M_Q * tau_H
is the all-ones vector  mod 2.  If true, this means columns 0 and 1 of
H_x agree mod 2, hence  Z_0 Z_1  is a Z-only logical of weight 2,
forcing d_min(Q(2, d)) <= 2 for all d.

Verify at d in {2, 3, 4, 5, 6}.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools, numpy as np


def build_field(p, m, poly):
    fs = p**m - 1
    pp = poly[:-1]
    def mul_x(a):
        new = [0]*m
        new[0] = (-pp[0]*a[m-1]) % p
        for k in range(1, m):
            new[k] = (a[k-1] - pp[k]*a[m-1]) % p
        return tuple(new)
    one = tuple(1 if k==0 else 0 for k in range(m))
    cur=one; seen={one:0}; aps=[one]
    for kk in range(1, fs):
        cur = mul_x(cur)
        if cur==one or cur in seen: return None
        seen[cur]=kk; aps.append(cur)
    if mul_x(cur) != one: return None
    return aps

def find_primitive(p, m):
    for c in itertools.product(range(p), repeat=m):
        if c[0]==0: continue
        poly = list(c)+[1]
        ap = build_field(p, m, poly)
        if ap is not None: return poly, ap
    return None, None

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n): M[i, j] = c[(i - j) % n]
    return M


for d in [2, 3, 4, 5, 6]:
    q = 2
    p = 2
    m = d + 1
    n_q = q**m - 1
    n = n_q  # since q-1 = 1
    poly, aps = find_primitive(p, m)

    fs = n_q
    zero = tuple(0 for _ in range(m))

    def Tr(i):
        s = [0]*m
        for k in range(m):
            ap = aps[(i * pow(q, k, fs)) % fs]
            for r in range(m):
                s[r] = (s[r] + ap[r]) % p
        return tuple(s)

    tau_H = np.array([1 if Tr(i) == zero else 0 for i in range(n)])
    xi = q + 1  # = 3 for q=2
    tau_Q = np.array([1 if Tr((xi * i) % fs) == zero else 0 for i in range(n)])

    A = circ(tau_H) % q
    M_Q = circ(tau_Q) % q
    u = (M_Q @ tau_H) % q

    print(f"\n--- Q(2, {d}): n = {n}, |E| = {int(tau_H.sum())}, "
          f"|Q| = {int(tau_Q.sum())} ---")
    print(f"  u = M_Q tau_H mod 2 = {u.tolist()}")
    print(f"  Is u constant?   {np.all(u == u[0])}    (value u[0] = {int(u[0])})")
    print(f"  Distinct values: {sorted(set(u.tolist()))}")

print("\n--- conclusion ---")
print("If u is constant 1 mod 2, then Z_0 Z_1 is a centralizer element of"
      " weight 2 (when not in row-span of H), giving d_min <= 2.")
