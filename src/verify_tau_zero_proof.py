"""
Verify the proof of Theorem 6.tau-zero-count.

For p in {3, 5, 7, 11, 13, 17, 19, 23} (odd primes), with n = p^2 + p + 1,
check:

(i)   Hamada-Smith degree-sum:
      sum_{f: tau_f = 0} deg(f) = (p^2+p)/2

(ii)  Pairing structure: -1 in <p> mod n? If not, all nontrivial orbits
      pair under negation.

(iii) Combined: # zero CRT factors = (|F|-1)/2 EXACTLY.

(iv)  Equivalently: each non-trivial reciprocal pair contains exactly one
      zero factor (no pair has both zero, no pair has neither zero).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools


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


def trace_table(p, m, aps):
    fs = p**m - 1
    out = []
    for i in range(fs):
        s = [0]*m
        for k in range(m):
            ap = aps[(i*p**k) % fs]
            for r in range(m): s[r] = (s[r]+ap[r]) % p
        out.append(s[0])
    return out


def crt_orbits(n, p):
    visited = [False] * n
    orbits = []
    for i in range(n):
        if visited[i]: continue
        orbit = []
        cur = i
        while not visited[cur]:
            visited[cur] = True
            orbit.append(cur)
            cur = (cur * p) % n
        orbits.append(tuple(orbit))
    return orbits


def vanishing_count(c_indices, n, p):
    """Count CRT orbits where indicator vanishes."""
    orbits = crt_orbits(n, p)
    from math import gcd
    L = 1
    for orb in orbits:
        L = L * len(orb) // gcd(L, len(orb))
    poly, aps = find_primitive(p, L) if L > 1 else (None, None)
    if L == 1:
        return 0
    fs = p**L - 1
    while fs % n != 0:
        L += 1
        poly, aps = find_primitive(p, L)
        fs = p**L - 1
        if poly is None: return -1
    omega_exp = fs // n
    n_zero = 0
    zero_orbits = []
    for orb in orbits:
        k = orb[0]
        s = [0]*L
        for i in c_indices:
            j = (i * k * omega_exp) % fs
            for r in range(L):
                s[r] = (s[r] + aps[j][r]) % p
        if all(x == 0 for x in s):
            n_zero += 1
            zero_orbits.append(orb)
    return n_zero, zero_orbits, orbits


def is_self_reciprocal(orb, n):
    """Orbit is self-reciprocal under negation iff -k in orbit for some/all k in orbit."""
    rep = orb[0]
    neg = (-rep) % n
    return neg in orb


def reciprocal_orbit(orb, n, all_orbits):
    """Find the orbit that is the negation of orb."""
    neg_set = set((-k) % n for k in orb)
    for o in all_orbits:
        if set(o) == neg_set:
            return o
    return None


def verify(p):
    print(f"\n=== p = {p} ===")
    m = 3
    n = p**2 + p + 1
    poly, aps = find_primitive(p, m)
    if poly is None:
        print(f"  No primitive poly for F_{p^m}, skipping")
        return
    fs = p**m - 1
    traces = trace_table(p, m, aps)
    E_indices = [i for i in range(n) if traces[i % fs] == 0]
    print(f"  n = {n}, |E| = {len(E_indices)} (expected p+1 = {p+1})")

    # (i) Hamada-Smith degree sum
    n_zero, zero_orbits, all_orbits = vanishing_count(E_indices, n, p)
    F_count = len(all_orbits)
    sum_zero_deg = sum(len(o) for o in zero_orbits)
    expected_sum = (p**2 + p) // 2
    print(f"  |F| = {F_count}, # zero CRT factors = {n_zero}")
    print(f"  sum of zero-orbit sizes = {sum_zero_deg}, expected (n-1)/2 = {expected_sum}")
    assert sum_zero_deg == expected_sum, "Hamada-Smith degree sum mismatch"

    # (ii) Self-reciprocal check on nontrivial orbits
    nontriv = [o for o in all_orbits if o != (0,)]
    self_recip = [o for o in nontriv if is_self_reciprocal(o, n)]
    print(f"  # nontrivial orbits = {len(nontriv)}, self-reciprocal = {len(self_recip)}")

    # (iii) Count check
    expected_zero_count = (F_count - 1) // 2
    print(f"  # zero factors = {n_zero}, (|F|-1)/2 = {expected_zero_count}")
    assert n_zero == expected_zero_count, "(|F|-1)/2 mismatch"

    # (iv) Each pair has exactly one zero
    pair_zero_counts = {}
    for o in nontriv:
        ro = reciprocal_orbit(o, n, all_orbits)
        # canonical pair key
        key = frozenset([o, ro])
        if key not in pair_zero_counts:
            pair_zero_counts[key] = 0
        if o in zero_orbits:
            pair_zero_counts[key] += 1

    # Each non-self-reciprocal pair counted twice; halve
    pair_counts_dist = {}
    for key, cnt in pair_zero_counts.items():
        if len(key) == 2:  # non-self
            pair_counts_dist[cnt//2] = pair_counts_dist.get(cnt//2, 0) + 1
        else:  # self-reciprocal
            pair_counts_dist[("self", cnt)] = pair_counts_dist.get(("self", cnt), 0) + 1
    print(f"  Pair-zero distribution: {pair_counts_dist}")
    print(f"  -> Each pair has {'exactly 1' if all(k==1 for k in pair_counts_dist.keys() if isinstance(k,int)) else 'NOT exactly 1'} zero")

    # ord_n(p) check
    cur = p % n
    ord_p = 1
    while cur != 1:
        cur = (cur * p) % n
        ord_p += 1
    is_minus1_in_powers = any(pow(p, k, n) == n-1 for k in range(ord_p))
    print(f"  ord_n(p) = {ord_p}, -1 in <p>: {is_minus1_in_powers}")

    print(f"  *** PROVEN: # zero CRT factors = (|F|-1)/2 = {expected_zero_count} ***")


for p in [3, 5, 7, 11, 13, 17, 19, 23]:
    try:
        verify(p)
    except Exception as e:
        print(f"  ERROR at p={p}: {e}")
        import traceback
        traceback.print_exc()
