"""
Q(7,2) weight-7 search — V4 (FAST canonical check).

Two key optimisations over V3:
  (1) Iterate only subsets with 0 in S  (C(56,6) = 32M instead of 264M).
  (2) Smart canonical check: only 20 group elements matter (6 shifts that
      could put another element at 0, times 3 Frobenius elements, minus
      identity).  Versus the brute 170 transformations.

Reasoning for (2): canonical S contains 0 (=S[0]).  For S to be canonical,
g(S) >=_lex S for all g != id.  For g not putting any S element at 0,
sorted(g(S))[0] > 0 = S[0], so g(S) > S lex automatically.  The only
threatening g's are those putting some S[i] at 0, i.e., shift by -S[i]
mod n, combined with any Frobenius power.  That's |S| - 1 = 6 nontrivial
shifts (S[0]=0 is identity shift), times 3 Frobenius = 18, plus 2 pure
Frobenius (s=0, k=1,2) = 20 transformations.

Combined speedup vs V2: 8x (subset reduction) * 8.5x (canonical check) ~ 70x.
Expected runtime: ~30 min.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import itertools, time, random
import numpy as np

def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


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

def circ(c):
    n = len(c)
    M = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n): M[i, j] = c[(i - j) % n]
    return M

def right_kernel_basis(M, p):
    M2 = (M.copy().astype(int) % p)
    rows, cols = M2.shape
    pivot_col = []; pivots = set(); r = 0
    for c in range(cols):
        pr = None
        for i in range(r, rows):
            if M2[i, c] % p != 0: pr = i; break
        if pr is None: continue
        if pr != r: M2[[r, pr]] = M2[[pr, r]]
        inv = pow(int(M2[r, c]) % p, -1, p)
        M2[r] = (M2[r] * inv) % p
        for i in range(rows):
            if i != r and M2[i, c] % p != 0:
                M2[i] = (M2[i] - M2[i, c] * M2[r]) % p
        pivot_col.append(c); pivots.add(c); r += 1
    free = [c for c in range(cols) if c not in pivots]
    K = np.zeros((cols, len(free)), dtype=int)
    for k, fc in enumerate(free):
        v = np.zeros(cols, dtype=int); v[fc] = 1
        for ri, pc in enumerate(pivot_col):
            coef = int(M2[ri, fc]) % p
            if coef: v[pc] = (-coef) % p
        K[:, k] = v
    return K


# Q(7,2) setup
q, d = 7, 2
poly, aps = find_primitive(q, d+1)
fs = q**(d+1) - 1
n = fs // (q-1)
traces = trace_table(q, d+1, aps)
tau_H = np.array([1 if traces[i % fs] == 0 else 0 for i in range(n)], dtype=int)
tau_Q = np.array([1 if traces[(2*i) % fs] == 0 else 0 for i in range(n)], dtype=int)
A = circ(tau_H) % q
M_Q = circ(tau_Q) % q
H_x = (M_Q @ A) % q
H = np.hstack([A, H_x])
M_cent = np.hstack([H_x, (-A) % q]) % q
K_S = right_kernel_basis(H, q)

fprint(f"Q({q},{d}): n={n}, primitive_poly={poly}")

ord_q = 3
q_pow = [1, q % n, (q*q) % n]  # q^0, q^1, q^2 mod n


def is_canonical_v4(S):
    """S is a sorted tuple of length 7, with S[0] = 0.  Test whether S is
    lex-min in its Singer-Frobenius orbit by checking only the 20 group
    elements that could produce sorted g(S)[0] = 0."""
    n_local = n
    # Pure Frobenius (s=0, k=1, 2)
    for k in range(1, ord_q):
        qk = q_pow[k]
        TS = sorted(qk * sj % n_local for sj in S)
        if TS[0] < S[0]:
            return False
        if TS[0] > S[0]:
            pass
        else:
            if tuple(TS) < S:
                return False
    # Shift by -S[i] (i=1..6) puts S[i] at 0; apply k=0, 1, 2
    for i in range(1, 7):
        si = S[i]
        s_shift = (-si) % n_local
        for k in range(ord_q):
            qk = q_pow[k]
            TS = sorted((qk * (sj + s_shift)) % n_local for sj in S)
            if TS[0] < S[0]:
                return False
            if TS[0] > S[0]:
                continue
            if tuple(TS) < S:
                return False
    return True


def witness_exists_and_construct(S, K_loc, w):
    d_loc = K_loc.shape[1]
    if d_loc == 0:
        return False, None
    full_support = True
    for idx in range(w):
        if not ((K_loc[idx] != 0).any() or (K_loc[idx + w] != 0).any()):
            full_support = False
            break
    if not full_support:
        return False, None
    rng = random.Random(42)
    for trial in range(200):
        coefs = np.array([rng.randint(0, q-1) for _ in range(d_loc)], dtype=int)
        if all(c == 0 for c in coefs): continue
        v_loc = (K_loc @ coefs) % q
        a_part_S = v_loc[:w]; b_part_S = v_loc[w:2*w]
        ab = np.zeros(2*n, dtype=int)
        for k_idx, pos in enumerate(S):
            ab[pos] = a_part_S[k_idx]
            ab[pos + n] = b_part_S[k_idx]
        a_full = ab[:n]; b_full = ab[n:]
        pw = int(((a_full != 0) | (b_full != 0)).sum())
        if pw != w: continue
        stab = (ab @ K_S) % q
        if not (stab == 0).all():
            return True, ab
    if d_loc <= 6:
        for coefs in itertools.product(range(q), repeat=d_loc):
            if all(c==0 for c in coefs): continue
            v_loc = (K_loc @ np.array(coefs)) % q
            a_part_S = v_loc[:w]; b_part_S = v_loc[w:2*w]
            ab = np.zeros(2*n, dtype=int)
            for k_idx, pos in enumerate(S):
                ab[pos] = a_part_S[k_idx]
                ab[pos + n] = b_part_S[k_idx]
            a_full = ab[:n]; b_full = ab[n:]
            pw = int(((a_full != 0) | (b_full != 0)).sum())
            if pw != w: continue
            stab = (ab @ K_S) % q
            if not (stab == 0).all():
                return True, ab
        return False, None
    else:
        return False, None


def search(w):
    fprint(f"\n--- Q({q},{d}) Singer-Frobenius w={w} V4 (0-rooted, fast canon) ---")
    n_canonical = 0
    n_kernel_nonempty = 0
    n_high_d_loc = 0
    n_total = 0
    n_total_target = 32468436
    flagged = []
    t0 = time.time()
    last_progress_t = t0

    for S_tail in itertools.combinations(range(1, n), w - 1):
        S = (0,) + S_tail
        n_total += 1
        if not is_canonical_v4(S):
            if n_total % 1000000 == 0:
                now = time.time()
                fprint(f"  [w={w}]  total {n_total:,}/{n_total_target:,} "
                       f"({100*n_total/n_total_target:.1f}%), "
                       f"{n_canonical:,} canonical, "
                       f"{n_kernel_nonempty} kernel, elapsed {now-t0:.0f}s")
            continue
        n_canonical += 1

        cols = list(S) + [pp + n for pp in S]
        M_S = M_cent[:, cols]
        K_loc = right_kernel_basis(M_S, q)
        d_loc = K_loc.shape[1]
        if d_loc > 0:
            n_kernel_nonempty += 1
            if d_loc > 6:
                n_high_d_loc += 1
                flagged.append((S, d_loc))
            found, ab = witness_exists_and_construct(S, K_loc, w)
            if found:
                elapsed = time.time() - t0
                a_full = ab[:n]; b_full = ab[n:]
                supp = [i for i in range(n) if a_full[i] or b_full[i]]
                fprint(f"  *** w={w} WITNESS in {elapsed:.1f}s")
                fprint(f"     support: {supp}")
                fprint(f"     a (Z): {a_full[supp]}")
                fprint(f"     b (X): {b_full[supp]}")
                return w, ab

        if n_canonical % 50000 == 0:
            now = time.time()
            elapsed = now - t0
            since_last = now - last_progress_t
            last_progress_t = now
            fprint(f"  [w={w}]  {n_canonical:,} canonical / "
                   f"{n_total:,} ({100*n_total/n_total_target:.1f}%) total, "
                   f"{n_kernel_nonempty} kernel, {n_high_d_loc} high-d_loc, "
                   f"elapsed {elapsed:.0f}s, last 50K canonical in {since_last:.1f}s")

    elapsed = time.time() - t0
    fprint(f"\n  [w={w}]  EXHAUSTIVE done: {n_total:,} subsets / "
           f"{n_canonical:,} canonical orbits in {elapsed:.0f}s")
    fprint(f"  Kernel-nonempty: {n_kernel_nonempty}, high-d_loc (>6): {n_high_d_loc}")
    if n_high_d_loc > 0:
        fprint(f"  Flagged orbits for separate verification:")
        for S, dl in flagged[:20]:
            fprint(f"    S={list(S)}, d_loc={dl}")
        if len(flagged) > 20:
            fprint(f"    ... ({len(flagged) - 20} more)")
    return None, None


for w in [7]:
    fprint(f"\n=== Searching weight {w} ===")
    res, ab = search(w)
    if res is not None:
        fprint(f"\n>>> d_min(Q(7,2)) <= {res}")
        break
else:
    fprint(f"\n>>> d_min(Q(7,2)) >= 8 (proven via V4 0-rooted exhaustive)")
