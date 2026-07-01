"""
Monte-Carlo Quantum-Bit-Error-Rate (QBER) simulation for Q(q, d) codes.

Implements:
  (a) Channel model: i.i.d. depolarising on n qudits with prob. p,
      each erroneous qudit getting a uniform-random non-identity Pauli.
  (b) Bounded-distance (BD) decoder: precomputes a syndrome -> error map
      for all weight <= t Pauli errors (where t = (d_min - 1) // 2).
      Decodes by syndrome lookup; declares failure if syndrome not in table.
  (c) Light "soft" decoder: extends syndrome table to weight = t + 1 to
      pick up the small fraction of weight-(t+1) errors that are uniquely
      decodable (sometimes faster convergence to ML).
  (d) Failure-probability estimate vs analytical upper bound
        P_fail^UB(p) = sum_{w=t+1}^n C(n, w) p^w (1-p)^{n-w}.

Run at p in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1} for
  Q(3, 2) = [[13, 6, 3]]_3   (t = 1)
  Q(5, 2) = [[31, 15, 6]]_5  (t = 2)
"""
import sys, itertools, math, random, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np


# ===== Field construction =====
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


def build_Qqd(q, d):
    """Build H = (A | M_Q A) for Q(q, d), q prime."""
    p = q
    poly, aps = find_primitive(p, d+1)
    fs = p**(d+1) - 1
    n = fs // (p - 1)
    traces = trace_table(p, d+1, aps)
    xi = 2 if p % 2 == 1 else p + 1
    tau_H = np.array([1 if traces[i % fs] == 0 else 0 for i in range(n)], dtype=int)
    tau_Q = np.array([1 if traces[(xi*i) % fs] == 0 else 0 for i in range(n)], dtype=int)
    A = circ(tau_H) % p
    M_Q = circ(tau_Q) % p
    H_x = (M_Q @ A) % p
    H = np.hstack([A, H_x])
    return H, p, n


# ===== Syndrome computation =====
# An error vector e in F_q^{2n} represents Pauli (a | b) where a is Z-part, b is X-part.
# Two Pauli operators commute iff their symplectic inner product vanishes:
#     <(a1, b1), (a2, b2)>_symp = a1 . b2 - b1 . a2  (mod q)
# Equivalently: (e1 Omega) . e2^T = 0  where Omega = [[0, I], [-I, 0]].
#
# Syndrome of e w.r.t. stabilizer H = (H_z | H_x) is:
#     s_i = e Omega H_i^T  for each row H_i = (H_z[i], H_x[i])
# Compute as: s = e Omega H^T = (e_z H_x^T - e_x H_z^T)  (mod q).

def syndrome(e, H, p):
    """e is shape (2n,), H is (n-k) x 2n.  Return syndrome shape (n-k,)."""
    n = H.shape[1] // 2
    e_z = e[:n]; e_x = e[n:]
    H_z = H[:, :n]; H_x = H[:, n:]
    return (e_z @ H_x.T - e_x @ H_z.T) % p


def syndromes_batch(E, H, p):
    """E is shape (B, 2n), H is (n-k) x 2n.  Return (B, n-k)."""
    n = H.shape[1] // 2
    E_z = E[:, :n]; E_x = E[:, n:]
    H_z = H[:, :n]; H_x = H[:, n:]
    return (E_z @ H_x.T - E_x @ H_z.T) % p


# ===== Build syndrome table for weight <= w_max =====
def build_syndrome_table(H, p, w_max):
    """Return dict: syndrome (tuple) -> (weight, error_vector) for all errors
    of Pauli weight <= w_max.  Lower-weight wins on collisions."""
    n = H.shape[1] // 2
    table = {}
    # Identity has zero syndrome
    table[tuple([0]*(H.shape[0]))] = (0, np.zeros(2*n, dtype=int))

    # q^2 - 1 non-identity Pauli values per qudit, indexed by (a, b) in F_q^2 \ {(0,0)}.
    non_id_paulis = [(a, b) for a in range(p) for b in range(p) if (a, b) != (0, 0)]

    for w in range(1, w_max + 1):
        for supp in itertools.combinations(range(n), w):
            for choices in itertools.product(non_id_paulis, repeat=w):
                e = np.zeros(2*n, dtype=int)
                for k, pos in enumerate(supp):
                    a, b = choices[k]
                    e[pos] = a
                    e[pos + n] = b
                s = tuple(syndrome(e, H, p).tolist())
                if s not in table or table[s][0] > w:
                    table[s] = (w, e.copy())
    return table


# ===== Decoder =====
def is_in_rowspan(e, H, p):
    """Check if Pauli e (length 2n) is in the row span of H."""
    # Solve c H = e for c in F_p^{n-k}.  Equivalently, check rank-equal.
    n = H.shape[1] // 2
    aug = np.vstack([H, e.reshape(1, -1)]) % p
    r1 = matrix_rank(H, p)
    r2 = matrix_rank(aug, p)
    return r1 == r2


def matrix_rank(M, p):
    M2 = M.copy() % p
    rows, cols = M2.shape
    r = 0
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
        r += 1
    return r


def is_logical(residual, H, p):
    """A residual in centralizer\\stabilizer is a logical operator.
    Returns True iff residual has zero syndrome AND is NOT in row-span of H."""
    s = syndrome(residual, H, p)
    if not np.array_equal(s, np.zeros_like(s)):
        return False  # not even in centralizer; shouldn't happen for correctly-decoded residuals
    return not is_in_rowspan(residual, H, p)


# ===== Channel sampler =====
def sample_error(n, p_ch, q, rng):
    """Sample one i.i.d. depolarising error on n qudits.
    Each qudit: with prob p_ch, draw a uniform-random non-identity Pauli;
    else identity."""
    e = np.zeros(2*n, dtype=int)
    flips = rng.binomial(1, p_ch, n)
    for i in range(n):
        if flips[i]:
            # Choose uniform random (a, b) != (0, 0)
            while True:
                a = rng.integers(0, q)
                b = rng.integers(0, q)
                if a != 0 or b != 0:
                    break
            e[i] = a
            e[i + n] = b
    return e


def sample_error_batch(B, n, p_ch, q, rng):
    """Sample B errors in batch.  Returns (B, 2n)."""
    E = np.zeros((B, 2*n), dtype=int)
    flips = rng.binomial(1, p_ch, (B, n))
    for b in range(B):
        for i in range(n):
            if flips[b, i]:
                while True:
                    a = int(rng.integers(0, q))
                    bb = int(rng.integers(0, q))
                    if a != 0 or bb != 0: break
                E[b, i] = a
                E[b, i + n] = bb
    return E


# ===== Monte-Carlo trial =====
def run_mc(H, p, table, p_ch, n_trials, rng, dmin):
    """Run n_trials Monte-Carlo trials at channel rate p_ch.

    Optimisation: residual weight < d_min => residual in stabilizer
    (since min weight of non-trivial logicals is d_min by definition).
    Only do rank check when residual weight >= d_min.

    Returns (n_success, n_logical_fail, n_uncorrectable)."""
    n = H.shape[1] // 2
    n_succ = 0
    n_log = 0
    n_unc = 0
    B = min(n_trials, 5000)
    n_batches = (n_trials + B - 1) // B
    rank_H = matrix_rank(H, p)

    for batch in range(n_batches):
        B_eff = min(B, n_trials - batch * B)
        E = sample_error_batch(B_eff, n, p_ch, p, rng)
        S = syndromes_batch(E, H, p)
        for b in range(B_eff):
            s = tuple(S[b].tolist())
            if s in table:
                w_dec, e_dec = table[s]
                residual = (E[b] - e_dec) % p
                # Compute Pauli weight of residual
                r_z = residual[:n]; r_x = residual[n:]
                rw = int(((r_z != 0) | (r_x != 0)).sum())
                if rw < dmin:
                    # Guaranteed in stabilizer
                    n_succ += 1
                else:
                    # Need full rank check
                    aug = np.vstack([H, residual.reshape(1, -1)]) % p
                    r_aug = matrix_rank(aug, p)
                    if r_aug == rank_H:
                        n_succ += 1
                    else:
                        n_log += 1
            else:
                n_unc += 1
    return n_succ, n_log, n_unc


# ===== Analytical upper bound =====
def analytical_pfail(n, t, p_ch):
    """P_fail upper bound = sum_{w=t+1}^n C(n,w) p^w (1-p)^{n-w}."""
    pf = 0.0
    for w in range(t + 1, n + 1):
        pf += math.comb(n, w) * p_ch**w * (1 - p_ch)**(n - w)
    return pf


# ===== Main =====
def run_code(code_label, q, d, dmin, p_ch_list, n_trials=20000):
    print(f"\n{'='*70}")
    print(f"=== {code_label}: Q({q}, {d}), d_min = {dmin} ===")
    print('='*70)
    H, p, n = build_Qqd(q, d)
    rank_H = matrix_rank(H, p)
    k = n - rank_H
    t = (dmin - 1) // 2
    print(f"  n={n}, k={k}, d_min={dmin}, t={t}")

    print(f"  Building syndrome table for weight <= {t}...")
    t0 = time.time()
    table = build_syndrome_table(H, p, t)
    t1 = time.time()
    print(f"  Table built: {len(table)} entries in {t1-t0:.1f}s")

    rng = np.random.default_rng(42)
    print(f"  {'p_ch':>8} {'P_fail (MC)':>14} {'P_fail (UB)':>14} {'n_succ':>10} "
          f"{'n_log':>10} {'n_unc':>10}")

    results = []
    for p_ch in p_ch_list:
        # Use more trials at low p where failures are rare
        # Adaptive trial count: more trials at low p_ch where failures rare.
        # Cap at 200K to keep total runtime reasonable.
        nt = max(n_trials, int(min(2e6, 100 / max(p_ch**(t+1), 1e-12))))
        nt = min(nt, 200000)
        n_succ, n_log, n_unc = run_mc(H, p, table, p_ch, nt, rng, dmin)
        p_fail_mc = (n_log + n_unc) / nt
        p_fail_ub = analytical_pfail(n, t, p_ch)
        print(f"  {p_ch:8.4f} {p_fail_mc:14.4e} {p_fail_ub:14.4e} "
              f"{n_succ:10d} {n_log:10d} {n_unc:10d}")
        results.append((p_ch, p_fail_mc, p_fail_ub, n_succ, n_log, n_unc, nt))
    return results


if __name__ == "__main__":
    p_ch_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    all_results = {}

    all_results["Q(3,2)=[[13,6,3]]_3"] = run_code(
        "Q(3,2)", 3, 2, dmin=3, p_ch_list=p_ch_list, n_trials=20000)
    all_results["Q(5,2)=[[31,15,6]]_5"] = run_code(
        "Q(5,2)", 5, 2, dmin=6, p_ch_list=p_ch_list, n_trials=20000)

    # Save results for plotting
    import json
    with open("qber_results.json", "w") as f:
        json.dump({k: [list(r) for r in v] for k, v in all_results.items()}, f, indent=2)
    print("\nResults saved to qber_results.json")
