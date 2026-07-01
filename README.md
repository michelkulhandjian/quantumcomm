# Singer-Difference-Set Qudit Stabilizer Codes from Non-Degenerate Quadrics in PG(d,q)

This repository contains all numerical verification scripts and simulation
tooling for the paper:

> M. Kulhandjian and L. Hanzo,
> *"Singer-Difference-Set Qudit Stabilizer Codes from Non-Degenerate
> Quadrics in PG(d,q): Construction, Structural Theorems, and Monte-Carlo
> Performance,"*
> IEEE Open Journal of the Communications Society, 2026 (accepted).

The scripts here reproduce every numerical claim in the paper, including
the code parameters $[[n,k,d_{\min}]]_q$ of every $\mathcal Q(q,d)$ instance,
the numerical verification of every structural theorem, and the
Monte-Carlo QBER curves.

---

## Contents

### Code-parameter computation and distance search

| Script | Purpose | Paper reference |
|---|---|---|
| `compute_dmin_q3d3.py` | Numerical centralizer / row-span check of $P_2 = Z_0 Z_{20}^{2}$ for $\mathcal Q(3,3)$; confirms $d_{\min}(\mathcal Q(3,3)) = 2$. | §VII, Proposition 17 |
| `compute_dmin_general.py` | Exhaustive low-weight Pauli search for $d_{\min}(\mathcal Q(q,d))$; produces Table VI. | §VII-E |
| `compute_dmin_isd_q5d2.py` | Information-set-decoding-style support-restriction search; settles $d_{\min}(\mathcal Q(5,2)) = 6$. | §VII-F |
| `compute_dmin_isd_q7q11.py` | ISD search extended to $\mathcal Q(7,2)$ and $\mathcal Q(11,2)$. | §VII-G |
| `compute_dmin_singerfrob_q72.py` | Singer-Frobenius canonical-form search over $|G| = 171$ orbits of $\mathcal Q(7,2)$. | §VII-H |
| `compute_dmin_w7_v4.py` | 0-rooted Singer-Frobenius exhaustive search at weight 7 for $\mathcal Q(7,2)$; proves $d_{\min} \geq 8$. | Theorem 28 |
| `compute_dmin_qeven.py` | Distance search for $\mathcal Q(q,d)$ instances at $q$ even and $d$ even. | §VII-E, Proposition 23 |

### Construction and structural-theorem verification

| Script | Purpose | Paper reference |
|---|---|---|
| `construct_clifford_d2.py` | Constructs the explicit palindromic $\mathbf m$ and Clifford gate $U_M = H^{\otimes n} \mathrm{CZ}^{(M)} H^{\otimes n}$; verified at $(q,d) \in \{(3,2), (5,2), (7,2)\}$. | Theorem 19 |
| `verify_lds_columns_centralizer.py` | Falsifies the LDS-columns-in-centralizer conjecture at $(3,2)$ and proves the rank-completion result. | §VI-D, Propositions 20 and 21 |
| `verify_tau_H.py` | Builds the Singer first column $\boldsymbol\tau_H$ and verifies $A A^\top = 9 I_{40} + 4 J_{40}$. | §VII-B |
| `verify_hamada_rank.py` | Verifies the Hamada-Smith rank formula $\rank_{F_p}(A) = \binom{p+d-1}{d} + 1$ on 8 $(p,d)$ cases. | Lemma 8, Table III |
| `verify_conjecture_cross_corr.py` | Verifies the cross-correlation modular-constancy theorem $\nu[k] \equiv 1 \pmod q$ at $(q,d) \in \{(3,3),(5,3),(7,3),(3,5)\}$. | Theorem 13, Table IV |
| `verify_tau_zero_proof.py` | Verifies the CRT zero-count theorem $\#\{f : \tau_{H,f} = 0\} = (\|\mathcal F\| - 1)/2$ at $p \in \{3, 5, 7, 11, 13, 17, 19, 23\}$. | Theorem 30 |
| `verify_bch_count.py` | Numerical study of the original (falsified) BCH-count conjecture; motivates the CRT zero-count theorem. | Remark 20 |
| `verify_q2_ceiling.py` | Verifies the $q = 2$ ceiling $u = M_Q \boldsymbol\tau_H \equiv \mathbf 1 \pmod 2$ at $d \in \{2, 3, 4, 5, 6\}$. | Proposition 23 |

### Waterloo-defect and negative-result diagnostics

| Script | Purpose | Paper reference |
|---|---|---|
| `compute_q3d3.py`, `compute_q3d3_deficit.py`, `compute_q3d3_variants.py` | Symplectic-deficit rank computation for the $V_1$, $V_6$, $V_7$ recipes at $(q,d) \in \{(3,2), (3,3), (2,3)\}$. | §IV-C, Table I |
| `compute_gf4_hermitian.py` | GF(4)/Hermitian lifting attempt; documents the negative result. | §IV-F, Table II |
| `compute_waterloo_correct.py`, `compute_waterloo_defect_Z.py` | $\mathbb Z$-coefficient lifting attempt; documents the negative result. | §IV-E |

### Monte-Carlo simulation and plotting

| Script | Purpose | Paper reference |
|---|---|---|
| `monte_carlo_qber.py` | Bounded-distance syndrome-table decoder Monte-Carlo simulation over $\sim 1.5 \times 10^6$ trials for $\mathcal Q(3,2)$ and $\mathcal Q(5,2)$; produces `qber_results.json`. | §IX-F, Table XI |
| `plot_qber.py` | Renders Fig. 3 (QBER curves) from `qber_results.json`. | Fig. 3 |
| `plot_code_params.py` | Renders Fig. 4 (code-parameter scatter plot) from Tables VI and VIII. | Fig. 4 |
| `plot_advantage.py` | Renders Fig. 5 (per-logical-qudit advantage bar chart) from Table X. | Fig. 5 |

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/mkulhandjian/quantumcomm.git
cd quantumcomm

# 2. Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# 3. Reproduce the QBER simulation (takes ~5 minutes on a laptop)
python monte_carlo_qber.py     # produces qber_results.json
python plot_qber.py            # renders Fig. 3 as qber_plot.pdf/.png

# 4. Verify a specific theorem — for example, the CRT zero-count theorem
python verify_tau_zero_proof.py

# 5. Reproduce the heavy exhaustive proof d_min(Q(7,2)) >= 8 (~33 minutes)
python compute_dmin_w7_v4.py
```

Each script prints a self-contained summary of what it verifies, so no
additional documentation is needed to interpret the output. The Monte-Carlo
script `monte_carlo_qber.py` is the only one that emits an intermediate
data file (`qber_results.json`); all others print their results to stdout.

---

## Dependencies

Everything is written in pure Python + NumPy + Matplotlib. No SageMath,
Magma, or GAP dependencies. See `requirements.txt` for exact versions.

The exhaustive $\mathcal Q(7,2)$ search
(`compute_dmin_w7_v4.py`) runs in ~33 minutes on a single core. All
other scripts complete in under 5 minutes on a laptop. No GPU is required.

---

## Citation

If you use this code or refer to the theoretical results, please cite:

```bibtex
@article{KulhandjianHanzo2026,
  author  = {Kulhandjian, Michel and Hanzo, Lajos},
  title   = {Singer-Difference-Set Qudit Stabilizer Codes from
             Non-Degenerate Quadrics in {$\PG(d,q)$}: Construction,
             Structural Theorems, and {M}onte-{C}arlo Performance},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026},
  note    = {accepted for publication},
  url     = {https://github.com/mkulhandjian/quantumcomm}
}
```

---

## License

MIT License — see `LICENSE` for the full text.  You may use, modify, and
redistribute the code freely, subject to preservation of the copyright
notice and the citation above in derivative work.

---

## Authors

- **Michel Kulhandjian** (mkulhandjian@outlook.com) — Rice University, USA
- **Lajos Hanzo** (lh@ecs.soton.ac.uk) — University of Southampton, UK

For questions about the theorems or code, please open an issue on this
repository. Substantive scientific correspondence is best directed to the
first author.
