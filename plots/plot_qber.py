"""
Generate publication-quality QBER plot from monte_carlo_qber.py results.
Output: qber_plot.pdf and qber_plot.png
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


with open("qber_results.json") as f:
    results = json.load(f)


def analytical_pfail(n, t, p_ch):
    return sum(math.comb(n, w) * p_ch**w * (1 - p_ch)**(n - w)
               for w in range(t + 1, n + 1))


# Set up figure with two subplots (one per code) plus uncoded reference
# Larger figure size to give the larger fonts room to breathe
fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

# Uncoded baseline:  P_fail = 1 - (1-p)^n for some reference n
p_dense = np.logspace(-4, -0.5, 100)


# Q(3,2) data
data32 = results["Q(3,2)=[[13,6,3]]_3"]
p_arr = np.array([d[0] for d in data32])
mc32 = np.array([d[1] for d in data32])
ub32 = np.array([d[2] for d in data32])
ub32_dense = np.array([analytical_pfail(13, 1, p) for p in p_dense])

# Q(5,2) data
data52 = results["Q(5,2)=[[31,15,6]]_5"]
mc52 = np.array([d[1] for d in data52])
ub52 = np.array([d[2] for d in data52])
ub52_dense = np.array([analytical_pfail(31, 2, p) for p in p_dense])

# Analytical reference curves for prior-art codes
steane_dense  = np.array([analytical_pfail(7,  1, p) for p in p_dense])
gandhi_dense  = np.array([analytical_pfail(21, 1, p) for p in p_dense])

# Uncoded reference: 1 qudit
uncoded_dense = p_dense  # P_fail = p for single qudit

# Plot upper-bound curves (continuous) — thicker lines, larger markers
ax.loglog(p_dense, uncoded_dense, '--', color='gray', linewidth=2.8,
          label=r'Uncoded baseline', zorder=1)
ax.loglog(p_dense, steane_dense, ':', color='C2', linewidth=3.0,
          label=r'Steane $[[7,1,3]]_2$ (analytical)', zorder=2)
ax.loglog(p_dense, gandhi_dense, ':', color='C3', linewidth=3.0,
          label=r'Gandhi $[[21,11,3]]_2$ (analytical)', zorder=2)
ax.loglog(p_dense, ub32_dense, '-', color='C0', linewidth=3.2,
          label=r'$\mathcal{Q}(3,2)$ analytical UB', zorder=3)
ax.loglog(p_dense, ub52_dense, '-', color='C1', linewidth=3.2,
          label=r'$\mathcal{Q}(5,2)$ analytical UB', zorder=4)

# Plot MC points (substitute MC = UB when MC = 0 for log-scale visibility)
mc32_plot = np.where(mc32 > 0, mc32, np.nan)
mc52_plot = np.where(mc52 > 0, mc52, np.nan)
ax.loglog(p_arr, mc32_plot, 'o', color='C0', markersize=15,
          markerfacecolor='white', markeredgewidth=2.6,
          label=r'$\mathcal{Q}(3,2)$ Monte-Carlo', zorder=5)
ax.loglog(p_arr, mc52_plot, 's', color='C1', markersize=15,
          markerfacecolor='white', markeredgewidth=2.6,
          label=r'$\mathcal{Q}(5,2)$ Monte-Carlo', zorder=6)

# Labels and aesthetics — bigger fonts, NO grid (per L. Hanzo)
ax.set_xlabel(r'Depolarising channel rate $p_{\mathrm{ch}}$',
              fontsize=20)
ax.set_ylabel(r'Decoding failure probability $P_{\mathrm{fail}}$',
              fontsize=20)
ax.set_title(r'QBER curves: bounded-distance decoder on $\mathcal{Q}(q,2)$ codes',
             fontsize=18)
ax.legend(loc='lower right', fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)
# Grid removed per L. Hanzo's request
ax.grid(False)
ax.set_xlim(p_dense[0], p_dense[-1])
ax.set_ylim(1e-9, 1.0)

# Add annotation for the Q(5,2) advantage
ax.annotate(r'$\sim 70\times$ better than Steane' + '\n' + r'per logical qudit',
            xy=(1e-3, 4.4e-6), xytext=(1.4e-4, 1e-7),
            fontsize=16, color='C1', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='C1',
                             alpha=0.7, lw=2.0))

plt.tight_layout()
plt.savefig("qber_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig("qber_plot.png", dpi=200, bbox_inches='tight')
print("Saved qber_plot.pdf and qber_plot.png")

# Also print a comparison table
print("\nDetailed comparison table:")
print(f"{'p_ch':>10} {'Q(3,2) MC':>13} {'Q(3,2) UB':>13} {'Q(5,2) MC':>13} {'Q(5,2) UB':>13}")
for i, p in enumerate(p_arr):
    print(f"{p:10.4f} {mc32[i]:13.4e} {ub32[i]:13.4e} {mc52[i]:13.4e} {ub52[i]:13.4e}")
