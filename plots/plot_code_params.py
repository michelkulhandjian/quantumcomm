"""
Scatter plot of (n, k) for all Q(q,d) instances and prior-art codes,
with d_min colour-coded and code family marker-coded.

Source: Table VI (tab:qd-instances) + Table VIII (tab:comparison-params).
Renders as Fig. 4 in the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# (n, k, d_min, family, label)  -- d_min=None means "ceiling = 2"
codes = [
    # Steane / Gandhi qubit codes
    (7,   1,  3, 'steane',  r'Steane $[[7,1,3]]_2$'),
    (21,  11, 3, 'gandhi',  r'Gandhi $[[21,11,3]]_2$'),
    (73,  45, 5, 'gandhi',  r'Gandhi $[[73,45,5]]_2$'),
    (273, 191, 11,'gandhi', r'Gandhi $[[273,191,11]]_2$'),
    # Tang-Bai-Feng qubit codes
    (18,  10, 2, 'tbf',     r'TBF $[[18,10,2]]_2$'),
    (27,  1,  5, 'tbf',     r'TBF $[[27,1,5]]_2$'),
    # Our Q(q,d) family
    (7,   3,  2, 'ours-q2', r'$\mathcal{Q}(2,2)$'),
    (15,  10, 2, 'ours-q2', r'$\mathcal{Q}(2,3)$'),
    (31,  25, 2, 'ours-q2', r'$\mathcal{Q}(2,4)$'),
    (13,  6,  3, 'ours',    r'$\mathcal{Q}(3,2)$'),
    (40,  29, 2, 'ours-ceil', r'$\mathcal{Q}(3,3)^{\dagger}$'),
    (121, 105,3, 'ours',    r'$\mathcal{Q}(3,4)$'),
    (31,  15, 6, 'ours',    r'$\mathcal{Q}(5,2)$'),
    (57,  28, 8, 'ours',    r'$\mathcal{Q}(7,2)$'),
    (133, 66, 5, 'ours',    r'$\mathcal{Q}(11,2)$'),
]

fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

family_style = {
    'steane':    dict(marker='D', s=320, c='C2',     label='Steane [3]'),
    'gandhi':    dict(marker='^', s=320, c='C3',     label='Gandhi [5]'),
    'tbf':       dict(marker='v', s=320, c='C4',     label='Tang-Bai-Feng [6]'),
    'ours':      dict(marker='o', s=380, c='C1',     label=r'$\mathcal{Q}(q,d)$ (this paper)'),
    'ours-q2':   dict(marker='o', s=320, c='gray',   label=r'$\mathcal{Q}(2,d)$ ($q{=}2$ ceiling, $d_{\min}{=}2$)'),
    'ours-ceil': dict(marker='o', s=320, c='C0',     label=r'$\mathcal{Q}(q,d)$ (2-torsion ceiling)'),
}

# Plot points
for n, k, dmin, fam, lbl in codes:
    style = family_style[fam]
    ax.scatter(n, k, marker=style['marker'], s=style['s'],
               edgecolors='black', linewidths=2.0,
               facecolors=style['c'], alpha=0.88, zorder=3)
    # annotate d_min in/around the marker (only for ours)
    if fam.startswith('ours') and fam != 'ours-q2':
        ax.annotate(f'$d{{=}}{dmin}$', xy=(n, k), xytext=(9, 9),
                    textcoords='offset points', fontsize=15,
                    fontweight='bold', color='black', zorder=4)

# Rate-1/2 reference line
n_dense = np.linspace(1, 300, 100)
ax.plot(n_dense, n_dense/2, ':', color='gray', linewidth=2.2,
        alpha=0.65, zorder=1, label='Rate-1/2 line ($k = n/2$)')
# Rate-1 line (uncoded)
ax.plot(n_dense, n_dense, '--', color='gray', linewidth=1.6,
        alpha=0.45, zorder=1, label='Rate-1 line (uncoded, $k=n$)')

# Build legend
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='C1', markeredgecolor='black', markersize=18,
           label=r'$\mathcal{Q}(q,d)$ (this paper)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='C0', markeredgecolor='black', markersize=16,
           label=r'$\mathcal{Q}(q,d)$ 2-torsion ceiling ($\dagger$)'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='gray', markeredgecolor='black', markersize=16,
           label=r'$\mathcal{Q}(2,d)$ qubit ceiling'),
    Line2D([0], [0], marker='D', color='w',
           markerfacecolor='C2', markeredgecolor='black', markersize=16,
           label=r'Steane [3]'),
    Line2D([0], [0], marker='^', color='w',
           markerfacecolor='C3', markeredgecolor='black', markersize=16,
           label=r'Gandhi [5]'),
    Line2D([0], [0], marker='v', color='w',
           markerfacecolor='C4', markeredgecolor='black', markersize=16,
           label=r'Tang-Bai-Feng [6]'),
    Line2D([0], [0], linestyle=':', color='gray', linewidth=2.2,
           label='Rate-1/2 line'),
]

ax.set_xlabel(r'Block length $n$', fontsize=20)
ax.set_ylabel(r'Logical dimension $k$', fontsize=20)
ax.set_title(r'Code parameters $[[n,k,d_{\min}]]$ across constructions',
             fontsize=18)
ax.legend(handles=legend_handles, loc='upper left', fontsize=14,
          framealpha=0.92)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlim(0, 290)
ax.set_ylim(0, 210)
ax.grid(False)

plt.tight_layout()
plt.savefig("code_params_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig("code_params_plot.png", dpi=200, bbox_inches='tight')
print("Saved code_params_plot.pdf and code_params_plot.png")
