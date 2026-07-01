"""
Bar chart of per-logical-qudit failure probability across codes at three
representative channel rates.  Visualises Table X (per-logical-qudit
comparison) graphically. Renders as Fig. 5 in the paper.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# From Table X: P_fail / k for each code at three channel rates
codes = [
    ('Uncoded',                   1.0e-4, 1.0e-3, 1.0e-2),
    (r'Steane $[[7,1,3]]_2$',     2.1e-7, 2.1e-5, 2.1e-3),
    (r'Gandhi $[[21,11,3]]_2$',   1.9e-7, 1.9e-5, 1.9e-3),
    (r'$\mathcal{Q}(3,2)$=$[[13,6,3]]_3$', 1.3e-7, 1.3e-5, 1.2e-3),
    (r'$\mathcal{Q}(5,2)$=$[[31,15,6]]_5$', 3.0e-10, 3.0e-7, 2.4e-4),
]

p_ch_labels = [r'$p_{\mathrm{ch}}=10^{-4}$',
               r'$p_{\mathrm{ch}}=10^{-3}$',
               r'$p_{\mathrm{ch}}=10^{-2}$']

fig, ax = plt.subplots(1, 1, figsize=(11, 7.5))

n_codes = len(codes)
n_groups = len(p_ch_labels)
bar_width = 0.16
x = np.arange(n_groups)

colors = ['gray', 'C2', 'C3', 'C0', 'C1']
labels = [c[0] for c in codes]
values = np.array([[c[1], c[2], c[3]] for c in codes])

for i in range(n_codes):
    offset = (i - (n_codes - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values[i], bar_width,
                  label=labels[i],
                  color=colors[i],
                  edgecolor='black', linewidth=1.4,
                  alpha=0.88, log=True, zorder=3)
    # Annotate Q(5,2) bars with advantage ratio over Steane
    if i == 4:  # Q(5,2)
        steane = values[1]
        for j, b in enumerate(bars):
            ratio = steane[j] / values[i][j]
            ax.annotate(rf'${int(round(ratio))}\times$',
                        xy=(b.get_x() + b.get_width()/2, values[i][j]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=17, fontweight='bold',
                        color='C1')

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(p_ch_labels, fontsize=18)
ax.set_ylabel(r'Per-logical-qudit failure rate $P_{\mathrm{fail}}/k$',
              fontsize=20)
ax.set_title(r'Per-logical-qudit failure probability: $\mathcal{Q}(5,2)$ vs. Steane / Gandhi',
             fontsize=17)
ax.legend(loc='upper left', fontsize=14, framealpha=0.92)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylim(1e-11, 1e-1)
ax.grid(False)

# Add a note about the advantage
ax.text(0.02, 0.02,
        r'Numbers above $\mathcal{Q}(5,2)$ bars: advantage over Steane',
        transform=ax.transAxes, fontsize=14, color='C1',
        style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig("advantage_plot.pdf", dpi=300, bbox_inches='tight')
plt.savefig("advantage_plot.png", dpi=200, bbox_inches='tight')
print("Saved advantage_plot.pdf and advantage_plot.png")
