"""
Figure 3: Side-by-side instruction count comparison — epilogue strategies
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                         gridspec_kw={"width_ratios": [1.6, 1]})
fig.patch.set_facecolor("white")

# ── Left: grouped bar chart ────────────────────────────────────────────────
ax = axes[0]

strategies = ["Naive\n(scalar ushort)", "XOR-vectorized\n(dword pairs)"]
n_instr    = [4, 2]   # store instructions per thread
bytes_each = [2, 4]   # bytes per instruction
# total bytes per thread = 8 B in both cases

x = np.array([0.0, 1.2])
width = 0.38

bar_instr = ax.bar(x - width / 2, n_instr, width,
                   label="# store instructions / thread",
                   color=["#C44E52", "#4C72B0"], edgecolor="black", lw=1.2,
                   zorder=3)
bar_bytes  = ax.bar(x + width / 2, bytes_each, width,
                    label="bytes per instruction",
                    color=["#FFAAAA", "#AEC6E8"], edgecolor="black", lw=1.2,
                    zorder=3, hatch="///")

ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11, fontweight="bold")
ax.set_ylabel("Count / Bytes", fontsize=10)
ax.set_ylim(0, 5.5)
ax.set_title("Store instructions per thread (per 8 bytes output)",
             fontsize=11, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_facecolor("white")

# Value labels
for bar in list(bar_instr) + list(bar_bytes):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
            str(int(h)), ha="center", va="bottom", fontsize=12,
            fontweight="bold", color="black")

# Annotations
ax.annotate(
    "2× reduction\nin instruction count",
    xy=(x[1] - width / 2, n_instr[1]),
    xytext=(x[1] + 0.45, n_instr[1] + 1.4),
    fontsize=9, color="#1A6696", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#1A6696", lw=1.5)
)
ax.annotate(
    "2× wider per store\n(better coalescing)",
    xy=(x[1] + width / 2, bytes_each[1]),
    xytext=(x[1] + 0.45, bytes_each[1] + 1.2),
    fontsize=9, color="#28723A", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#28723A", lw=1.5)
)

ax.legend(fontsize=9, loc="upper left")

# ── Right: summary table ───────────────────────────────────────────────────
ax2 = axes[1]
ax2.axis("off")

col_labels = ["Strategy", "# Stores", "Width", "Total\nBytes", "Coalescing"]
rows = [
    ["Naive\n(ushort)",         "4",  "2 B",  "8 B", "scalar\n(stride 16)"],
    ["XOR\n(dword pairs)",      "2",  "4 B",  "8 B", "2×bf16\ncontiguous"],
    ["Permlane\n(transposed)",  "2",  "4 B",  "8 B", "row-\ncontiguous"],
]

cell_colors = [
    ["#F5F5F5", "#F9CBCB", "#F9CBCB", "#EAF4EA", "#F9CBCB"],
    ["#EAF0FB", "#D4E6F1", "#EAF4EA", "#EAF4EA", "#D4EDDA"],
    ["#F3EAF9", "#D4E6F1", "#EAF4EA", "#EAF4EA", "#D4EDDA"],
]

table = ax2.table(
    cellText=rows,
    colLabels=col_labels,
    cellColours=cell_colors,
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.12, 1.0, 0.82]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#999999")
    cell.set_linewidth(0.8)
    if row == 0:
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#2C3E50")

ax2.set_title("Epilogue strategy summary", fontsize=11, fontweight="bold", pad=8)
ax2.text(0.5, 0.03,
         "Total bytes/thread constant = 8 B (= 4 × f32 accumulated, converted to bf16)",
         ha="center", va="bottom", transform=ax2.transAxes,
         fontsize=7.5, color="#555555", style="italic")

plt.suptitle(
    "MFMA epilogue optimization: instruction count & coalescing comparison",
    fontsize=12, fontweight="bold", y=1.01
)

plt.tight_layout()
out = "/home/adespirl/wave/July1/report/fig_epilogue_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
