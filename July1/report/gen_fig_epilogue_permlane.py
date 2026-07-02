"""
Figure 2: Permlane Transposed Epilogue
Three-panel diagram showing standard layout, transposed accumulation, and permlane shuffle.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor("white")

ROWS = 4
COLS = 4   # simplified 4×4 for clarity

THREAD_COLORS = [
    "#4C72B0",  # T0 – col 0
    "#DD8452",  # T1 – col 1
    "#55A868",  # T2 – col 2
    "#C44E52",  # T3 – col 3
]

CELL_W = 1.0
CELL_H = 0.85

def draw_4x4(ax, label_fn, color_fn, title, subtitle="",
             highlight_axis=None, extra_note=""):
    """
    label_fn(r, c)  -> cell text
    color_fn(r, c)  -> matplotlib color
    highlight_axis: 'row' highlights diagonal; 'col' highlights transpose
    """
    ax.set_xlim(-0.8, COLS * CELL_W + 0.2)
    ax.set_ylim(-2.8, ROWS * CELL_H + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8, wrap=True)
    if subtitle:
        ax.text(COLS * CELL_W / 2, ROWS * CELL_H + 0.72, subtitle,
                ha="center", va="bottom", fontsize=8.5,
                color="#444444", style="italic")

    for r in range(ROWS):
        for c in range(COLS):
            rect = plt.Rectangle(
                (c * CELL_W, (ROWS - 1 - r) * CELL_H),
                CELL_W, CELL_H,
                linewidth=1.2, edgecolor="black",
                facecolor=color_fn(r, c), alpha=0.85, zorder=2
            )
            ax.add_patch(rect)
            ax.text(
                c * CELL_W + CELL_W / 2,
                (ROWS - 1 - r) * CELL_H + CELL_H / 2,
                label_fn(r, c),
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3
            )

    # Axis labels
    for c in range(COLS):
        ax.text(c * CELL_W + CELL_W / 2, ROWS * CELL_H + 0.12,
                f"col {c}", ha="center", va="bottom", fontsize=7.5,
                color=THREAD_COLORS[c], fontweight="bold")
    for r in range(ROWS):
        ax.text(-0.15, (ROWS - 1 - r) * CELL_H + CELL_H / 2,
                f"r{r}", ha="right", va="center", fontsize=7.5)

    if extra_note:
        ax.text(COLS * CELL_W / 2, -2.55, extra_note,
                ha="center", va="center", fontsize=8,
                color="#333333", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                          edgecolor="#AAAAAA", alpha=0.9))


# ── Panel A: standard MFMA layout ─────────────────────────────────────────
ax = axes[0]

def lbl_std(r, c):
    return f"C[{r},{c}]"

def col_std(r, c):
    # Thread c owns column c for all rows
    return THREAD_COLORS[c]

draw_4x4(ax, lbl_std, col_std,
         "Panel A — Standard MFMA layout\n(C in registers)",
         subtitle="Thread T owns column T for all rows",
         extra_note="Store C[r][c]: each thread issues\n4 scattered stride-16 writes")

# Show problematic stride annotation
ax.annotate(
    "", xy=(COLS * CELL_W + 0.1, (ROWS - 1 - 3) * CELL_H + CELL_H / 2),
    xytext=(COLS * CELL_W + 0.1, (ROWS - 1 - 0) * CELL_H + CELL_H / 2),
    arrowprops=dict(arrowstyle="<->", color="#CC0000", lw=1.8)
)
ax.text(COLS * CELL_W + 0.18, ROWS * CELL_H / 2,
        "stride = 16\n(non-\ncontiguous)",
        ha="left", va="center", fontsize=7, color="#CC0000")

# Scatter store arrows
for c in range(COLS):
    for r in range(ROWS):
        x = c * CELL_W + CELL_W / 2
        y_start = (ROWS - 1 - r) * CELL_H
        y_end   = -0.6 - r * 0.4
        ax.annotate(
            "", xy=(x + (c - 1.5) * 0.05, y_end),
            xytext=(x, y_start),
            arrowprops=dict(arrowstyle="-|>", color=THREAD_COLORS[c],
                            lw=0.8, linestyle="dashed"),
            zorder=1
        )

ax.add_patch(plt.Rectangle((0, -2.35), COLS * CELL_W, 0.2,
                            facecolor="#DDDDDD", edgecolor="black", lw=1))
ax.text(COLS * CELL_W / 2, -2.25, "Global Memory",
        ha="center", va="center", fontsize=7.5)


# ── Panel B: transposed accumulation ──────────────────────────────────────
ax = axes[1]

def lbl_trans(r, c):
    # The register holds C^T, so position (r,c) in register = C[c][r]
    return f"C[{c},{r}]\n=C^T[{r},{c}]"

def col_trans(r, c):
    # After swapping A,B: thread T holds row T (not column T) of C^T
    # So in C^T grid, thread r owns row r
    return THREAD_COLORS[r]

draw_4x4(ax, lbl_trans, col_trans,
         "Panel B — Transposed accumulation\n(C^T in registers)",
         subtitle="Swap A,B inputs → MFMA accumulates C^T",
         extra_note="Thread T now owns a contiguous\nrow of C^T[T][:]")

# Arrow indicating input swap
ax.annotate(
    "pass inputs\ntransposed", xy=(0.5 * CELL_W, ROWS * CELL_H + 0.85),
    xytext=(-0.5, ROWS * CELL_H + 0.85),
    ha="center", va="center", fontsize=8, color="#1A6696",
    fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#1A6696", lw=2.0)
)

# Row legend: now threads own rows
for r in range(ROWS):
    ax.text(-0.15, (ROWS - 1 - r) * CELL_H + CELL_H / 2,
            f"T{r}", ha="right", va="center", fontsize=7.5,
            color=THREAD_COLORS[r], fontweight="bold")

# Contiguous highlight: bracket on row 0
y_r0 = (ROWS - 1) * CELL_H
ax.annotate(
    "", xy=(COLS * CELL_W + 0.05, y_r0 + CELL_H / 2),
    xytext=(0, y_r0 + CELL_H / 2),
    arrowprops=dict(arrowstyle="-", color="#00800A", lw=2.5,
                    connectionstyle="arc3,rad=0")
)
ax.text(COLS * CELL_W + 0.1, y_r0 + CELL_H / 2,
        "T0 row:\ncontiguous!",
        ha="left", va="center", fontsize=7, color="#00800A", fontweight="bold")


# ── Panel C: after permlane → store C ─────────────────────────────────────
ax = axes[2]

PERM_COLORS = [
    "#1F4E79",  # deep blue
    "#1E6E41",  # deep green
    "#6A1A4D",  # deep purple
    "#7B2C00",  # deep brown
]

def lbl_perm(r, c):
    return f"C[{r},{c}]"

def col_perm(r, c):
    return PERM_COLORS[r]

draw_4x4(ax, lbl_perm, col_perm,
         "Panel C — After permlane shuffle\n(store C to global memory)",
         subtitle="permlane reorders C^T layout → correct C",
         extra_note="2 × buffer_store_dword per thread\n(2×bf16, fully coalesced)")

# Show permlane arrow between panels B and C (outside axis, use fig transform)
ax.text(COLS * CELL_W / 2, ROWS * CELL_H + 0.85,
        "permlane\nshuffle ↓",
        ha="center", va="center", fontsize=9, color="#7B3F00",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD",
                  edgecolor="#E8A800", alpha=0.9))

# Vector store arrows (solid, wide)
for r in range(ROWS):
    y_start = (ROWS - 1 - r) * CELL_H
    for k in range(2):
        x = k * CELL_W * 2 + CELL_W
        ax.annotate(
            "", xy=(x, -0.5),
            xytext=(x, y_start),
            arrowprops=dict(arrowstyle="-|>", color=PERM_COLORS[r],
                            lw=1.8, linestyle="solid"),
            zorder=4
        )

ax.add_patch(plt.Rectangle((0, -0.75), COLS * CELL_W, 0.25,
                            facecolor="#DDDDDD", edgecolor="black", lw=1))
ax.text(COLS * CELL_W / 2, -0.62, "Global Memory",
        ha="center", va="center", fontsize=7.5)

ax.text(COLS * CELL_W / 2, -1.1,
        "Row-contiguous stores",
        ha="center", va="center", fontsize=8, color="#1A6696",
        fontweight="bold")


plt.suptitle(
    "MFMA epilogue: permlane transposed accumulation\n"
    "(simplified 4-thread / 4×4 tile view)",
    fontsize=12, fontweight="bold", y=1.04
)

plt.tight_layout(rect=[0, 0, 1, 1])
out = "/home/adespirl/wave/July1/report/fig_epilogue_permlane.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
