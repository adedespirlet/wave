"""
Figure 1: MFMA Register Layout + XOR Shuffle epilogue
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# Color palette: one per thread (8 threads, T0–T7)
THREAD_COLORS = [
    "#4C72B0",  # T0 blue
    "#DD8452",  # T1 orange
    "#55A868",  # T2 green
    "#C44E52",  # T3 red
    "#8172B2",  # T4 purple
    "#937860",  # T5 brown
    "#DA8BC3",  # T6 pink
    "#8C8C8C",  # T7 grey
]
PAIR_COLORS = [
    ("#4C72B0", "#DD8452"),   # pair (T0,T1)
    ("#55A868", "#C44E52"),   # pair (T2,T3)
    ("#8172B2", "#937860"),   # pair (T4,T5)
    ("#DA8BC3", "#8C8C8C"),   # pair (T6,T7)
]

ROWS = 4
COLS = 8
CELL_W = 1.0
CELL_H = 0.8

def draw_grid(ax, ownership, cell_labels, title, alpha=0.85):
    """
    ownership[r][c] = thread index (0–7)
    cell_labels[r][c] = text to show inside cell
    """
    ax.set_xlim(0, COLS * CELL_W)
    ax.set_ylim(-2.5, ROWS * CELL_H + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for r in range(ROWS):
        for c in range(COLS):
            t = ownership[r][c]
            color = THREAD_COLORS[t]
            rect = plt.Rectangle(
                (c * CELL_W, (ROWS - 1 - r) * CELL_H),
                CELL_W, CELL_H,
                linewidth=1.2, edgecolor="black",
                facecolor=color, alpha=alpha,
                zorder=2
            )
            ax.add_patch(rect)
            ax.text(
                c * CELL_W + CELL_W / 2,
                (ROWS - 1 - r) * CELL_H + CELL_H / 2,
                cell_labels[r][c],
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white",
                zorder=3
            )

    # Column header: thread names
    for c in range(COLS):
        ax.text(
            c * CELL_W + CELL_W / 2,
            ROWS * CELL_H + 0.25,
            f"T{c}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=THREAD_COLORS[c]
        )

    # Row labels
    for r in range(ROWS):
        ax.text(
            -0.25,
            (ROWS - 1 - r) * CELL_H + CELL_H / 2,
            f"row {r}",
            ha="right", va="center",
            fontsize=8
        )

# ── Panel A: naive layout ──────────────────────────────────────────────────
ax = axes[0]

# Each column owned entirely by that thread
ownership_naive = [[c for c in range(COLS)] for r in range(ROWS)]
labels_naive = [[f"T{c}[r{r}]" for c in range(COLS)] for r in range(ROWS)]

draw_grid(ax, ownership_naive, labels_naive,
          "Panel A — Naive epilogue: MFMA register layout")

# Store arrows: 4 dashed arrows per thread going down
arrow_y_start = -0.3
arrow_y_end   = -1.7
for c in range(COLS):
    for k in range(4):
        x = c * CELL_W + CELL_W / 2 + (k - 1.5) * 0.15
        ax.annotate(
            "", xy=(x, arrow_y_end), xytext=(x, arrow_y_start),
            arrowprops=dict(arrowstyle="-|>", color=THREAD_COLORS[c],
                            lw=0.9, linestyle="dashed"),
            zorder=1
        )

# Global memory bar
ax.add_patch(plt.Rectangle((0, arrow_y_end - 0.25), COLS * CELL_W, 0.25,
                            facecolor="#DDDDDD", edgecolor="black", lw=1, zorder=0))
ax.text(COLS * CELL_W / 2, arrow_y_end - 0.12, "Global Memory",
        ha="center", va="center", fontsize=8.5, color="black")
ax.text(COLS * CELL_W / 2, arrow_y_end - 0.55,
        "4 × buffer_store_ushort (2 B) per thread\n→ 4 scattered 2-byte writes per thread",
        ha="center", va="center", fontsize=8, color="#333333", style="italic")

# ── Panel B: after XOR shuffle ────────────────────────────────────────────
ax = axes[1]

# After XOR exchange: even thread (T0) owns rows 0-1 of both cols;
#                     odd thread (T1) owns rows 2-3 of both cols.
# Represented by pair color (blend via hatching or pair index)
PAIR_EVEN_ROWS = [0, 1]   # even thread now owns these rows in its pair
PAIR_ODD_ROWS  = [2, 3]   # odd thread

# Ownership after XOR: col pair (0,1) → same pair color, but split by row
# We'll use new "virtual thread" indices that encode the pair
ownership_xor = [[0] * COLS for _ in range(ROWS)]
labels_xor    = [[""  ] * COLS for _ in range(ROWS)]

PAIR_COLORS_FLAT = [
    "#2166AC",  # pair 0 even
    "#4DAC26",  # pair 1 even
    "#762A83",  # pair 2 even
    "#B2182B",  # pair 3 even
]
PAIR_COLORS_ODD = [
    "#74ADD1",  # pair 0 odd
    "#A6D96A",  # pair 1 odd
    "#C2A5CF",  # pair 2 odd
    "#F4A582",  # pair 3 odd
]

# Rebuild with fine-grained colors
def draw_grid_xor(ax, title):
    ax.set_xlim(0, COLS * CELL_W)
    ax.set_ylim(-2.5, ROWS * CELL_H + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    pair_labels = {0: ("T0+T1", "T0+T1"), 1: ("T2+T3", "T2+T3"),
                   2: ("T4+T5", "T4+T5"), 3: ("T6+T7", "T6+T7")}

    for c in range(COLS):
        pair = c // 2
        is_even_col = (c % 2 == 0)
        for r in range(ROWS):
            if r in PAIR_EVEN_ROWS:
                owner_str = f"T{pair*2}"
                color = PAIR_COLORS_FLAT[pair]
            else:
                owner_str = f"T{pair*2+1}"
                color = PAIR_COLORS_ODD[pair]
            # Label: owner[row][col]
            lbl = f"{owner_str}\n[r{r},c{c}]"
            rect = plt.Rectangle(
                (c * CELL_W, (ROWS - 1 - r) * CELL_H),
                CELL_W, CELL_H,
                linewidth=1.2, edgecolor="black",
                facecolor=color, alpha=0.88,
                zorder=2
            )
            ax.add_patch(rect)
            ax.text(
                c * CELL_W + CELL_W / 2,
                (ROWS - 1 - r) * CELL_H + CELL_H / 2,
                lbl, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                zorder=3
            )

    # Pair headers
    for pair in range(COLS // 2):
        cx = (pair * 2 + 0.5) * CELL_W + 0.5 * CELL_W
        ax.text(cx, ROWS * CELL_H + 0.25,
                f"pair (T{pair*2},T{pair*2+1})",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color=PAIR_COLORS_FLAT[pair])

    # Row labels
    for r in range(ROWS):
        ax.text(-0.25, (ROWS - 1 - r) * CELL_H + CELL_H / 2,
                f"row {r}", ha="right", va="center", fontsize=8)

    # XOR exchange arrows between adjacent columns (at top)
    for pair in range(COLS // 2):
        c0 = pair * 2
        c1 = c0 + 1
        x0 = c0 * CELL_W + CELL_W / 2
        x1 = c1 * CELL_W + CELL_W / 2
        y  = ROWS * CELL_H + 0.08
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="<->", color="#333333",
                                    lw=2.0, connectionstyle="arc3,rad=0.3"),
                    zorder=5)
        ax.text((x0 + x1) / 2, y + 0.18, "XOR",
                ha="center", va="bottom", fontsize=7,
                color="#333333", fontweight="bold")

    # Store arrows: 2 wide solid arrows per pair (one per thread in pair)
    arrow_y_start = -0.3
    arrow_y_end   = -1.7
    for pair in range(COLS // 2):
        for sub, (color, rows_owned) in enumerate(
            [(PAIR_COLORS_FLAT[pair], PAIR_EVEN_ROWS),
             (PAIR_COLORS_ODD[pair],  PAIR_ODD_ROWS)]):
            x_base = (pair * 2 + sub) * CELL_W + CELL_W / 2
            # Two arrows (2 dword stores)
            for k in range(2):
                x = x_base + (k - 0.5) * 0.25
                ax.annotate("", xy=(x, arrow_y_end), xytext=(x, arrow_y_start),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=2.0, linestyle="solid"),
                            zorder=4)

    # Global memory bar
    ax.add_patch(plt.Rectangle((0, arrow_y_end - 0.25), COLS * CELL_W, 0.25,
                                facecolor="#DDDDDD", edgecolor="black", lw=1, zorder=0))
    ax.text(COLS * CELL_W / 2, arrow_y_end - 0.12, "Global Memory",
            ha="center", va="center", fontsize=8.5, color="black")
    ax.text(COLS * CELL_W / 2, arrow_y_end - 0.55,
            "2 × buffer_store_dword (4 B = 2×bf16) per thread\n"
            "→ 2× fewer instructions, fully coalesced",
            ha="center", va="center", fontsize=8, color="#333333", style="italic")

draw_grid_xor(axes[1], "Panel B — After XOR shuffle: vectorized stores")

plt.suptitle(
    "MFMA epilogue: XOR shuffle vectorization\n"
    "(simplified 8-thread / 4×8 sub-tile view)",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
out = "/home/adespirl/wave/July1/report/fig_epilogue_xor.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
