"""Plot the copy-engine collective steps from peer_copy.csv."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEMM, OTHER = "#2a78d6", "#9ec5f4"
INK, MUTED = "#1c1e21", "#6b7280"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": "#d1d5db",
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv("peer_copy.csv")
df["other_ms"] = df["total_ms"] - df["assemble_gemm_ms"]
y = np.arange(len(df))[::-1]
h = 0.62

fig, ax = plt.subplots(figsize=(12, 3.9))
ax.barh(y, df["assemble_gemm_ms"], h, color=GEMM, label="Assemble + GEMM",
        zorder=3, edgecolor="white", linewidth=2)
ax.barh(y, df["other_ms"], h, left=df["assemble_gemm_ms"], color=OTHER,
        label="Other stages", zorder=3, edgecolor="white", linewidth=2)

prev = None
for yi, total in zip(y, df["total_ms"]):
    delta = "" if prev is None else f"−{prev - total:.2f} ms"
    ax.text(total + 0.15, yi, f"{total:.2f} ms", va="center", fontsize=12,
            weight="bold", color=INK)
    if delta:
        ax.text(17.2, yi, delta, va="center", ha="left", fontsize=12, color=MUTED, clip_on=False)
    prev = total

ax.set_yticks(y, df["step"], fontsize=12)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xlim(0, 16.5)
ax.set_xlabel("Slowest-rank time per rep (ms)")
ax.grid(axis="x", color="#e5e7eb", linewidth=1)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.text(17.2, y[0] + 0.62, "Δ vs. previous", ha="left", va="center",
        fontsize=11, color=MUTED, clip_on=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          frameon=False, fontsize=11)

fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.savefig("peer_copy.png", dpi=200, facecolor="white", bbox_inches="tight",
            pad_inches=0.1)
print("Saved peer_copy.png")
