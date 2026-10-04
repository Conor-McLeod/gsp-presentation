"""Plot Nvidia low-precision vs FP64 throughput from nvidia_precision.csv."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

GREEN, GREY, BLUE = "#76b900", "#8a8f98", "#3b6fd4"
INK, MUTED = "#1c1e21", "#6b7280"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": "#d1d5db",
    "xtick.color": INK,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv("nvidia_precision.csv")
df["ratio"] = df["low_precision_dense_tflops"] / df["fp64_native_tflops"]
labels = [f"{g}\n{y} · {f}" for g, y, f in
          zip(df["gpu"], df["year"], df["low_precision_format"])]
x = np.arange(len(df))
w = 0.38


def style(ax, ticklabels):
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks(x, ticklabels, fontsize=11)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.grid(axis="y", color="#e5e7eb", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6),
                               gridspec_kw={"width_ratios": [1.3, 1]})

# Left: absolute throughput
b1 = ax1.bar(x - w / 2, df["low_precision_dense_tflops"], w, color=GREEN,
             label="Lowest-precision tensor core", zorder=3)
b2 = ax1.bar(x + w / 2, df["fp64_native_tflops"], w, color=GREY,
             label="Native FP64", zorder=3)
style(ax1, labels)
ax1.set_ylim(0.6, 1e5)
ax1.set_ylabel("TFLOPS (log scale)")
ax1.set_title("Throughput", loc="left", fontsize=15, weight="bold", pad=12)
ax1.legend(loc="upper left", frameon=False, fontsize=11)
ax1.bar_label(b1, fmt="{:,.0f}", fontsize=10, padding=3, color=INK)
ax1.bar_label(b2, fmt="{:g}", fontsize=10, padding=3, color=MUTED)

# Right: ratio
b3 = ax2.bar(x, df["ratio"], 0.55, color=BLUE, zorder=3)
style(ax2, list(df["gpu"]))
ax2.set_ylim(8, 5e4)
ax2.set_ylabel("Low precision ÷ native FP64")
ax2.set_title("How far low precision outpaces FP64", loc="left",
              fontsize=15, weight="bold", pad=12)
ax2.bar_label(b3, labels=[f"{r:,.0f}×" for r in df["ratio"]], fontsize=11,
              padding=3, weight="bold", color=INK)

fig.text(0.01, 0.01, "Dense figures. B200 approximate; Rubin preliminary "
         "(NVFP4 training figure). Sources: Nvidia datasheets/blogs, "
         "Hot Chips 2025.", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 1], w_pad=3)
fig.savefig("nvidia_precision.png", dpi=200, facecolor="white")
print("Saved nvidia_precision.png")
