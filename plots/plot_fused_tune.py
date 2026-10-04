"""Plot the fused GEMM tuning candidates from fused_tune.csv.

Copied from par_gemmul8/analysis/fused_tune.csv. Each candidate is compared
with the baseline passes either side of it in the same job, on the slowest
rank, and split into clock and cycles: time = cycles / clock.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED = "#1c1e21", "#6b7280"
FAMILIES = {  # label prefix -> (legend, colour)
    "swizzle": ("Tile swizzle 2/4/8", "#2a78d6"),
    "tile": ("Smaller tiles / other shapes", "#e8743b"),
    "cluster": ("Bigger clusters (2×2, 4×1)", "#9b59b6"),
    "sms": ("Capped SM count", "#c0392b"),
    "other": ("Raster, epilogue, Stream-K, panels", "#9ca3af"),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": "#d1d5db",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def family(label: str) -> str:
    if label.startswith("swizzle") and "panel" not in label:
        return "swizzle"
    if label.startswith("sms"):
        return "sms"
    if "_2x2" in label or "_4x1" in label:
        return "cluster"
    if label[0] in "cp" and "_sk" not in label:
        return "tile"
    return "other"


rows = []
for _, g in pd.read_csv("fused_tune.csv").groupby("job_id"):
    g = g.reset_index(drop=True)
    for i, row in g.iterrows():
        # A/A passes and the reverse check (old default against swizzle 8)
        # are not candidates.
        if row.label in ("base", "base_again", "swizzle1", "occupancy"):
            continue
        base = g.loc[[i - 1, i + 1]]
        t = [row[f"gemm_us_per_rep_r{r}"] / base[f"gemm_us_per_rep_r{r}"].mean()
             for r in range(4)]
        clk = [row[f"mhz_r{r}"] / base[f"mhz_r{r}"].mean() for r in range(4)]
        w = int(np.argmax(t))  # the slowest rank paces the run
        rows.append((row.label, family(row.label), clk[w], t[w] * clk[w], t[w]))
df = pd.DataFrame(rows, columns=["label", "family", "clock", "cycles", "time"])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.linspace(0.6, 1.4, 50)
for time in (0.8, 0.9, 1.0, 1.2, 1.4):
    ax.plot(x, time * x, color="#111827" if time == 1.0 else "#e5e7eb",
            linewidth=1.4 if time == 1.0 else 1, zorder=1)
    xe = min(1.395, 1.69 / time)
    ax.text(xe, time * xe + 0.012, "same time" if time == 1.0 else f"{time:g}× time",
            fontsize=10, color=INK if time == 1.0 else MUTED, ha="right",
            va="bottom")
ax.fill_between(x, 0, x, color="#2a78d6", alpha=0.06, zorder=0)
ax.text(1.35, 0.92, "faster", color="#2a78d6", fontsize=12, weight="bold",
        ha="right")

for fam, (name, colour) in FAMILIES.items():
    d = df[df.family == fam]
    ax.scatter(d.clock, d.cycles, s=70, color=colour, label=name, zorder=3,
               edgecolor="white", linewidth=1)
ax.scatter([1], [1], s=120, marker="*", color=INK, zorder=4)
ax.annotate("default", (1, 1), xytext=(-12, 8), textcoords="offset points",
            ha="right", fontsize=11, color=INK)

sw8 = df[df.label == "swizzle8"]
ax.annotate("swizzle 8", (sw8.clock.mean(), sw8.cycles.mean()), xytext=(6, -22),
            textcoords="offset points", fontsize=12, weight="bold",
            color="#2a78d6")

ax.set_xlim(0.6, 1.4)
ax.set_ylim(0.9, 1.7)
ax.set_xlabel("GPC clock vs. baseline  (slowest rank)")
ax.set_ylabel("Cycles vs. baseline")
ax.legend(loc="upper left", frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig("fused_tune.png", dpi=200, facecolor="white", bbox_inches="tight",
            pad_inches=0.1)
print("Saved fused_tune.png")
