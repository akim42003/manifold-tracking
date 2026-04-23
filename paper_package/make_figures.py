"""Generate figures for the revised paper.

All figures use matplotlib, saved as PDF (for LaTeX) and PNG (for quick viewing).
"""
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS = Path("data")

def load_gemma():
    with open(UPLOADS / "per_cluster_svd_spherical.json") as f:
        return json.load(f)["clusters"]

def has_coh(c, k=2):
    return any(p.get("count", 1) >= k for p in c.get("top_pairs", []))

def load_olmo_final():
    with open(UPLOADS / "3896b_per_cluster_svd_spherical_nltk.log") as f:
        text = f.read()
    pattern = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\[(\d+)-(\d+)\]\s+(\d+)\s+(.*?)$", re.MULTILINE)
    clusters = []
    for m in pattern.findall(text):
        c_id, n, d50, ci_lo, ci_hi, d90, tail = m
        pair_pattern = re.compile(r"(\S+?)\u2192(\S+?)\((\d+)\)")
        pairs = [{"in": a, "out": b, "count": int(c)} for a, b, c in pair_pattern.findall(tail)]
        clusters.append({"cluster": int(c_id), "n": int(n), "d50": int(d50), "top_pairs": pairs})
    return clusters

gemma = load_gemma()
olmo = load_olmo_final()

def split(cs):
    return [c for c in cs if has_coh(c)], [c for c in cs if not has_coh(c)]

gemma_coh, gemma_noise = split(gemma)
olmo_coh, olmo_noise = split(olmo)

print(f"Gemma: {len(gemma)} total, {len(gemma_coh)} coh, {len(gemma_noise)} noise")
print(f"OLMo-2: {len(olmo)} total, {len(olmo_coh)} coh, {len(olmo_noise)} noise")

bins = [(30,50), (50,75), (75,100), (100,150), (150,250), (250,1000)]

# ------------------------------------------------------------------
# Figure 1: d50 vs n scatter, Gemma
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.scatter([c["n"] for c in gemma_noise], [c["d50"] for c in gemma_noise],
           s=18, alpha=0.35, color="#888", label=f"noise (n={len(gemma_noise)})", edgecolor="none")
ax.scatter([c["n"] for c in gemma_coh], [c["d50"] for c in gemma_coh],
           s=24, alpha=0.85, color="#1D9E75", label=f"content-coherent (n={len(gemma_coh)})", edgecolor="none")

coh_meds, noi_meds = [], []
for lo, hi in bins:
    xc = np.sqrt(lo*hi)
    cc = [c["d50"] for c in gemma_coh if lo <= c["n"] < hi]
    nn = [c["d50"] for c in gemma_noise if lo <= c["n"] < hi]
    if cc: coh_meds.append((xc, st.median(cc)))
    if nn: noi_meds.append((xc, st.median(nn)))

if noi_meds:
    xs, ys = zip(*noi_meds)
    ax.plot(xs, ys, "o-", color="#444", lw=1.5, ms=7, zorder=5, label="noise, median by size bin")
if coh_meds:
    xs, ys = zip(*coh_meds)
    ax.plot(xs, ys, "s-", color="#0E7F5B", lw=1.5, ms=7, zorder=5, label="coherent, median by size bin")

ax.axhspan(5, 22, alpha=0.15, color="#FFA500", zorder=0, label="Hernandez 5\u201322D")
ax.set_xlabel("cluster size $n$")
ax.set_ylabel("$d_{50}$ (50%-variance dimensionality)")
ax.set_xscale("log")
ax.set_title(f"Gemma-3 4B: $r(n, d_{{50}})$ = 0.71 (coherent), 0.73 (noise)")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax.grid(True, alpha=0.3)
ax.set_xlim(25, 700)
ax.set_ylim(0, 38)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_d50_vs_size_gemma.pdf")
plt.savefig(OUT_DIR / "fig1_d50_vs_size_gemma.png", dpi=150)
plt.close()
print("Saved fig1_d50_vs_size_gemma")

# ------------------------------------------------------------------
# Figure 2: median d50 by size bin, Gemma vs OLMo-2
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
bin_labels = ["30\u201349", "50\u201374", "75\u201399", "100\u2013149", "150\u2013249", "250+"]

def size_bin_medians(cs):
    return [st.median([c["d50"] for c in cs if lo <= c["n"] < hi]) if any(lo <= c["n"] < hi for c in cs) else 0 for lo, hi in bins]

g_coh = size_bin_medians(gemma_coh)
g_noi = size_bin_medians(gemma_noise)
o_coh = size_bin_medians(olmo_coh)
o_noi = size_bin_medians(olmo_noise)

x = np.arange(len(bin_labels))
width = 0.35

for ax, coh, noi, title in [(axes[0], g_coh, g_noi, "Gemma-3 4B ($K$=256)"),
                              (axes[1], o_coh, o_noi, "OLMo-2 7B ($K$=512)")]:
    ax.bar(x - width/2, noi, width, label="noise", color="#888", edgecolor="none")
    ax.bar(x + width/2, coh, width, label="coherent", color="#1D9E75", edgecolor="none")
    ax.axhspan(5, 22, alpha=0.15, color="#FFA500", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30)
    ax.set_title(title)
    ax.set_xlabel("cluster size bin")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")
axes[0].set_ylabel("median $d_{50}$ in bin")
fig.suptitle("Median $d_{50}$ by cluster size: coherent and noise scale the same way", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_size_bin_comparison.pdf")
plt.savefig(OUT_DIR / "fig2_size_bin_comparison.png", dpi=150)
plt.close()
print("Saved fig2_size_bin_comparison")

# ------------------------------------------------------------------
# Figure 3: OLMo-2 trajectory
# ------------------------------------------------------------------

tokens = [1, 3, 13, 51, 214, 911, 3896]
pass_e = [88, 89, 96, 91, 42, 44, 44]
pass_sr = [92, 89, 96, 97, 90, 91, 84]
coh_pct = [2.3, 0.0, 0.0, 0.2, 2.2, 12.3, 11.6]

fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax1.semilogx(tokens, pass_e, "o-", color="#AA3333", ms=7, lw=1.8, label="pass% (Euclidean $k$-means)")
ax1.semilogx(tokens, pass_sr, "s-", color="#5555AA", ms=7, lw=1.8, label="pass% (spherical, random init)")
ax1.set_xlabel("training tokens (billions)")
ax1.set_ylabel("pass rate (%)")
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.semilogx(tokens, coh_pct, "^-", color="#0E7F5B", ms=9, lw=2.2, label="content-coherence rate")
ax2.set_ylabel("content-coherence rate (%)", color="#0E7F5B")
ax2.set_ylim(0, 15)
ax2.tick_params(axis="y", labelcolor="#0E7F5B")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8.5)

ax1.axvspan(214, 911, alpha=0.08, color="#0E7F5B")
ax1.text(440, 97, "coherence\nemergence", fontsize=9, ha="center", color="#0E7F5B", alpha=0.9)
ax1.set_title("OLMo-2 7B trajectory: pass rate vs content coherence")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_olmo_trajectory.pdf")
plt.savefig(OUT_DIR / "fig3_olmo_trajectory.png", dpi=150)
plt.close()
print("Saved fig3_olmo_trajectory")

# ------------------------------------------------------------------
# Figure 4: cross-model pass% vs coh%
# ------------------------------------------------------------------

models = ["Gemma-3 4B", "Mistral 7B", "Qwen-2 7B", "Amber 7B", "OLMo-2 7B"]
pass_rates = [70, 43, 14, 65, 88]
coh_rates = [35, 10, 4, 0.8, 12]

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, pass_rates, width, label="pass rate (%)", color="#AA3333", edgecolor="none")
ax.bar(x + width/2, coh_rates, width, label="content-coherence rate (%)", color="#1D9E75", edgecolor="none")

amber_idx = models.index("Amber 7B")
ax.annotate("fake-pass:\n65% numerical,\n<1% coherent",
            xy=(amber_idx + width/2, coh_rates[amber_idx]),
            xytext=(amber_idx + 0.7, 45),
            arrowprops=dict(arrowstyle="->", color="#444"),
            fontsize=8.5, ha="left")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20)
ax.set_ylabel("rate (%)")
ax.set_title("Cross-model panel: pass rate diverges sharply from content coherence")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_crossmodel.pdf")
plt.savefig(OUT_DIR / "fig4_crossmodel.png", dpi=150)
plt.close()
print("Saved fig4_crossmodel")

# ------------------------------------------------------------------
# Figure 5: d50 distribution, coh highlighted
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, sharex=True)
bin_edges = np.arange(0, 36, 3)

for ax, coh_cs, noi_cs, title in [
    (axes[0], gemma_coh, gemma_noise, f"Gemma-3 4B ({len(gemma_coh)} coherent of {len(gemma)})"),
    (axes[1], olmo_coh, olmo_noise, f"OLMo-2 7B ({len(olmo_coh)} coherent of {len(olmo)})"),
]:
    ax.hist([c["d50"] for c in noi_cs], bins=bin_edges, alpha=0.5, color="#888", label="noise", edgecolor="white")
    ax.hist([c["d50"] for c in coh_cs], bins=bin_edges, alpha=0.85, color="#1D9E75", label="coherent", edgecolor="white")
    ax.axvspan(5, 22, alpha=0.12, color="#FFA500", zorder=0)
    ax.set_title(title)
    ax.set_xlabel("$d_{50}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    coh_d50s = [c["d50"] for c in coh_cs]
    if coh_d50s:
        m = st.median(coh_d50s)
        ax.axvline(m, color="#0E7F5B", ls="--", alpha=0.85, lw=1.4)
        ax.text(m + 0.5, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] else 80,
                f"coh median: {m:.0f}", fontsize=8.5, color="#0E7F5B")

axes[0].set_ylabel("cluster count")
fig.suptitle("$d_{50}$ distributions: both models span the full range", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_d50_distribution.pdf")
plt.savefig(OUT_DIR / "fig5_d50_distribution.png", dpi=150)
plt.close()
print("Saved fig5_d50_distribution")

print()
print(f"All figures in {OUT_DIR}:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
