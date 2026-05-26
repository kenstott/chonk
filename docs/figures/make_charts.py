"""Generate all paper figures as high-res PNGs."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FONT = "Times New Roman"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FONT, "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GREY = "#888888"
BLUE = "#2166ac"
ORANGE = "#d6604d"
GREEN = "#4dac26"
PURPLE = "#7b2d8b"

# ── Fig 1: GRB Leaderboard grouped bar ────────────────────────────────────────
def fig_grb_leaderboard():
    systems = [
        "G-reasoner\n(LLM index)",
        "AutoPruned\n(LLM index)",
        "HippoRAG2\n(LLM index)",
        "Fast-\nGraphRAG",
        "LightRAG",
        "RAG+rerank\n(baseline)",
        "Ours\n(full stack)",
    ]
    med = [0.733, 0.670, 0.648, 0.641, 0.626, 0.624, 0.756]
    nov = [0.589, 0.637, 0.565, 0.520, 0.451, 0.483, 0.656]
    all_ = [0.661, 0.654, 0.607, 0.581, 0.538, 0.554, 0.712]

    x = np.arange(len(systems))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    bars_med = ax.bar(x - w, med, w, label="Medical", color=BLUE, alpha=0.85)
    bars_nov = ax.bar(x,     nov, w, label="Novel",   color=ORANGE, alpha=0.85)
    bars_all = ax.bar(x + w, all_, w, label="All",    color=GREEN, alpha=0.85)

    # highlight ours
    for bars in (bars_med, bars_nov, bars_all):
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.4)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, ha="center")
    ax.set_ylabel("answer_correctness")
    ax.set_ylim(0.40, 0.82)
    ax.set_title("Figure 1 — GraphRAG-Bench Leaderboard\n(All systems use gpt-4o-mini judge; LLM index cost noted)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0.712, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.text(6.55, 0.715, "0.712", fontsize=8, va="bottom")
    ax.text(0, 0.42, "★ Index cost = 0 LLM calls", fontsize=8, color=BLUE, style="italic")

    fig.tight_layout()
    fig.savefig("docs/figures/fig1_grb_leaderboard.png")
    plt.close(fig)
    print("fig1 done")


# ── Fig 2: Sequential feature addition waterfall ──────────────────────────────
def fig_waterfall():
    labels = [
        "Vanilla+\nrerank",
        "+Semantic\nchunking",
        "+Entity-ref\n+lane (0.45)",
        "+Community\ncontext",
        "+Tighter\nlane (0.60)",
        "→SR, k=30",
        "→SRR\n(reprompt)",
    ]
    values = [0.652, 0.661, 0.662, 0.664, 0.670, 0.710, 0.712]
    deltas = [0, 0.009, 0.001, 0.002, 0.006, 0.040, 0.002]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottoms = [v - d for v, d in zip(values, deltas)]
    bottoms[0] = 0

    colors = []
    for i, d in enumerate(deltas):
        if i == 0:
            colors.append(GREY)
        elif d >= 0.010:
            colors.append(BLUE)
        else:
            colors.append("#aec7e8")

    # invisible base bars
    ax.bar(range(len(values)), bottoms, color="white", edgecolor="none")
    bars = ax.bar(range(len(values)), [d if i > 0 else values[0] for i, d in enumerate(deltas)],
                  bottom=[0] + bottoms[1:], color=colors, edgecolor="white", linewidth=0.5)

    # value labels
    for i, (v, b) in enumerate(zip(values, [0] + bottoms[1:])):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold" if i in (0, 5, 6) else "normal")

    # delta labels inside bars for big step
    ax.text(5, 0.690, "+0.040", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("All (answer_correctness)")
    ax.set_ylim(0.630, 0.730)
    ax.set_title("Figure 2 — Sequential Feature Addition (full-corpus, n=4,072)\nDominant gain: replacing reranking with Structured Response at k=30 (+0.040)")
    ax.axhline(0.712, color="black", lw=0.8, ls="--", alpha=0.4)

    blue_patch = mpatches.Patch(color=BLUE, label="Large gain (≥0.010)")
    grey_patch = mpatches.Patch(color="#aec7e8", label="Incremental gain")
    base_patch = mpatches.Patch(color=GREY, label="Baseline")
    ax.legend(handles=[base_patch, grey_patch, blue_patch], loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig("docs/figures/fig2_waterfall.png")
    plt.close(fig)
    print("fig2 done")


# ── Fig 3: Mini vs Haiku SRR diverging bar ────────────────────────────────────
def fig_srr_model():
    conditions = [
        "Vanilla+rerank\n(no SRR)",
        "Vanilla+rerank\n+SRR",
        "Graph k30\n(no SRR)",
        "Graph k30\n+SRR",
    ]
    mini =  [0.652, 0.685, 0.678, 0.712]
    haiku = [0.621, 0.598, 0.623, 0.632]

    x = np.arange(len(conditions))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7, 3.6))

    ax.bar(x - w/2, mini,  w, label="Mini (gpt-4o-mini)", color=BLUE,   alpha=0.85)
    ax.bar(x + w/2, haiku, w, label="Haiku 4.5",          color=ORANGE, alpha=0.85)

    # annotate the haiku vanilla+SRR drop
    ax.annotate("−0.023\n(SRR hurts\nbelow capability\nthreshold)",
                xy=(1 + w/2, 0.598), xytext=(1.9, 0.580),
                fontsize=7.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))

    for i, (m, h) in enumerate(zip(mini, haiku)):
        ax.text(i - w/2, m + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, h + 0.002, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("All (answer_correctness, GRB)")
    ax.set_ylim(0.560, 0.740)
    ax.set_title("Figure 4 — SRR Gains Are Model-Capability-Gated\nHaiku vanilla+SRR degrades (−0.023); Mini gains in both graph and non-graph settings")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0.652, color=GREY, lw=0.7, ls=":", alpha=0.7, label="Mini baseline")

    fig.tight_layout()
    fig.savefig("docs/figures/fig3_srr_model.png")
    plt.close(fig)
    print("fig3 done")


# ── Fig 4: HARE-Bench heatmap ─────────────────────────────────────────────────
def fig_hare_heatmap():
    runs_short = [
        "k50+BC+BM25+ADF",
        "k50+BC+BM25",
        "k30+BC+BM25",
        "k50+SRR",
        "k50+SRR (no GLEIF)",
        "k50+SRR+ADF",
        "k30+BC+BM25+ADF",
        "k30+SRR",
        "k30+rerank+ADF",
        "k30+rerank",
        "k30+rerank (no GLEIF)",
        "vanilla+rerank",
        "Haiku k50",
        "Sonnet k30",
        "Haiku k30",
        "sovereign k30",
    ]
    # TAL, DAL, MDJ, TVR, CE  (post-embedder-rescore, n=500)
    data = [
        [0.694, 0.600, 0.848, 0.802, 0.832],  # k50+BC+BM25+ADF  0.755
        [0.696, 0.590, 0.849, 0.798, 0.829],  # k50+BC+BM25      0.752
        [0.662, 0.625, 0.854, 0.792, 0.801],  # k30+BC+BM25      0.747
        [0.670, 0.580, 0.848, 0.810, 0.808],  # k50+SRR          0.743
        [0.686, 0.565, 0.858, 0.799, 0.796],  # k50+SRR no GLEIF 0.741
        [0.655, 0.583, 0.849, 0.794, 0.799],  # k50+SRR+ADF      0.736
        [0.667, 0.553, 0.838, 0.795, 0.811],  # k30+BC+BM25+ADF  0.733
        [0.595, 0.548, 0.854, 0.786, 0.830],  # k30+SRR          0.723
        [0.639, 0.565, 0.830, 0.792, 0.785],  # k30+rerank+ADF   0.722
        [0.645, 0.520, 0.833, 0.808, 0.793],  # k30+rerank       0.720
        [0.628, 0.550, 0.840, 0.794, 0.776],  # k30+rerank noGL  0.718
        [0.517, 0.592, 0.763, 0.717, 0.000],  # vanilla+rerank   0.647 (CE=NaN→0)
        [0.695, 0.644, 0.679, 0.726, 0.616],  # Haiku k50        0.672
        [0.688, 0.666, 0.647, 0.713, 0.586],  # Sonnet k30       0.660
        [0.631, 0.616, 0.601, 0.712, 0.578],  # Haiku k30        0.628
        [0.622, 0.565, 0.643, 0.652, 0.552],  # sovereign k30    0.607
    ]
    col_labels = ["TAL", "DAL", "MDJ", "TVR", "CE"]

    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0.48, vmax=0.90)

    ax.set_xticks(range(5))
    ax.set_xticklabels(col_labels, fontweight="bold")
    ax.set_yticks(range(len(runs_short)))
    ax.set_yticklabels(runs_short, fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # cell values
    for i in range(len(runs_short)):
        for j in range(5):
            v = arr[i, j]
            color = "white" if v < 0.62 or v > 0.84 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color=color)

    # separator between Mini configs and others
    ax.axhline(11.5, color="black", lw=1.2, ls="--")
    ax.text(4.6, 11.0, "Mini", fontsize=7.5, ha="right", color=BLUE)
    ax.text(4.6, 12.0, "other models", fontsize=7.5, ha="right", color=GREY)

    plt.colorbar(im, ax=ax, shrink=0.6, label="typed_score")
    ax.set_title("Figure 5 — HARE-Bench Per-Type Scores (n=500, post-rescore)\nRuns sorted by Mean; MDJ highest-variance type; DAL hardest overall", pad=14)

    fig.tight_layout()
    fig.savefig("docs/figures/fig4_hare_heatmap.png")
    plt.close(fig)
    print("fig4 done")


# ── Fig 5: k-depth effect on HARE-Bench ──────────────────────────────────────
def fig_k_depth():
    k_vals = [10, 30, 50]
    laned = [0.364, 0.720, 0.743]  # laned60+community+rerank+SRR (post-rescore)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(k_vals, laned, "o-", color=BLUE, lw=2, markersize=7, label="laned60+community+rerank+SRR (Mini)")
    ax.axhline(0.647, color=GREY, lw=0.9, ls="--", label="vanilla+rerank baseline (0.647)")

    for k, v in zip(k_vals, laned):
        ax.text(k, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)

    ax.annotate("+0.023\n(k30→k50)", xy=(50, 0.743), xytext=(44, 0.710),
                fontsize=8, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9))

    ax.set_xlabel("Retrieval depth k")
    ax.set_ylabel("Mean typed_score (HARE-Bench)")
    ax.set_ylim(0.30, 0.80)
    ax.set_xticks(k_vals)
    ax.set_title("Figure 6 — Retrieval Depth Is the Primary Driver on HARE-Bench\nSame strategy (laned60+community+rerank+SRR); only k varies")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig("docs/figures/fig5_k_depth.png")
    plt.close(fig)
    print("fig5 done")


# ── Fig 6: Community×depth and Pruning×depth ─────────────────────────────────
def fig_superadditivity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Community × depth
    k1 = [5, 10, 15]
    no_comm = [0.646, 0.654, None]
    with_comm = [0.661, 0.659, 0.666]
    ax1.plot(k1[:2], no_comm[:2], "s--", color=ORANGE, lw=1.8, markersize=6, label="laned (no community)")
    ax1.plot(k1, [v for v in with_comm], "o-", color=BLUE, lw=1.8, markersize=6, label="laned + community")
    ax1.plot([15], [0.669], "x", color=GREY, markersize=8, label="additive prediction (k=15)", zorder=5)
    for k, v in zip(k1, with_comm):
        ax1.text(k, v + 0.0015, f"{v:.3f}", ha="center", fontsize=8)
    ax1.set_xlabel("k")
    ax1.set_ylabel("All (GRB grid, n=300)")
    ax1.set_ylim(0.635, 0.678)
    ax1.set_xticks(k1)
    ax1.set_title("Community × Depth\n(superadditivity near-additive at k=15)")
    ax1.legend(fontsize=7.5, framealpha=0.9)

    # Pruning × depth
    k2 = [10, 20]
    no_prune = [0.659, None]
    with_prune = [0.652, 0.674]
    ax2.plot(k2[:1], no_prune[:1], "s--", color=ORANGE, lw=1.8, markersize=6, label="no pruning")
    ax2.plot(k2, with_prune, "o-", color=BLUE, lw=1.8, markersize=6, label="pruning (cosine≥0.92)")
    ax2.plot([20], [0.669], "x", color=GREY, markersize=8, label="additive prediction (k=20)", zorder=5)
    for k, v in zip([10, 10, 20], [0.659, 0.652, 0.674]):
        ax2.text(k + 0.2, v + 0.0015, f"{v:.3f}", ha="left" if k < 15 else "center", fontsize=8)
    ax2.set_xlabel("k")
    ax2.set_ylim(0.635, 0.685)
    ax2.set_xticks(k2)
    ax2.set_title("Pruning × Depth\n(superadditivity +0.005 at k=20)")
    ax2.legend(fontsize=7.5, framealpha=0.9)

    fig.suptitle("Figure 3 — Superadditivity Requires Pruning at k=20 (GRB grid, n=300)\n"
                 "× = additive prediction; actual exceeds prediction only with pruning", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("docs/figures/fig6_superadditivity.png")
    plt.close(fig)
    print("fig6 done")


if __name__ == "__main__":
    fig_grb_leaderboard()
    fig_waterfall()
    fig_srr_model()
    fig_hare_heatmap()
    fig_k_depth()
    fig_superadditivity()
    print("All figures written to docs/figures/")
