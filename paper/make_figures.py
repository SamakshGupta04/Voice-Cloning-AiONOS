#!/usr/bin/env python3
"""
make_figures.py
===============
Regenerate every figure in the paper directly from ``outputs/metrics.json``.

Run:
    ../.venv-metrics/bin/python make_figures.py          # from paper/
    .venv-metrics/bin/python paper/make_figures.py       # from repo root

All figures are written as vector PDFs into ``paper/figures/`` and are picked up
by ``main.tex``. Nothing here is hand-typed: every number plotted is read from
the metrics file, so re-running after recomputing metrics keeps the paper and the
data in sync.

Design choices (kept consistent across every figure):
  * One fixed colour per system, reused everywhere, so identity never depends on
    reading a caption twice. Colours are colourblind-safe (blue / orange / violet,
    validated worst-adjacent CVD delta-E ~= 97).
  * A distinct hatch per system as a second, colour-independent cue, so the
    figures survive greyscale printing.
  * Direct value labels on bars rather than a dense value grid.
  * English (n=3) is shown as individual dots, never a box plot; a box on three
    points would imply a distribution that is not there. Hindi (n=9) is shown as
    dots over a light box.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
METRICS = os.path.join(REPO, "outputs", "metrics.json")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# --- house style -------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# system -> (display name, colour, hatch)
SYS = {
    "orpheus":   ("Orpheus-3B", "#2a78d6", "///"),
    "voxcpm2":   ("VoxCPM2",    "#eb6834", ".."),
    "vibevoice": ("VibeVoice",  "#4a3aa7", "xx"),
}
ORDER = ["orpheus", "voxcpm2", "vibevoice"]
GRID = dict(color="0.85", linewidth=0.6)


def load():
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)["models"]


def load_ref(field="pitch_mean_hz"):
    with open(METRICS, "r", encoding="utf-8") as f:
        return json.load(f)["reference_metrics"][field]


def clips(models, key, lang=None):
    cs = list(models[key]["clips"].values())
    if lang:
        cs = [c for c in cs if c["language"] == lang]
    return cs


def mean(cs, field):
    vals = [c[field] for c in cs if isinstance(c.get(field), (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


def vals(cs, field):
    return [c[field] for c in cs if isinstance(c.get(field), (int, float))]


def label_bars(ax, bars, fmt="{:.3f}", dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=7.5)


# =============================================================================
# Figure 1: headline comparison -- similarity, intelligibility, naturalness
# =============================================================================
def fig_overview(models):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    x = np.arange(len(ORDER))
    names = [SYS[k][0] for k in ORDER]
    cols = [SYS[k][1] for k in ORDER]
    hs = [SYS[k][2] for k in ORDER]

    # (a) Speaker similarity (SECS), all clips
    ax = axes[0]
    secs = [mean(clips(models, k), "secs") for k in ORDER]
    bars = ax.bar(x, secs, color=cols, edgecolor="white", linewidth=0.6)
    for b, h in zip(bars, hs):
        b.set_hatch(h)
    label_bars(ax, bars, "{:.3f}", dy=0.003)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("SECS (cosine)  $\\uparrow$")
    ax.set_title("(a) Speaker similarity")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)

    # (b) Intelligibility: English WER vs Hindi CER
    ax = axes[1]
    w = 0.38
    en = [mean(clips(models, k, "English"), "wer") for k in ORDER]
    hi = [mean(clips(models, k, "Hindi"), "cer") for k in ORDER]
    b1 = ax.bar(x - w / 2, en, w, color=cols, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, hi, w, color=cols, edgecolor="white",
                linewidth=0.6, alpha=0.55)
    for b, h in zip(b1, hs):
        b.set_hatch(h)
    for b, h in zip(b2, hs):
        b.set_hatch(h)
    label_bars(ax, b1, "{:.2f}", dy=0.01)
    label_bars(ax, b2, "{:.2f}", dy=0.01)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Error rate  $\\downarrow$")
    ax.set_title("(b) Intelligibility")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
    # legend for the solid/faded pairing -- single row across the top
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.35", label="English WER"),
                       Patch(facecolor="0.35", alpha=0.55, label="Hindi CER")],
              loc="upper center", ncol=2, frameon=False,
              handlelength=1.1, columnspacing=1.2)

    # (c) Naturalness (UTMOS), all clips
    ax = axes[2]
    ut = [mean(clips(models, k), "utmos") for k in ORDER]
    bars = ax.bar(x, ut, color=cols, edgecolor="white", linewidth=0.6)
    for b, h in zip(bars, hs):
        b.set_hatch(h)
    label_bars(ax, bars, "{:.2f}", dy=0.02)
    ax.set_ylim(3.0, 4.2)
    ax.set_ylabel("UTMOS (1--5)  $\\uparrow$")
    ax.set_title("(c) Naturalness")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.4)
    out = os.path.join(FIGDIR, "fig_overview.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# =============================================================================
# Figure 2: identity / naturalness trade-off, split by language
# =============================================================================
def fig_tradeoff(models):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    markers = {"English": "o", "Hindi": "s"}
    for k in ORDER:
        name, col, _ = SYS[k]
        for lang, mk in markers.items():
            cs = clips(models, k, lang)
            ax.scatter(mean(cs, "secs"), mean(cs, "utmos"),
                       s=70, marker=mk, color=col, edgecolor="white",
                       linewidth=0.8, zorder=3)
    ax.set_xlabel("Speaker similarity (SECS)  $\\uparrow$")
    ax.set_ylabel("Naturalness (UTMOS)  $\\uparrow$")
    ax.grid(True, **GRID); ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    sys_handles = [Line2D([], [], marker="o", linestyle="", color=SYS[k][1],
                          markeredgecolor="white", label=SYS[k][0]) for k in ORDER]
    lang_handles = [
        Line2D([], [], marker="o", linestyle="", color="0.4",
               markeredgecolor="white", label="English"),
        Line2D([], [], marker="s", linestyle="", color="0.4",
               markeredgecolor="white", label="Hindi"),
    ]
    leg1 = ax.legend(handles=sys_handles, loc="lower right", frameon=False,
                     fontsize=7.5, title="System", title_fontsize=7.5)
    ax.add_artist(leg1)
    ax.legend(handles=lang_handles, loc="upper left", frameon=False,
              fontsize=7.5, title="Language", title_fontsize=7.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_tradeoff.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# =============================================================================
# Figure 3: per-clip spread -- SECS (all) and Hindi CER
# =============================================================================
def fig_distributions(models):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    x = np.arange(len(ORDER))
    names = [SYS[k][0] for k in ORDER]

    rng = np.random.default_rng(0)

    # (a) SECS across all 12 clips
    ax = axes[0]
    for i, k in enumerate(ORDER):
        col = SYS[k][1]
        y = vals(clips(models, k), "secs")
        jx = i + (rng.random(len(y)) - 0.5) * 0.22
        ax.scatter(jx, y, s=26, color=col, edgecolor="white", linewidth=0.5,
                   zorder=3, alpha=0.9)
        ax.hlines(np.mean(y), i - 0.2, i + 0.2, color=col, linewidth=2.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("SECS (cosine)  $\\uparrow$")
    ax.set_title("(a) Speaker similarity, per clip (n=12)")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
    ax.set_ylim(0.82, 0.98)

    # (b) Hindi CER across the 9 Hindi clips
    ax = axes[1]
    for i, k in enumerate(ORDER):
        col = SYS[k][1]
        y = vals(clips(models, k, "Hindi"), "cer")
        jx = i + (rng.random(len(y)) - 0.5) * 0.22
        ax.scatter(jx, y, s=26, color=col, edgecolor="white", linewidth=0.5,
                   zorder=3, alpha=0.9)
        ax.hlines(np.mean(y), i - 0.2, i + 0.2, color=col, linewidth=2.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Hindi CER  $\\downarrow$")
    ax.set_title("(b) Hindi intelligibility, per clip (n=9)")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
    ax.set_ylim(0, 0.9)

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(FIGDIR, "fig_distributions.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# =============================================================================
# Figure 4: pitch fidelity -- |Delta F0| vs reference
# =============================================================================
def fig_pitch(models):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(ORDER))
    names = [SYS[k][0] for k in ORDER]
    cols = [SYS[k][1] for k in ORDER]
    hs = [SYS[k][2] for k in ORDER]
    pf = [mean(clips(models, k), "pitch_mean_diff_hz") for k in ORDER]
    bars = ax.bar(x, pf, color=cols, edgecolor="white", linewidth=0.6)
    for b, h in zip(bars, hs):
        b.set_hatch(h)
    label_bars(ax, bars, "{:.1f}", dy=0.2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("$|\\Delta F_0|$ vs. reference (Hz)  $\\downarrow$")
    ax.set_title("Mean pitch deviation from the reference voice")
    ax.set_ylim(0, max(pf) * 1.25)
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_pitch.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# =============================================================================
# Figure 5: cross-lingual stability -- how identity and naturalness move from
# English to Hindi within each system (the dissociation of Section V)
# =============================================================================
def fig_crosslingual(models):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    xpos = {"English": 0.0, "Hindi": 1.0}

    def slope_panel(ax, field, ylabel, title, fmt):
        for k in ORDER:
            name, col, _ = SYS[k]
            en = mean(clips(models, k, "English"), field)
            hi = mean(clips(models, k, "Hindi"), field)
            ax.plot([xpos["English"], xpos["Hindi"]], [en, hi],
                    color=col, linewidth=1.8, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                    label=name)
            ax.annotate(fmt.format(en), (xpos["English"], en),
                        textcoords="offset points", xytext=(-6, 0),
                        ha="right", va="center", fontsize=7, color=col)
            ax.annotate(fmt.format(hi), (xpos["Hindi"], hi),
                        textcoords="offset points", xytext=(6, 0),
                        ha="left", va="center", fontsize=7, color=col)
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["English", "Hindi"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)

    slope_panel(axes[0], "secs", "SECS (cosine)  $\\uparrow$",
                "(a) Speaker identity", "{:.3f}")
    axes[0].set_ylim(0.86, 0.97)
    slope_panel(axes[1], "utmos", "UTMOS (1--5)  $\\uparrow$",
                "(b) Naturalness", "{:.2f}")
    axes[1].set_ylim(3.2, 4.1)
    axes[1].legend(loc="lower left", frameon=False, fontsize=7.5)

    fig.tight_layout(w_pad=2.4)
    out = os.path.join(FIGDIR, "fig_crosslingual.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


# =============================================================================
# Figure 6: pitch register -- per-clip mean F0 against the reference voice,
# showing the *direction* of the shift that |Delta F0| (absolute) hides
# =============================================================================
def fig_pitch_register(models):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(ORDER))
    names = [SYS[k][0] for k in ORDER]
    rng = np.random.default_rng(1)

    ref = load_ref()
    ax.axhline(ref, color="0.45", linewidth=1.1, linestyle="--", zorder=2)
    ax.annotate("reference {:.0f} Hz".format(ref), (len(ORDER) - 0.5, ref),
                textcoords="offset points", xytext=(0, 3), ha="right",
                va="bottom", fontsize=7, color="0.35")

    for i, k in enumerate(ORDER):
        col = SYS[k][1]
        y = vals(clips(models, k), "pitch_mean_hz")
        jx = i + (rng.random(len(y)) - 0.5) * 0.22
        ax.scatter(jx, y, s=26, color=col, edgecolor="white", linewidth=0.5,
                   zorder=3, alpha=0.9)
        ax.hlines(np.mean(y), i - 0.2, i + 0.2, color=col, linewidth=2.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("Mean $F_0$ per clip (Hz)")
    ax.set_title("Pitch register vs. the reference voice")
    ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_pitch_register.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    models = load()
    fig_overview(models)
    fig_tradeoff(models)
    fig_distributions(models)
    fig_pitch(models)
    fig_crosslingual(models)
    fig_pitch_register(models)
    print("all figures ->", FIGDIR)
