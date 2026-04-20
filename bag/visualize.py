"""Visualization for BAG sampler output.

Generates the 4 key figures for the web UI (matching the paper's style):
  1. Expected grades with uncertainty (sorted bar chart)
  2. Mixture components over score distribution
  3. Grade probability transition curves
  4. Grade heatmap per student

All figures use matplotlib, return Figure objects, and are safe to call
repeatedly in a Streamlit context.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.special import ndtr

from .sampler import (
    GRADE_LABELS,
    GRADE_VALUES,
    GRANULATED_GRID,
    GRANULATED_LABELS,
    GraderConfig,
    compute_cutoffs,
    nearest_granulated_label,
)

# Matches the color scheme from Final.py
COMPONENT_COLORS = ["#d62728", "#ff7f0e", "#41b6c4", "#225ea8", "#0c2c84"]
GRADE_COLORS = {
    "A":  "#d62728", "A-": "#ff7f0e",
    "B+": "#ffbb78", "B":  "#ffd699", "B-": "#ffffb3",
    "C+": "#c7e9b4", "C":  "#7fcdbb", "C-": "#41b6c4",
    "D+": "#1d91c0", "D":  "#225ea8",
    "E":  "#0c2c84",
}

EPS = 1e-6


def _setup_style():
    """Consistent academic-clean figure style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })


def _truncnorm_pdf(x, mu, sigma, a=0.0, b=100.0):
    """Truncated-normal PDF on [a, b]."""
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    Z = np.maximum(ndtr(b_std) - ndtr(a_std), 1e-300)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi) * Z)


def _responsibilities_from_params(x, pi, mu, sigma, a=0.0, b=100.0):
    """Compute responsibilities r(x) for a grid of x values."""
    fx = np.column_stack([_truncnorm_pdf(x, mu[k], sigma[k], a, b) for k in range(5)])
    num = fx * np.asarray(pi)[None, :]
    den = np.clip(num.sum(axis=1, keepdims=True), 1e-300, None)
    return num / den


def figure_expected_grades(
    scores: np.ndarray, out: Dict, cfg: GraderConfig,
) -> plt.Figure:
    """Figure 1: Expected grades with 95% HDI, sorted."""
    _setup_style()

    exp_grade = out["expected_grade"]
    gran_lab = nearest_granulated_label(exp_grade)
    gran_lab = np.asarray(gran_lab, dtype=str)

    sort_idx = np.argsort(exp_grade)
    exp_sorted = exp_grade[sort_idx]
    gran_sorted = gran_lab[sort_idx]
    colors_sorted = [GRADE_COLORS.get(g, "#808080") for g in gran_sorted]

    # Per-draw expected grades for uncertainty
    T = out["pi_samples"].shape[0]
    exp_per_draw = np.zeros((T, scores.size))
    for t in range(T):
        r_t = _responsibilities_from_params(
            scores, out["pi_samples"][t], out["mu_samples"][t],
            out["sigma_samples"][t], cfg.trunc_a, cfg.trunc_b,
        )
        exp_per_draw[t] = r_t @ GRADE_VALUES
    lo = np.percentile(exp_per_draw, 2.5, axis=0)[sort_idx]
    hi = np.percentile(exp_per_draw, 97.5, axis=0)[sort_idx]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    x_pos = np.arange(len(exp_sorted))

    ax.fill_between(x_pos, lo, hi, alpha=0.25, color="gray", label="95% HDI")
    ax.scatter(x_pos, exp_sorted, c=colors_sorted, s=80,
               edgecolors="black", linewidths=1, zorder=3)

    for thresh, label in zip(GRANULATED_GRID, GRANULATED_LABELS):
        ax.axhline(thresh, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.text(-1.5, thresh, label, ha="right", va="center",
                fontsize=8, color="gray")

    ax.set_xlabel("Student (sorted by expected grade)")
    ax.set_ylabel("Expected grade value")
    ax.set_title("Expected grades with uncertainty")
    ax.set_xlim(-2.5, len(exp_sorted))
    ax.set_ylim(-0.2, 4.2)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    return fig


def figure_mixture_components(
    scores: np.ndarray, out: Dict, cfg: GraderConfig,
) -> plt.Figure:
    """Figure 2: Histogram + mixture components + model-based grade regions."""
    _setup_style()

    pi_mean = out["pi_samples"].mean(axis=0)
    mu_mean = out["mu_samples"].mean(axis=0)
    sg_mean = np.maximum(out["sigma_samples"].mean(axis=0), EPS)
    boundaries = compute_cutoffs(pi_mean, mu_mean, sg_mean, cfg.trunc_a, cfg.trunc_b)

    xgrid = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    ax.hist(scores, bins=20, density=True, alpha=0.35, color="lightgray",
            edgecolor="black", label="Score histogram")

    # Grade regions
    grade_regions = [
        (boundaries[0], 100, "A", "#ffe6e6"),
        (boundaries[1], boundaries[0], "B", "#fff9e6"),
        (boundaries[2], boundaries[1], "C", "#d9f2e6"),
        (boundaries[3], boundaries[2], "D", "#d4e5f7"),
        (0, boundaries[3], "E", "#cce0f4"),
    ]
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.05
    for x_start, x_end, label, color in grade_regions:
        ax.axvspan(x_start, x_end, alpha=0.18, color=color)

    # Component PDFs
    mixture_pdf = np.zeros_like(xgrid)
    for k in range(5):
        pdf_k = pi_mean[k] * _truncnorm_pdf(xgrid, mu_mean[k], sg_mean[k])
        mixture_pdf += pdf_k
        ax.plot(xgrid, pdf_k, linewidth=2, color=COMPONENT_COLORS[k],
                label=f"Component {GRADE_LABELS[k]}")

    ax.plot(xgrid, mixture_pdf, "k--", linewidth=2.2, label="Total mixture")

    # Label regions
    for x_start, x_end, label, _ in grade_regions:
        mid = 0.5 * (x_start + x_end)
        ax.text(mid, ax.get_ylim()[1] * 0.94, label,
                ha="center", va="top", fontsize=13, fontweight="bold",
                color="dimgray")

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title("Mixture components with model-based grade regions")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def figure_probability_curves(
    scores: np.ndarray, out: Dict, cfg: GraderConfig,
) -> plt.Figure:
    """Figure 3: P(grade k | x) curves with boundaries and score rug."""
    _setup_style()

    pi_mean = out["pi_samples"].mean(axis=0)
    mu_mean = out["mu_samples"].mean(axis=0)
    sg_mean = np.maximum(out["sigma_samples"].mean(axis=0), EPS)
    boundaries = compute_cutoffs(pi_mean, mu_mean, sg_mean, cfg.trunc_a, cfg.trunc_b)

    xgrid = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    denom = np.zeros_like(xgrid)
    for j in range(5):
        denom += pi_mean[j] * _truncnorm_pdf(xgrid, mu_mean[j], sg_mean[j])

    for k in range(5):
        num = pi_mean[k] * _truncnorm_pdf(xgrid, mu_mean[k], sg_mean[k])
        probs = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        ax.plot(xgrid, probs, linewidth=2.2, color=COMPONENT_COLORS[k],
                label=f"P({GRADE_LABELS[k]} | score)")

    for i, b in enumerate(boundaries):
        ax.axvline(b, color="gray", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.text(b, -0.16,
                f"{GRADE_LABELS[i+1]}|{GRADE_LABELS[i]}\n{b:.1f}",
                ha="center", fontsize=8, fontweight="bold")

    # Rug
    ax.scatter(scores, [-0.025] * len(scores), marker="|", s=80,
               c="black", alpha=0.65, linewidths=1.5)

    ax.set_xlabel("Score")
    ax.set_ylabel("Probability")
    ax.set_title("Grade probability as a function of score")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.28, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def figure_grade_heatmap(
    scores: np.ndarray, out: Dict, cfg: GraderConfig,
    student_ids: Optional[list] = None,
) -> plt.Figure:
    """Figure 4: Per-student posterior probability heatmap."""
    _setup_style()

    resp = out["resp_mean"]  # (N, 5)
    exp_grade = out["expected_grade"]
    sort_idx = np.argsort(-exp_grade)  # highest on top
    resp_sorted = resp[sort_idx]
    scores_sorted = scores[sort_idx]
    ids_sorted = (
        [student_ids[i] for i in sort_idx]
        if student_ids is not None
        else [f"Student {i+1}" for i in sort_idx]
    )

    N = resp_sorted.shape[0]
    fig_h = max(4.0, 0.25 * N + 2.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor("white")

    cmap = LinearSegmentedColormap.from_list(
        "bagheat", ["#ffffff", "#225ea8"]
    )
    im = ax.imshow(resp_sorted, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(GRADE_LABELS)
    ax.set_yticks(range(N))
    ax.set_yticklabels([f"{ids_sorted[i]} ({scores_sorted[i]:g})" for i in range(N)],
                       fontsize=8)
    ax.set_xlabel("Grade")
    ax.set_title("Posterior probability of each grade, per student")

    # Annotate cells with probability > 0.05
    for i in range(N):
        for j in range(5):
            p = resp_sorted[i, j]
            if p > 0.05:
                color = "white" if p > 0.55 else "black"
                ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

    fig.colorbar(im, ax=ax, label="Probability", fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig
