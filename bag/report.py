"""PDF report generator for BAG output.

Produces a multi-page PDF that documents everything needed to defend the grades:
  - Header with tool name, timestamp, app version
  - Configuration summary (Q, prior, MCMC settings)
  - Diagnostic summary (plain English + raw numbers)
  - The grade table
  - Key figures (expected grades, mixture components)

No external dependencies beyond matplotlib + numpy.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

from .sampler import GraderConfig, diagnose, nearest_granulated_label
from .visualize import (
    GRADE_COLORS,
    figure_expected_grade_vs_score,
    figure_expected_grades,
    figure_mixture_components,
    figure_probability_curves,
)


APP_VERSION = "0.1.0"


def _fig_header(title_text: str, subtitle: str) -> plt.Figure:
    """Make a title page as a matplotlib figure."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.85, title_text,
            ha="center", va="top", fontsize=22, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.79, subtitle,
            ha="center", va="top", fontsize=12, color="#555",
            transform=ax.transAxes)
    ax.text(0.5, 0.06,
            f"Generated with BAG Grader v{APP_VERSION}  •  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center", va="bottom", fontsize=8, color="#888",
            transform=ax.transAxes)
    return fig


def _fig_summary(
    scores: np.ndarray, cfg: GraderConfig, diag: Dict, out: Dict,
) -> plt.Figure:
    """Summary page: configuration and diagnostics."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = []
    lines.append(("Run configuration", None, "header"))
    lines.append(("Number of students",              f"{scores.size}", None))
    lines.append(("Effective exam items (Q)",        f"{cfg.Q}", None))
    lines.append(("Expected grade distribution (π)", "  ".join(
        f"{g}={p:.2f}" for g, p in zip(list("ABCDE"), cfg.pi0)), None))
    lines.append(("π locked to this distribution?",   "Yes" if cfg.fix_pi else "No (learned from data)", None))
    lines.append(("Prior concentration on π (K_π)",   f"{cfg.K_pi:.2f}", None))
    lines.append(("Type-1 error (E[1-p₀])",           f"{cfg.type1_error:.3f}", None))
    lines.append(("p₁ prior mean",                     f"{cfg.p1_mean:.3f}", None))
    lines.append(("MCMC chains × iterations",          f"{cfg.n_chains} × {cfg.iters_per_chain}", None))
    lines.append(("Burn-in proportion",                f"{cfg.burn_in_prop:.2f}", None))
    lines.append(("", None, "blank"))

    lines.append(("Convergence diagnostics", None, "header"))
    status_map = {
        "good": "✓ Converged cleanly — results trustworthy",
        "ok":   "~ Minor instability — results usable",
        "poor": "✗ Poor convergence — re-run with more iterations",
        "unknown": "? Could not evaluate",
    }
    lines.append(("Status",                   status_map.get(diag["status"], "?"), None))
    lines.append(("Max split-R̂",               f"{diag['max_rhat']:.4f} (target ≤ 1.01)", None))
    lines.append(("Min effective sample size", f"{diag['min_ess']:.0f} (target ≥ 400)", None))
    lines.append(("Proposal acceptance rate",   f"{diag['accept_rate']:.3f} (target ≈ 0.25)", None))
    lines.append(("", None, "blank"))

    lines.append(("Grade distribution in this class", None, "header"))
    mode_label = out["mode_label"]
    for k, g in enumerate("ABCDE"):
        n = int(np.sum(mode_label == k))
        pct = 100.0 * n / scores.size
        lines.append((f"  {g}", f"{n} students ({pct:.1f}%)", None))

    y = 0.96
    for label, value, kind in lines:
        if kind == "header":
            ax.text(0.05, y, label, fontsize=13, fontweight="bold",
                    color="#222", transform=ax.transAxes)
            y -= 0.025
            ax.plot([0.05, 0.95], [y + 0.015, y + 0.015],
                    color="#ccc", linewidth=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.015
        elif kind == "blank":
            y -= 0.025
        else:
            ax.text(0.08, y, label, fontsize=10, color="#333",
                    transform=ax.transAxes)
            if value is not None:
                ax.text(0.62, y, value, fontsize=10,
                        color="#333", family="monospace",
                        transform=ax.transAxes)
            y -= 0.024

    # Explanation box
    y -= 0.03
    explain = (
        "About the method. Bayesian Adaptive Grading (BAG) models the class score "
        "distribution as a 5-component truncated-normal mixture. Each component "
        "corresponds to one of the five letter types (A–E). The Metropolis-within-"
        "Gibbs sampler infers the posterior distribution over the mixture parameters. "
        "Each student receives a full posterior over grades; the reported letter is "
        "the granulated grade closest to the posterior-mean expected grade."
    )
    ax.text(0.05, y - 0.04, explain,
            fontsize=9, color="#555", wrap=True,
            transform=ax.transAxes, verticalalignment="top")
    ax.annotate("", xy=(0.95, y - 0.005), xytext=(0.05, y - 0.005),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5))
    return fig


def _fig_grade_table(
    scores: np.ndarray, out: Dict, student_ids: Optional[List[str]] = None,
) -> plt.Figure:
    """Table page listing every student with score, letter, expected grade, confidence."""
    resp = out["resp_mean"]
    exp_grade = out["expected_grade"]
    gran = nearest_granulated_label(exp_grade)
    mode_prob = resp.max(axis=1)

    ids = student_ids if student_ids is not None else [f"{i+1}" for i in range(scores.size)]

    # Sort by expected grade descending
    order = np.argsort(-exp_grade)

    fig_h = max(4.0, 0.30 * scores.size + 2.0)
    fig = plt.figure(figsize=(8.5, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.985, "Grade assignments",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.955,
            "Sorted by expected grade value (highest first)",
            ha="center", va="top", fontsize=9, color="#666",
            transform=ax.transAxes)

    headers = ["Student", "Score", "Grade", "Expected", "P(top)", "Flag"]
    col_widths = [0.25, 0.12, 0.13, 0.15, 0.13, 0.10]

    tab = Table(ax, bbox=[0.05, 0.05, 0.9, 0.88])
    row_h = 0.92 / (len(order) + 1)

    # Header row
    x_pos = 0.0
    for header, w in zip(headers, col_widths):
        c = tab.add_cell(0, len(tab._cells), w, row_h, text=header,
                         loc="center", facecolor="#e8e8e8")
        c.get_text().set_fontweight("bold")
        c.get_text().set_fontsize(9)

    for row_i, i in enumerate(order, start=1):
        if mode_prob[i] >= 0.80:
            flag = "●"
            flag_color = "#2ca02c"
        elif mode_prob[i] >= 0.60:
            flag = "●"
            flag_color = "#dbba00"
        else:
            flag = "●"
            flag_color = "#d62728"

        cells_data = [
            (str(ids[i]), None),
            (f"{scores[i]:g}", None),
            (gran[i], GRADE_COLORS.get(str(gran[i]), None)),
            (f"{exp_grade[i]:.2f}", None),
            (f"{mode_prob[i]:.2f}", None),
            (flag, flag_color),
        ]
        for col_j, ((txt, col), w) in enumerate(zip(cells_data, col_widths)):
            c = tab.add_cell(row_i, col_j,
                             w, row_h, text=txt, loc="center")
            c.get_text().set_fontsize(8)
            if col:
                if txt == "●":
                    c.get_text().set_color(col)
                    c.get_text().set_fontsize(14)
                else:
                    c.set_facecolor(col)
                    c.set_alpha(0.35)

    ax.add_table(tab)

    # Legend
    ax.text(0.05, 0.02,
            "Flag:  ● green = confident (>80%),  ● yellow = uncertain (60–80%),  "
            "● red = on the fence (<60%)",
            fontsize=8, color="#555", transform=ax.transAxes)
    return fig


def build_pdf_report(
    scores: np.ndarray,
    out: Dict,
    cfg: GraderConfig,
    student_ids: Optional[List[str]] = None,
    course_name: Optional[str] = None,
) -> bytes:
    """Build a multi-page PDF report and return its raw bytes."""
    diag = diagnose(out, cfg)
    buf = BytesIO()

    title = "Bayesian Adaptive Grading — Grade Report"
    subtitle = course_name if course_name else "Class grade justification document"

    with PdfPages(buf) as pdf:
        fig1 = _fig_header(title, subtitle)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = _fig_summary(scores, cfg, diag, out)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3 = _fig_grade_table(scores, out, student_ids=student_ids)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        for fn in [
            figure_mixture_components,
            figure_expected_grades,
            figure_expected_grade_vs_score,
            figure_probability_curves,
        ]:
            f = fn(scores, out, cfg)
            pdf.savefig(f, bbox_inches="tight")
            plt.close(f)

    return buf.getvalue()
