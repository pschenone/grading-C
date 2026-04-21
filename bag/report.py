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
from matplotlib.patches import Circle, Rectangle

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
            f"Generated with BAG Grader v{APP_VERSION}  \u2022  "
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
    lines.append(("Expected grade distribution (\u03c0)", "  ".join(
        f"{g}={p:.2f}" for g, p in zip(list("ABCDE"), cfg.pi0)), None))
    lines.append(("\u03c0 locked to this distribution?",   "Yes" if cfg.fix_pi else "No (learned from data)", None))
    lines.append(("Prior concentration on \u03c0 (K_\u03c0)",   f"{cfg.K_pi:.2f}", None))
    lines.append(("Type-1 error (E[1-p\u2080])",           f"{cfg.type1_error:.3f}", None))
    lines.append(("p\u2081 prior mean",                     f"{cfg.p1_mean:.3f}", None))
    lines.append(("MCMC chains \u00d7 iterations",          f"{cfg.n_chains} \u00d7 {cfg.iters_per_chain}", None))
    lines.append(("Burn-in proportion",                f"{cfg.burn_in_prop:.2f}", None))
    lines.append(("", None, "blank"))

    lines.append(("Convergence diagnostics", None, "header"))
    status_map = {
        "good": "\u2713 Converged cleanly \u2014 results trustworthy",
        "ok":   "~ Minor instability \u2014 results usable",
        "poor": "\u2717 Poor convergence \u2014 re-run with more iterations",
        "unknown": "? Could not evaluate",
    }
    lines.append(("Status",                   status_map.get(diag["status"], "?"), None))
    lines.append(("Max split-R\u0302",               f"{diag['max_rhat']:.4f} (target \u2264 1.01)", None))
    lines.append(("Min effective sample size", f"{diag['min_ess']:.0f} (target \u2265 400)", None))
    lines.append(("Proposal acceptance rate",   f"{diag['accept_rate']:.3f} (target \u2248 0.25)", None))
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
        "corresponds to one of the five letter types (A\u2013E). The Metropolis-within-"
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


# ----------------------------------------------------------------------
# Grade-assignments page — manual layout (no matplotlib.table.Table)
# ----------------------------------------------------------------------
#
# The earlier implementation used matplotlib.table.Table with a custom
# `bbox`, private `_cells` attribute, and per-cell `set_alpha`. That
# combination produces inconsistent rendering across matplotlib versions
# (notably on the Streamlit Cloud deployment, where body cells render
# empty). We replace it with explicit Rectangle/Circle/text primitives
# positioned in axes coordinates, which are version-stable.

def _flag_color(p: float) -> str:
    """Traffic-light color for confidence level p in [0, 1]."""
    if p >= 0.80:
        return "#2ca02c"   # green
    if p >= 0.60:
        return "#dbba00"   # yellow
    return "#d62728"       # red


def _fig_grade_table(
    scores: np.ndarray, out: Dict, student_ids: Optional[List[str]] = None,
) -> plt.Figure:
    """Table page listing every student with score, letter, expected grade, confidence.

    Rendered with primitive artists (Rectangle, Circle, text) rather than
    matplotlib.table.Table to keep output stable across mpl versions.
    """
    resp = out["resp_mean"]
    exp_grade = out["expected_grade"]
    gran = nearest_granulated_label(exp_grade)
    mode_prob = resp.max(axis=1)

    ids = student_ids if student_ids is not None \
        else [f"{i+1}" for i in range(scores.size)]

    # Sort by expected grade descending
    order = np.argsort(-exp_grade)
    n_rows = len(order)

    # Figure sizing: give each row a consistent vertical slot
    row_inches = 0.28
    header_inches = 0.55
    top_pad = 0.85      # room for title + subtitle
    bot_pad = 0.65      # room for legend
    fig_h = top_pad + header_inches + row_inches * n_rows + bot_pad
    fig_h = max(4.5, fig_h)

    fig = plt.figure(figsize=(8.5, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title + subtitle (in axes coords). Keep comfortable vertical separation.
    ax.text(0.5, 0.99, "Grade assignments",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.945,
            "Sorted by expected grade value (highest first)",
            ha="center", va="top", fontsize=9, color="#666",
            transform=ax.transAxes)

    # Vertical layout in axes coordinates
    y_top_legend = 0.03             # legend sits below this
    y_bot_legend = y_top_legend + 0.025
    y_table_bottom = y_bot_legend + 0.015
    y_table_top = 0.91              # top of header row (leaves room for title + subtitle)

    table_h = y_table_top - y_table_bottom
    header_h = table_h / (n_rows + 1.7)   # header slightly taller
    row_h = (table_h - header_h) / n_rows

    # Horizontal layout: columns sum to exactly 1.0 of (left .. right)
    left_edge = 0.04
    right_edge = 0.96
    inner_w = right_edge - left_edge
    col_fracs = np.array([0.25, 0.13, 0.14, 0.16, 0.15, 0.17])
    col_fracs = col_fracs / col_fracs.sum()   # defensive normalization
    col_widths = col_fracs * inner_w
    col_lefts = left_edge + np.concatenate([[0.0], np.cumsum(col_widths)[:-1]])
    col_centers = col_lefts + col_widths / 2.0

    headers = ["Student", "Score", "Grade", "Expected", "P(top)", "Flag"]

    # --- Header row background + text ---
    header_bottom = y_table_top - header_h
    ax.add_patch(Rectangle(
        (left_edge, header_bottom), inner_w, header_h,
        facecolor="#e8e8e8", edgecolor="#888888", linewidth=0.6,
        transform=ax.transAxes, zorder=1,
    ))
    for cx, h in zip(col_centers, headers):
        ax.text(cx, header_bottom + header_h / 2, h,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#222", transform=ax.transAxes, zorder=3)

    # --- Body rows ---
    for row_i, i in enumerate(order):
        y_row_top = header_bottom - row_i * row_h
        y_row_bot = y_row_top - row_h
        y_mid = (y_row_top + y_row_bot) / 2

        # Row outline
        ax.add_patch(Rectangle(
            (left_edge, y_row_bot), inner_w, row_h,
            facecolor="white", edgecolor="#bbbbbb", linewidth=0.4,
            transform=ax.transAxes, zorder=1,
        ))

        # Grade-column colored background (alpha baked into rgba, not set_alpha)
        grade_label = str(gran[i])
        grade_col = GRADE_COLORS.get(grade_label)
        if grade_col is not None:
            ax.add_patch(Rectangle(
                (col_lefts[2], y_row_bot), col_widths[2], row_h,
                facecolor=_blend_with_white(grade_col, alpha=0.35),
                edgecolor="#bbbbbb", linewidth=0.4,
                transform=ax.transAxes, zorder=2,
            ))

        # Text cells
        cell_texts = [
            str(ids[i]),
            f"{scores[i]:g}",
            grade_label,
            f"{exp_grade[i]:.2f}",
            f"{mode_prob[i]:.2f}",
        ]
        for cx, txt in zip(col_centers[:5], cell_texts):
            ax.text(cx, y_mid, txt,
                    ha="center", va="center", fontsize=8, color="#222",
                    transform=ax.transAxes, zorder=3)

        # Flag column: explicit circle patch, not a text bullet
        fcol = _flag_color(float(mode_prob[i]))
        # Circle radius in axes coords; scaled to fit comfortably
        r = min(col_widths[5] * 0.18, row_h * 0.35)
        ax.add_patch(Circle(
            (col_centers[5], y_mid), radius=r,
            facecolor=fcol, edgecolor="#333333", linewidth=0.5,
            transform=ax.transAxes, zorder=3,
        ))

    # Legend at bottom — three fixed columns so items never overlap
    legend_y = y_top_legend + 0.01
    ax.text(left_edge, legend_y,
            "Flag:",
            ha="left", va="center", fontsize=8, color="#555",
            transform=ax.transAxes)
    legend_items = [
        ("#2ca02c", "green = confident (>80%)"),
        ("#dbba00", "yellow = uncertain (60\u201380%)"),
        ("#d62728", "red = on the fence (<60%)"),
    ]
    # Evenly space the three items across the inner table width,
    # reserving a bit of room on the left for the "Flag:" label.
    legend_start = left_edge + 0.07
    legend_span = right_edge - legend_start
    slot_w = legend_span / len(legend_items)
    for idx, (color, label) in enumerate(legend_items):
        cx = legend_start + idx * slot_w
        ax.add_patch(Circle(
            (cx, legend_y), radius=0.008,
            facecolor=color, edgecolor="#333333", linewidth=0.5,
            transform=ax.transAxes, zorder=3,
        ))
        ax.text(cx + 0.013, legend_y, label,
                ha="left", va="center", fontsize=8, color="#555",
                transform=ax.transAxes)

    return fig


def _blend_with_white(hex_color: str, alpha: float) -> tuple:
    """Return an RGB tuple for `hex_color` alpha-blended onto white.

    Avoids relying on matplotlib's alpha compositing, which has caused
    rendering issues in the PDF backend across mpl versions.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    a = float(np.clip(alpha, 0.0, 1.0))
    return (r * a + (1 - a), g * a + (1 - a), b * a + (1 - a))


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

    title = "Bayesian Adaptive Grading \u2014 Grade Report"
    subtitle = course_name if course_name else "Class grade justification document"

    with PdfPages(buf) as pdf:
        fig1 = _fig_header(title, subtitle)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = _fig_summary(scores, cfg, diag, out)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3 = _fig_grade_table(scores, out, student_ids=student_ids)
        # NOTE: do NOT use bbox_inches="tight" on the grade table page.
        # Tight bbox re-computes a crop that can miss patches placed via
        # axes-coordinate transforms in certain mpl/backend combinations.
        pdf.savefig(fig3)
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
