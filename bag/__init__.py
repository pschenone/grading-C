"""Bayesian Adaptive Grading — Python package.

This package implements the BAG method from Schenone (2025),
"The Hitchhiker's Guide to the Grading Galaxy".
"""

from .sampler import (
    GRADE_LABELS,
    GRADE_VALUES,
    GRANULATED_GRID,
    GRANULATED_LABELS,
    GraderConfig,
    compute_cutoffs,
    default_config_for_class,
    diagnose,
    nearest_granulated_label,
    run_bag_sampler,
    run_bag_sampler_with_retry,
)
from .report import build_pdf_report
from .visualize import (
    figure_expected_grades,
    figure_grade_heatmap,
    figure_mixture_components,
    figure_probability_curves,
)

__version__ = "0.2.0"

__all__ = [
    "GRADE_LABELS", "GRADE_VALUES", "GRANULATED_GRID", "GRANULATED_LABELS",
    "GraderConfig", "compute_cutoffs", "default_config_for_class", "diagnose",
    "nearest_granulated_label", "run_bag_sampler", "run_bag_sampler_with_retry",
    "build_pdf_report",
    "figure_expected_grades", "figure_grade_heatmap",
    "figure_mixture_components", "figure_probability_curves",
]
