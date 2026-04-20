"""BAG Grader — Home / main tool page.

Paste scores, optionally tweak knobs, click compute, get grades.
"""

from __future__ import annotations

import io
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from bag import (
    GRADE_LABELS,
    GRADE_VALUES,
    GraderConfig,
    build_pdf_report,
    compute_cutoffs,
    default_config_for_class,
    diagnose,
    figure_expected_grades,
    figure_grade_heatmap,
    figure_mixture_components,
    figure_probability_curves,
    nearest_granulated_label,
    run_bag_sampler,
    run_bag_sampler_with_retry,
)

# ======================================================================
# Page config + custom CSS
# ======================================================================

st.set_page_config(
    page_title="BAG Grader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.6rem;
    }
    .badge-good  { background:#e6f4ea; color:#137333; padding:4px 10px; border-radius:6px; font-weight:600; }
    .badge-ok    { background:#fef7e0; color:#b06000; padding:4px 10px; border-radius:6px; font-weight:600; }
    .badge-poor  { background:#fce8e6; color:#a50e0e; padding:4px 10px; border-radius:6px; font-weight:600; }
    .alex-blake-box {
        background: #f8f9fb;
        border-left: 3px solid #225ea8;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 0.8rem 0 1.4rem 0;
        font-size: 0.95rem;
        color: #333;
    }
    .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        font-size: 0.8rem;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# Session state defaults
# ======================================================================

for key, default in [
    ("out", None),
    ("cfg", None),
    ("scores_arr", None),
    ("ids", None),
    ("diag", None),
    ("last_inputs_hash", None),
]:
    st.session_state.setdefault(key, default)


# ======================================================================
# Hero
# ======================================================================

st.markdown(
    '<div class="hero-title">📊 BAG Grader</div>'
    '<div class="hero-subtitle">'
    "Turn numerical scores into letter grades using Bayesian Adaptive Grading."
    "</div>",
    unsafe_allow_html=True,
)

with st.container():
    st.markdown(
        """
        <div class="alex-blake-box">
        Alex scores 75, Blake scores 65. Is that 10-point gap strong enough
        evidence that Alex and Blake belong in different grade categories?
        It depends on the exam structure, the per-item difficulty, and the
        overall distribution of scores in the class. BAG models all three jointly,
        then reports a full posterior probability over grades for every student.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================================
# Input zone
# ======================================================================

st.subheader("1. Your class scores")

input_tab, upload_tab, sample_tab = st.tabs(
    ["Paste scores", "Upload CSV", "Try sample data"]
)

scores: Optional[np.ndarray] = None
student_ids: Optional[List[str]] = None

with input_tab:
    txt = st.text_area(
        "One score per line (0–100). Optionally add a comma and an ID: "
        "e.g. `82.5, Alex`",
        height=180,
        placeholder="67.5\n77.5\n92.5\n80\n52.5\n70\n...",
        key="scores_textarea",
    )
    if txt.strip():
        try:
            rows = [r.strip() for r in txt.splitlines() if r.strip()]
            sc_list, id_list = [], []
            for i, r in enumerate(rows):
                if "," in r:
                    parts = [p.strip() for p in r.split(",", 1)]
                    sc_list.append(float(parts[0]))
                    id_list.append(parts[1] if len(parts) > 1 and parts[1] else f"Student {i+1}")
                else:
                    sc_list.append(float(r))
                    id_list.append(f"Student {i+1}")
            scores = np.array(sc_list, dtype=float)
            student_ids = id_list
        except ValueError as e:
            st.error(f"Could not parse scores: {e}. Each line must be a number.")

with upload_tab:
    uploaded = st.file_uploader(
        "CSV file — one score per row, or two columns (ID, score). "
        "A header row is fine; it's auto-detected.",
        type=["csv"], key="csv_upload",
    )
    if uploaded is not None:
        try:
            # Try reading with auto header detection
            df = pd.read_csv(uploaded)
            # If the first column is numeric and we have exactly one column,
            # there was no header — re-read
            if df.shape[1] == 1:
                try:
                    scores = df.iloc[:, 0].astype(float).to_numpy()
                    student_ids = [f"Student {i+1}" for i in range(scores.size)]
                except (ValueError, TypeError):
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, header=None)
                    scores = df.iloc[:, 0].astype(float).to_numpy()
                    student_ids = [f"Student {i+1}" for i in range(scores.size)]
            elif df.shape[1] >= 2:
                # Find the numeric column
                numeric_col, id_col = None, None
                for c in df.columns:
                    try:
                        df[c].astype(float)
                        numeric_col = c
                        break
                    except (ValueError, TypeError):
                        pass
                if numeric_col is None:
                    # No header: re-read
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, header=None)
                    if df.shape[1] >= 2:
                        try:
                            scores = df.iloc[:, 1].astype(float).to_numpy()
                            student_ids = df.iloc[:, 0].astype(str).tolist()
                        except (ValueError, TypeError):
                            scores = df.iloc[:, 0].astype(float).to_numpy()
                            student_ids = df.iloc[:, 1].astype(str).tolist()
                else:
                    scores = df[numeric_col].astype(float).to_numpy()
                    other_cols = [c for c in df.columns if c != numeric_col]
                    if other_cols:
                        student_ids = df[other_cols[0]].astype(str).tolist()
                    else:
                        student_ids = [f"Student {i+1}" for i in range(scores.size)]
            st.success(f"Loaded {scores.size} scores.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

with sample_tab:
    st.caption(
        "A 33-student example from the paper's appendix (an intermediate "
        "econ class). Useful for trying out the tool without your own data."
    )
    if st.button("Load sample data", key="btn_sample"):
        st.session_state["_use_sample"] = True
    if st.session_state.get("_use_sample"):
        scores = np.array([
            25, 32.5, 35, 37.5, 37.5, 42.5, 43, 45, 45, 46, 47.5, 48.5,
            51.5, 52.5, 53.5, 53.5, 54, 54.5, 57.5, 57.5, 61, 62, 62.5,
            62.5, 65, 65, 70, 75, 80, 82.5, 82.5, 90, 94,
        ], dtype=float)
        student_ids = [f"Student {i+1}" for i in range(scores.size)]
        st.info(f"Sample data loaded: {scores.size} scores.")

# Validate
if scores is not None:
    if scores.size == 0:
        st.error("No scores provided.")
        scores = None
    elif np.any(scores < 0) or np.any(scores > 100):
        st.error("All scores must be between 0 and 100.")
        scores = None
    elif scores.size < 5:
        st.error("Need at least 5 scores to run the sampler.")
        scores = None
    elif scores.size < 15:
        st.warning(
            f"⚠️  You have {scores.size} students. With fewer than 15, the posterior "
            "is heavily influenced by your prior settings. Consider whether this "
            "is appropriate for your context."
        )

# ======================================================================
# Options
# ======================================================================

st.subheader("2. Options")

PI_PRESETS = {
    "Baseline business-school prior (default)": np.array(
        [0.165, 0.378, 0.373, 0.074, 0.010], float
    ),
    "Alternative A/B-heavy prior":              np.array(
        [0.27, 0.65, 0.056, 0.016, 0.008], float
    ),
    "Alternative symmetric bell prior":         np.array(
        [0.0668, 0.24173, 0.38292, 0.24173, 0.0667], float
    ),
    "Let the data decide (near-uniform prior)": np.array(
        [0.2, 0.2, 0.2, 0.2, 0.2], float
    ),
    "Skewed high (honors class)":               np.array(
        [0.30, 0.40, 0.22, 0.06, 0.02], float
    ),
    "Skewed low (intro / weed-out)":            np.array(
        [0.06, 0.18, 0.38, 0.28, 0.10], float
    ),
}

pi_preset_name = st.selectbox(
    "Expected grade distribution",
    list(PI_PRESETS.keys()),
    index=0,
    help=(
        "Your prior belief about the proportion of A/B/C/D/E students in the "
        "class, before seeing any scores. The data will update this prior. "
        "The default is the 'Baseline business-school prior' used in the "
        "paper's example; 'Let the data decide' starts near-uniform if you "
        "prefer the data to dominate."
    ),
    key="pi_preset_name",
)
pi_vec = PI_PRESETS[pi_preset_name].copy()

# If the preset just changed, overwrite the slider values in session state
# so the sliders in the Customize expander actually reflect the new preset.
if st.session_state.get("_last_preset") != pi_preset_name:
    st.session_state["pi_A"] = float(pi_vec[0])
    st.session_state["pi_B"] = float(pi_vec[1])
    st.session_state["pi_C"] = float(pi_vec[2])
    st.session_state["pi_D"] = float(pi_vec[3])
    st.session_state["pi_E"] = float(pi_vec[4])
    st.session_state["_last_preset"] = pi_preset_name

# --- Customize expander ---
with st.expander("Customize: exam structure and grade distribution"):
    col_q, col_wizard = st.columns([3, 2])

    with col_q:
        Q = st.number_input(
            "Effective number of exam items (Q)",
            min_value=4, max_value=200, value=60, step=1,
            help=(
                "The effective number of scorable items in your exam. For a "
                "multiple-choice exam, this is the number of questions. For an "
                "essay graded in 5-point increments on a 100-point scale, it's "
                "100/5 = 20. See the wizard on the right →"
            ),
        )
    with col_wizard:
        st.markdown("**Not sure what Q is?**")
        num_q = st.number_input("Questions on the exam",
                                min_value=1, value=10, step=1,
                                key="wizard_nq")
        scale = st.selectbox(
            "Each question graded in increments of",
            ["Full points", "Half points", "Third points",
             "Quarter points", "Fifth points"],
            key="wizard_scale",
        )
        scale_factor = {"Full points": 1, "Half points": 2, "Third points": 3,
                        "Quarter points": 4, "Fifth points": 5}[scale]
        suggested_Q = num_q * scale_factor
        st.caption(f"→ Suggested Q = **{suggested_Q}**")

    fix_pi = st.checkbox(
        "Enforce this distribution exactly (don't update π from data)",
        value=False,
        help=(
            "If checked, π is locked to your chosen distribution and will not "
            "be updated from the observed scores. Useful for institutions with "
            "strict quotas. Trade-off: you lose BAG's ability to adapt to the "
            "actual class."
        ),
    )

    st.markdown("**Custom π (optional)** — sliders override the preset above.")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        piA = st.slider("A", 0.0, 1.0, float(pi_vec[0]), 0.01, key="pi_A")
    with col_b:
        piB = st.slider("B", 0.0, 1.0, float(pi_vec[1]), 0.01, key="pi_B")
    with col_c:
        piC = st.slider("C", 0.0, 1.0, float(pi_vec[2]), 0.01, key="pi_C")
    with col_d:
        piD = st.slider("D", 0.0, 1.0, float(pi_vec[3]), 0.01, key="pi_D")
    with col_e:
        piE = st.slider("E", 0.0, 1.0, float(pi_vec[4]), 0.01, key="pi_E")

    pi_custom = np.array([piA, piB, piC, piD, piE], float)
    pi_sum = pi_custom.sum()
    if pi_sum > 0:
        pi_vec = pi_custom / pi_sum  # auto-normalize
        st.caption(
            f"Normalized π: A={pi_vec[0]:.3f}, B={pi_vec[1]:.3f}, "
            f"C={pi_vec[2]:.3f}, D={pi_vec[3]:.3f}, E={pi_vec[4]:.3f}"
        )

# --- Advanced expander ---
with st.expander("Advanced — you don't need this", expanded=False):
    st.info(
        "You don't need these. Defaults are well-calibrated. These exist "
        "for reproducibility of published results and for users who know "
        "what MCMC is."
    )
    col1, col2 = st.columns(2)
    with col1:
        type1_error = st.slider(
            "Type-1 error (probability a well-calibrated question is missed)",
            0.01, 0.30, 0.10, 0.01,
            help="E[1 - p₀]. The probability that a type-θ student misses a "
                 "question calibrated for their own type. Together with the two "
                 "tail tolerances below, this fully determines the prior on "
                 "(p₀, p₁).",
        )
        tail_v = st.slider(
            "Tail tolerance for (p₀, p₁) prior — TAIL",
            0.001, 0.10, 0.01, 0.001, format="%.3f",
            help="Controls concentration of the Dirichlet prior on "
                 "(p₁, p₀-p₁, 1-p₀). Specifically sets P(p₀ ≤ 0.5) = TAIL "
                 "and P(p₁ ≥ 0.9·p₀) = TAIL. Lower = tighter prior. "
                 "The implied p₁ prior mean is shown below.",
        )
        tail_pi = st.slider(
            "Tail tolerance for π concentration — TAIL_PI",
            0.01, 0.20, 0.05, 0.01,
            help="K_π is auto-calibrated so P(at least one A in class of N) "
                 "≥ 1 - TAIL_PI. Lower = larger K_π = prior has more "
                 "influence over π relative to data.",
        )
    with col2:
        n_chains = st.number_input("MCMC chains", 1, 8, 4, 1)
        iters_per_chain = st.number_input("Iterations per chain (starting)",
                                          1000, 25000, 3000, 500,
                                          help="Starting iterations. If the "
                                               "run fails convergence, iterations "
                                               "are doubled automatically up to "
                                               "the retry cap below.")
        max_retries = st.selectbox(
            "Retry doublings if diagnostics are poor",
            [0, 1, 2, 3],
            index=2,
            help="How many times to double iterations when R̂ or ESS "
                 "fails. 0 = no retry (single run, most transparent). "
                 "2 = up to 3 attempts (e.g., 3000 → 6000 → 12000). Matches "
                 "the behavior of the reference implementation.",
        )
        rhat_target = st.number_input(
            "Max R̂ target", 1.00, 1.20, 1.01, 0.01, format="%.2f",
        )
        ess_target = st.number_input(
            "Min ESS target", 50, 5000, 400, 50,
        )

    # Show derived quantities so the user knows what the tail conditions imply
    try:
        from bag.sampler import calibrate_dirichlet_prior
        _alpha, _mass, _p1_mean, _ = calibrate_dirichlet_prior(
            type1_error=float(type1_error), tail=float(tail_v),
        )
        st.caption(
            f"**Derived from the above:**  "
            f"implied p₁ prior mean = **{_p1_mean:.3f}**  •  "
            f"implied concentration (p01_mass) = **{_mass:.2f}**"
        )
    except Exception as _e:
        st.caption(f"*(Derived prior could not be computed: {_e})*")

# ======================================================================
# Run button
# ======================================================================

st.subheader("3. Compute")

can_run = scores is not None and scores.size >= 5
run_clicked = st.button(
    "Compute my grades",
    type="primary",
    disabled=not can_run,
    use_container_width=False,
)

if not can_run and scores is None:
    st.caption("Enter or upload scores above to enable this button.")

# ======================================================================
# Run sampler
# ======================================================================

def _hash_inputs(scores_arr, pi_vec, Q, fix_pi, type1_error,
                 p1_guess, n_chains, iters_per_chain) -> str:
    import hashlib
    h = hashlib.md5()
    h.update(scores_arr.tobytes())
    h.update(pi_vec.tobytes())
    h.update(str((Q, bool(fix_pi), float(type1_error),
                  float(p1_guess), int(n_chains),
                  int(iters_per_chain))).encode())
    return h.hexdigest()


if run_clicked and can_run:
    inputs_hash = _hash_inputs(
        scores, pi_vec, int(Q), bool(fix_pi), float(type1_error),
        float(tail_v), int(n_chains), int(iters_per_chain),
    )

    cfg = default_config_for_class(
        scores,
        Q=int(Q),
        pi0=pi_vec,
        fix_pi=bool(fix_pi),
        n_chains=int(n_chains),
        iters_per_chain=int(iters_per_chain),
        type1_error=float(type1_error),
        tail=float(tail_v),
        pi_tail=float(tail_pi),
        # p1_mean is left to None so it is derived from (type1_error, tail)
    )
    st.session_state["cfg"] = cfg
    st.session_state["scores_arr"] = scores
    st.session_state["ids"] = student_ids
    st.session_state["last_inputs_hash"] = inputs_hash

    # Total iters estimate for progress bar (will underestimate if retries happen)
    total_iters = cfg.n_chains * cfg.iters_per_chain
    progress_bar = st.progress(0.0, text="Starting MCMC sampler…")
    status_slot = st.empty()

    def _progress_cb(ch, t, n_ch, iters, attempt=1, total_attempts=1):
        prog_frac = (ch * iters + t + 1) / (n_ch * iters)
        burn = int(cfg.burn_in_prop * iters)
        phase = "burning in" if t < burn else "sampling"
        attempt_suffix = (
            f"  |  Attempt {attempt}/{total_attempts}"
            if total_attempts > 1 else ""
        )
        progress_bar.progress(
            min(prog_frac, 1.0),
            text=(
                f"Chain {ch+1}/{n_ch} ({phase}): iter {t+1}/{iters}"
                f"{attempt_suffix}"
            ),
        )

    with st.status("Running Bayesian inference…", expanded=True) as status:
        st.write("**Model:** 5-component truncated-normal mixture on [0, 100]")
        st.write("**Sampler:** Metropolis-within-Gibbs with adaptive Metropolis block")
        st.write(
            f"**Chains × iterations (starting):** {cfg.n_chains} × "
            f"{cfg.iters_per_chain} (burn-in {int(cfg.burn_in_prop*100)}%)"
        )
        if int(max_retries) > 0:
            st.write(
                f"**Auto-retry:** up to {int(max_retries)} doublings if "
                f"R̂ > {float(rhat_target):.2f} or ESS < {int(ess_target)}."
            )
        st.write(f"**Calibrated K_π:** {cfg.K_pi:.2f}")
        st.write("")

        out, cfg_used = run_bag_sampler_with_retry(
            scores,
            cfg=cfg,
            seed=1704,
            rhat_thresh=float(rhat_target),
            ess_min=float(ess_target),
            max_retries=int(max_retries),
            progress_callback=_progress_cb,
        )
        # cfg_used reflects the actual iteration count that ran (post-retry)
        diag = diagnose(out, cfg_used)
        st.session_state["out"] = out
        st.session_state["cfg"] = cfg_used   # store the used config
        st.session_state["diag"] = diag
        progress_bar.progress(1.0, text="Done.")
        attempts = out.get("attempts", 1)
        iters_used = out.get("iters_per_chain_used", cfg_used.iters_per_chain)
        if attempts > 1:
            status.update(
                label=f"✓ Done — converged after {attempts} attempts "
                      f"({iters_used} iters per chain)",
                state="complete", expanded=False,
            )
        else:
            status.update(label="✓ Done", state="complete", expanded=False)


# ======================================================================
# Results
# ======================================================================

out = st.session_state.get("out")
cfg = st.session_state.get("cfg")
scores_stored = st.session_state.get("scores_arr")
ids_stored = st.session_state.get("ids")
diag = st.session_state.get("diag")

if out is not None and cfg is not None and scores_stored is not None:
    st.divider()
    st.subheader("4. Results")

    # --- Confidence banner ---
    status_cls = {"good": "badge-good", "ok": "badge-ok", "poor": "badge-poor",
                  "unknown": "badge-ok"}
    status_text = {
        "good": "Converged cleanly",
        "ok":   "Minor instability",
        "poor": "Poor convergence",
        "unknown": "Diagnostics unavailable",
    }
    cls = status_cls.get(diag["status"], "badge-ok")
    txt = status_text.get(diag["status"], "")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown(
            f'<span class="{cls}">{txt}</span> &nbsp;&nbsp; '
            f'<small>{diag["message"]}</small>',
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Max split-R̂", f"{diag['max_rhat']:.3f}",
                  help="Target: ≤ 1.01 for clean convergence, ≤ 1.05 acceptable.")
    with c3:
        st.metric("Min ESS", f"{diag['min_ess']:.0f}",
                  help="Effective sample size. Target: ≥ 400 for stable estimates.")
    with c4:
        iters_used = out.get("iters_per_chain_used", cfg.iters_per_chain)
        attempts = out.get("attempts", 1)
        st.metric(
            "Iters / chain",
            f"{int(iters_used)}",
            delta=(f"{attempts} attempts" if attempts > 1 else None),
            delta_color="off",
            help="Actual iterations per chain run. If more than the starting "
                 "value, auto-retry doubled iterations until diagnostics passed.",
        )

    # --- Grade table ---
    st.markdown("#### Per-student grades")
    resp_mean = out["resp_mean"]
    exp_grade = out["expected_grade"]
    gran_labels = nearest_granulated_label(exp_grade)
    mode_prob = resp_mean.max(axis=1)

    def _flag(p):
        if p >= 0.80:
            return "🟢"
        elif p >= 0.60:
            return "🟡"
        else:
            return "🔴"

    rows = []
    for i in range(scores_stored.size):
        rows.append({
            "Student": ids_stored[i] if ids_stored else f"Student {i+1}",
            "Score": float(scores_stored[i]),
            "Grade": str(gran_labels[i]),
            "Expected": round(float(exp_grade[i]), 2),
            "Confidence": f"{mode_prob[i]:.2f}",
            "Flag": _flag(mode_prob[i]),
            "P(A)": round(float(resp_mean[i, 0]), 2),
            "P(B)": round(float(resp_mean[i, 1]), 2),
            "P(C)": round(float(resp_mean[i, 2]), 2),
            "P(D)": round(float(resp_mean[i, 3]), 2),
            "P(E)": round(float(resp_mean[i, 4]), 2),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Expected", ascending=False).reset_index(drop=True)

    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Score":      st.column_config.NumberColumn(format="%.2f"),
            "Expected":   st.column_config.NumberColumn(format="%.2f"),
            "P(A)":       st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "P(B)":       st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "P(C)":       st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "P(D)":       st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "P(E)":       st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
        height=min(38 * (len(df) + 1), 560),
    )

    st.caption(
        "🟢 confident (>80%) &nbsp;&nbsp; 🟡 uncertain (60–80%) "
        "&nbsp;&nbsp; 🔴 on the fence (<60%)"
    )

    # --- Boundaries ---
    pi_mean = out["pi_samples"].mean(axis=0)
    mu_mean = out["mu_samples"].mean(axis=0)
    sg_mean = out["sigma_samples"].mean(axis=0)
    bounds = compute_cutoffs(pi_mean, mu_mean, sg_mean,
                             cfg.trunc_a, cfg.trunc_b)
    st.markdown(
        f"**Model-based boundary cutoffs:** "
        f"A/B = **{bounds[0]:.1f}**, &nbsp; "
        f"B/C = **{bounds[1]:.1f}**, &nbsp; "
        f"C/D = **{bounds[2]:.1f}**, &nbsp; "
        f"D/E = **{bounds[3]:.1f}**"
    )
    st.caption(
        "These are the score levels where the posterior probability of two "
        "adjacent grades is equal. They are a consequence of the fitted "
        "model, not a user-supplied cutoff."
    )

    # --- Figures ---
    st.markdown("#### Figures")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Expected grades", "Mixture components",
        "Grade probabilities", "Per-student heatmap",
    ])
    with tab1:
        fig = figure_expected_grades(scores_stored, out, cfg)
        st.pyplot(fig, use_container_width=True)
    with tab2:
        fig = figure_mixture_components(scores_stored, out, cfg)
        st.pyplot(fig, use_container_width=True)
    with tab3:
        fig = figure_probability_curves(scores_stored, out, cfg)
        st.pyplot(fig, use_container_width=True)
    with tab4:
        fig = figure_grade_heatmap(scores_stored, out, cfg, student_ids=ids_stored)
        st.pyplot(fig, use_container_width=True)

    # --- Downloads ---
    st.markdown("#### Downloads")
    d1, d2, d3, d4 = st.columns(4)

    # CSV
    with d1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📄 CSV (grades)",
            data=csv_buf.getvalue(),
            file_name="bag_grades.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # JSON
    with d2:
        payload = {
            "scores": scores_stored.tolist(),
            "student_ids": ids_stored,
            "expected_grade": exp_grade.tolist(),
            "mode_label": out["mode_label"].tolist(),
            "granulated_grade": [str(g) for g in gran_labels],
            "responsibilities": resp_mean.tolist(),
            "posterior_pi_mean": pi_mean.tolist(),
            "posterior_mu_mean": mu_mean.tolist(),
            "posterior_sigma_mean": sg_mean.tolist(),
            "boundaries": bounds.tolist(),
            "diagnostics": {
                "status": diag["status"],
                "max_rhat": diag["max_rhat"],
                "min_ess": diag["min_ess"],
                "accept_rate": diag["accept_rate"],
            },
            "config": {
                "Q": int(cfg.Q),
                "pi0": cfg.pi0.tolist(),
                "K_pi": float(cfg.K_pi) if cfg.K_pi is not None else None,
                "fix_pi": bool(cfg.fix_pi),
                "type1_error": float(cfg.type1_error),
                "p1_mean": float(cfg.p1_mean),
                "n_chains": int(cfg.n_chains),
                "iters_per_chain": int(cfg.iters_per_chain),
                "burn_in_prop": float(cfg.burn_in_prop),
            },
        }
        st.download_button(
            "🗂 JSON (full)",
            data=json.dumps(payload, indent=2),
            file_name="bag_output.json",
            mime="application/json",
            use_container_width=True,
        )

    # PDF
    with d3:
        pdf_bytes = build_pdf_report(scores_stored, out, cfg,
                                     student_ids=ids_stored)
        st.download_button(
            "📑 PDF report",
            data=pdf_bytes,
            file_name="bag_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # PNG bundle as ZIP
    with d4:
        import zipfile
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, fn in [
                ("expected_grades.png", figure_expected_grades),
                ("mixture_components.png", figure_mixture_components),
                ("grade_probabilities.png", figure_probability_curves),
                ("grade_heatmap.png", figure_grade_heatmap),
            ]:
                fig = fn(scores_stored, out, cfg) if "heatmap" not in name \
                    else fn(scores_stored, out, cfg, student_ids=ids_stored)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, dpi=150, bbox_inches="tight")
                zf.writestr(name, img_buf.getvalue())
                import matplotlib.pyplot as plt
                plt.close(fig)
        st.download_button(
            "🖼 PNG figures (zip)",
            data=zip_buf.getvalue(),
            file_name="bag_figures.zip",
            mime="application/zip",
            use_container_width=True,
        )

# ======================================================================
# Footer
# ======================================================================

st.markdown(
    '<div class="footer-note">'
    "Your data stays in your browser session. Nothing is stored on a server "
    "or transmitted outside your own use. Refreshing the page clears "
    "everything. &nbsp;•&nbsp; Based on Schenone (2025), "
    '<em>The Hitchhiker&apos;s Guide to the Grading Galaxy</em>.'
    "</div>",
    unsafe_allow_html=True,
)
