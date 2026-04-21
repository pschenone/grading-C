"""
Bayesian Adaptive Grading (BAG) sampler.

Implements the Metropolis-within-Gibbs sampler for the 5-component truncated-normal
mixture model from Schenone (2025), "The Hitchhiker's Guide to the Grading Galaxy".

Model
-----
    π ~ Dirichlet(K_pi * pi0)                           # grade proportions
    z_i ~ Categorical(π)                                 # student type
    x_i | z_i = θ ~ TruncNormal(μ(θ), σ²(θ); 0, 100)    # observed score

Means and variances of each type are deterministic functions of (p0, p1, Q) via
the Exponential Decay Separating Exam (EDSE) parametrization:

    p(d) = sigmoid(logit(p0) + b*d)   with   b = logit(p0) - logit(p1)
    μ(θ)   = 25 * Σ_{h=0..3} p(d = h - θ)
    σ²(θ) = (100² / (4Q)) * Σ_{h=0..3} p(d) * (1 - p(d))

Prior: (p1, p0-p1, 1-p0) ~ Dirichlet(p01_mass * (p1_mean, gap, type1_error))
       where gap = 1 - p1_mean - type1_error.

Only the sampler is here — no experimental FSG / curving comparison code.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.random import default_rng
from scipy.optimize import brentq
from scipy.special import betaln, expit as sigmoid, logsumexp, ndtr
from scipy.stats import beta as beta_dist

GRADE_LABELS = np.array(["A", "B", "C", "D", "E"])
GRADE_VALUES = np.array([4, 3, 2, 1, 0], dtype=float)
GRANULATED_GRID = np.array(
    [4.00, 3.6667, 3.3333, 3.00, 2.6667, 2.3333, 2.00, 1.6667, 1.3333, 1.00, 0.00]
)
GRANULATED_LABELS = np.array(
    ["A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "E"]
)


# ======================================================================
# Prior calibration helpers (from Cell 1 of Final.py)
# ======================================================================

def _beta_mean_left_tail_to_ab(
    mean: float, x: float = 0.5, p_left: float = 0.01,
    k_lo: float = 2.0, k_hi: float = 200000.0,
) -> Tuple[float, float]:
    """Find Beta(a,b) with given mean and left-tail probability P(X <= x) = p_left."""
    if not (0.0 < mean < 1.0):
        raise ValueError("mean must be in (0,1)")
    if mean <= x:
        raise ValueError(f"Infeasible: mean={mean} <= x={x}")

    def f(k):
        a = mean * k
        b = (1.0 - mean) * k
        return beta_dist.cdf(x, a, b) - p_left

    # Ensure we bracket a root. If not, expand / return closest endpoint.
    f_lo = f(k_lo)
    if f_lo < 0:
        k_lo = 1.01
        f_lo = f(k_lo)
    f_hi = f(k_hi)
    if f_hi > 0:
        k = k_hi
        while f(k) > 0 and k < 1e8:
            k *= 2.0
        k_hi = k
        f_hi = f(k_hi)

    # If we still don't bracket (e.g., p_left is too large to be achievable),
    # fall back to the endpoint that minimizes |f|.
    if f_lo * f_hi > 0:
        k_star = k_lo if abs(f_lo) < abs(f_hi) else k_hi
    else:
        k_star = brentq(f, k_lo, k_hi)
    return mean * k_star, (1.0 - mean) * k_star


def _beta_right_tail_split(
    total: float, threshold: float = 0.9, p_upper: float = 0.01,
) -> Tuple[float, float]:
    """Split total = a1+a2 so that Beta(a1,a2) satisfies P(V >= threshold) ≈ p_upper."""
    def tail_prob(m: float) -> float:
        if m <= 0.0 or m >= 1.0:
            return 0.0
        a1 = m * total
        a2 = (1.0 - m) * total
        return beta_dist.sf(threshold, a1, a2)

    eps = 1e-3
    m_lo, m_hi = eps, 1.0 - eps
    p_lo, p_hi = tail_prob(m_lo), tail_prob(m_hi)
    if (p_lo - p_upper) * (p_hi - p_upper) > 0.0:
        m_star = m_lo if abs(p_lo - p_upper) < abs(p_hi - p_upper) else m_hi
    else:
        m_star = brentq(lambda m: tail_prob(m) - p_upper, m_lo, m_hi)
    return m_star * total, (1.0 - m_star) * total


def calibrate_dirichlet_prior(
    type1_error: float = 0.10, tail: float = 0.01,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Calibrate the Dirichlet prior on (p1, p0-p1, 1-p0) from two user-friendly knobs:
      - type1_error: E[1 - p0] (i.e. probability a well-calibrated question is missed)
      - tail: tail tolerances for the constraints

    Returns (alpha_dir, mass, p1_mean, type1_mean).
    """
    # (1) p0 = w1+w2 ~ Beta(a_p0, a3) with mean 1-type1_error and P(p0 <= 0.5) = tail
    a_p0, a3 = _beta_mean_left_tail_to_ab(
        mean=1 - type1_error, x=0.5, p_left=tail
    )
    # (2) Split a_p0 = a1 + a2 so that V = p1/p0 ~ Beta(a1,a2) has P(V >= 0.9) ≈ tail
    a1, a2 = _beta_right_tail_split(total=a_p0, threshold=0.9, p_upper=tail)
    alpha = np.array([a1, a2, a3], float)
    mass = float(alpha.sum())
    p1_mean = float(alpha[0] / mass)
    type1_mean = float(alpha[2] / mass)
    return alpha, mass, p1_mean, type1_mean


def _prob_at_least_one_A(K_pi: float, pi0_A: float, N: int) -> float:
    """Prior-predictive P(n_A >= 1) under Dirichlet-multinomial."""
    a = K_pi * pi0_A
    b = K_pi * (1.0 - pi0_A)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    log_ratio = betaln(a, b + N) - betaln(a, b)
    return 1.0 - float(np.exp(log_ratio))


def calibrate_pi_mass(
    pi0_A: float, N: int, target: float = 0.95,
    K_lo: float = 2.0, K_hi: float = 20.0, K_cap: float = 100.0,
) -> float:
    """Find K_pi so that P(at least one A in class of N) >= target."""
    if pi0_A <= 0.0 or N <= 0:
        return K_cap
    p_max = 1.0 - (1.0 - pi0_A) ** N
    if target > p_max + 1e-12:
        return K_cap
    while _prob_at_least_one_A(K_hi, pi0_A, N) < target and K_hi < K_cap:
        K_hi *= 2.0
    if _prob_at_least_one_A(K_hi, pi0_A, N) < target:
        return K_cap
    return float(
        brentq(lambda K: _prob_at_least_one_A(K, pi0_A, N) - target, K_lo, K_hi)
    )


# ======================================================================
# Config
# ======================================================================

@dataclass
class GraderConfig:
    """Configuration for the BAG sampler.

    Most users only touch `Q` and `pi0`. Everything else has a reasonable default.

    The hidden MCMC defaults below intentionally mirror the tuned reference-app
    settings from the original `Final.py`, rather than generic package-neutral
    values. That makes the web app track the original BAG engine more closely in
    finite samples.
    """

    # --- core ---
    Q: int = 60
    pi0: np.ndarray = field(default_factory=lambda: np.array(
        [0.165, 0.378, 0.373, 0.074, 0.010], dtype=float))

    # --- MCMC settings ---
    n_chains: int = 4
    iters_per_chain: int = 10000
    burn_in_prop: float = 0.25
    thinning: int = 1

    # --- π prior ---
    K_pi: Optional[float] = None     # None -> auto-calibrate from class size
    fix_pi: bool = False              # If True, don't update π from data

    # --- tempered π update ---
    pi_update_period: int = 1
    pi_update_warmup: Optional[int] = None  # None -> burn_in_prop * iters
    pi_temper: float = 1.0
    pi_update_mode: str = "sample"
    pi_ema: float = 1.0

    # --- Dirichlet prior on (p1, p0-p1, 1-p0) ---
    type1_error: float = 0.10
    p01_mass: Optional[float] = None  # None -> auto-calibrate
    p1_mean: Optional[float] = None   # None -> auto-calibrate
    tail: float = 0.01

    # --- proposal distribution ---
    prop_sd_u0: float = 0.07
    prop_sd_logb: float = 0.07

    # --- numerics ---
    sigma_min: float = 0.20
    trunc_a: float = 0.0
    trunc_b: float = 100.0

    # --- Adaptive Metropolis ---
    use_block_am: bool = True
    am_t0: int = 30
    am_eps: float = 1e-8
    am_scale: float = 0.3
    am_diag_scale: float = 0.3
    am_target_acc: float = 0.30
    am_adapt_window: int = 30
    am_adapt_gain: float = 0.40
    am_min_scale: float = 0.01
    am_max_scale: float = 2.00
    am_diag_min_scale: float = 0.20
    am_diag_max_scale: float = 2.00
    mix_prob_joint: float = 1.0

    # --- populated in __post_init__ ---
    _alpha_dir: np.ndarray = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Normalize π prior
        self.pi0 = np.asarray(self.pi0, dtype=float)
        if self.pi0.shape != (5,):
            raise ValueError("pi0 must have exactly 5 components")
        s = self.pi0.sum()
        if s <= 0:
            raise ValueError("pi0 must have positive mass")
        self.pi0 = self.pi0 / s

        # Auto-calibrate Dirichlet hyperparameters if not provided
        if self.p01_mass is None or self.p1_mean is None:
            alpha_dir, mass, p1_mean, _ = calibrate_dirichlet_prior(
                type1_error=self.type1_error, tail=self.tail
            )
            if self.p01_mass is None:
                self.p01_mass = mass
            if self.p1_mean is None:
                self.p1_mean = p1_mean

        gap = 1.0 - self.type1_error - self.p1_mean
        if gap <= 0.0:
            raise ValueError("Infeasible: p1_mean + type1_error must be < 1")
        self._alpha_dir = self.p01_mass * np.array(
            [self.p1_mean, gap, self.type1_error], dtype=float
        )

        # Tempered π knobs
        self.pi_update_period = max(1, int(self.pi_update_period))
        if self.pi_update_warmup is None:
            self.pi_update_warmup = int(self.burn_in_prop * self.iters_per_chain)
        self.pi_update_warmup = max(0, int(self.pi_update_warmup))
        self.pi_temper = float(np.clip(self.pi_temper, 0.0, 1.0))
        if self.pi_update_mode not in {"mean", "sample"}:
            self.pi_update_mode = "sample"
        self.pi_ema = float(np.clip(self.pi_ema, 0.0, 1.0))


def default_config_for_class(
    scores: np.ndarray,
    Q: int = 60,
    pi0: Optional[np.ndarray] = None,
    pi_tail: float = 0.05,
    **kwargs,
) -> GraderConfig:
    """Build a GraderConfig with π-mass auto-calibrated for the class size.

    pi_tail: tolerance such that P(at least one A in class of N) >= 1 - pi_tail.
    """
    cfg = GraderConfig(
        Q=Q,
        pi0=pi0 if pi0 is not None else np.array(
            [0.165, 0.378, 0.373, 0.074, 0.010], float),
        **kwargs,
    )
    if cfg.K_pi is None:
        N = int(np.asarray(scores).size)
        cfg.K_pi = calibrate_pi_mass(
            pi0_A=float(cfg.pi0[0]), N=N, target=1.0 - pi_tail,
        )
    return cfg


# ======================================================================
# Likelihood utilities
# ======================================================================

def _clamp_prob(p, eps=1e-12):
    return float(np.clip(p, eps, 1.0 - eps))


def _safe_logit(p, eps=1e-12):
    p = _clamp_prob(p, eps)
    return float(np.log(p) - np.log1p(-p))


def type_moments(
    p0: float, p1: float, Q: int, sigma_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (μ, σ) vectors over the 5 types given (p0, p1, Q)."""
    u0 = _safe_logit(p0)
    u1 = _safe_logit(p1)
    b = max(1e-12, u0 - u1)

    H = np.arange(4, dtype=float)[None, :]
    TH = np.arange(5, dtype=float)[:, None]
    D = H - TH
    P = sigmoid(u0 + b * D)

    mus = 25.0 * P.sum(axis=1)
    sigma2 = (100.0 ** 2) * (P * (1.0 - P)).sum(axis=1) / (4.0 * Q)
    sigmas = np.sqrt(np.maximum(0.0, sigma2))
    if sigma_min > 0.0:
        sigmas = np.maximum(sigmas, float(sigma_min))
    return mus.astype(float), sigmas.astype(float)


def _compute_logF(
    scores: np.ndarray, mu: np.ndarray, sigma: np.ndarray, cfg: GraderConfig,
    x_cached: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Truncated normal log-pdf matrix logF (N, K) on [trunc_a, trunc_b]."""
    x = x_cached if x_cached is not None else scores[:, None]
    mu = np.asarray(mu, dtype=float)[None, :]
    sd = np.asarray(sigma, dtype=float)[None, :]
    z_lo = (cfg.trunc_a - mu) / sd
    z_hi = (cfg.trunc_b - mu) / sd
    denom = np.maximum(ndtr(z_hi) - ndtr(z_lo), 1e-300)
    z = (x - mu) / sd
    log_phi = -0.5 * z ** 2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)
    return log_phi - np.log(denom)


def _responsibilities_from_logF(
    logF: np.ndarray, pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    log_pi = np.log(np.asarray(pi, float) + 1e-300)[None, :]
    logw = logF + log_pi
    row_lse = logsumexp(logw, axis=1, keepdims=True)
    R = np.exp(logw - row_lse)
    return R, logw


def _dir_prior_logprob(u0: float, logb: float, cfg: GraderConfig) -> float:
    """Log prior p(u0, logb) under Dirichlet prior on (p1, p0-p1, 1-p0) with Jacobian."""
    p0 = _clamp_prob(sigmoid(u0))
    b = float(np.exp(logb))
    p1 = _clamp_prob(sigmoid(u0 - b))
    w1 = _clamp_prob(p1)
    w2 = max(1e-300, p0 - p1)
    w3 = _clamp_prob(1.0 - p0)
    a1, a2, a3 = cfg._alpha_dir
    lp_dir = (a1 - 1.0) * np.log(w1) + (a2 - 1.0) * np.log(w2) + (a3 - 1.0) * np.log(w3)
    log_det = np.log(b) + np.log(p1) + np.log(1.0 - p1) + np.log(p0) + np.log(1.0 - p0)
    return float(lp_dir + log_det)


def _logpost_theta(
    theta, scores, pi_current, cfg, x_cached: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Full log posterior at theta=(u0, logb). Returns (logp, mu, sigma, logF)."""
    u0, logb = map(float, theta)
    p0 = _clamp_prob(sigmoid(u0))
    b = float(np.exp(logb))
    p1 = _clamp_prob(sigmoid(u0 - b))
    mu, sigma = type_moments(p0, p1, cfg.Q, sigma_min=cfg.sigma_min)
    logF = _compute_logF(scores, mu, sigma, cfg, x_cached=x_cached)
    log_pi = np.log(np.asarray(pi_current, float) + 1e-300)[None, :]
    ll = float(np.sum(logsumexp(logF + log_pi, axis=1)))
    lp = _dir_prior_logprob(u0, logb, cfg)
    return ll + lp, mu, sigma, logF


# ======================================================================
# Sampler state and MH steps
# ======================================================================

@dataclass
class _SamplerState:
    pi: np.ndarray
    p0: float
    p1: float
    b: float


def _initialize_state(
    scores: np.ndarray, cfg: GraderConfig, rng: np.random.Generator,
) -> _SamplerState:
    """Draw initial state from the Dirichlet prior."""
    pi = np.asarray(cfg.pi0, float).copy()
    w = rng.dirichlet(cfg._alpha_dir)
    p1 = float(w[0])
    p0 = float(w[0] + w[1])
    b = max(1e-12, _safe_logit(p0) - _safe_logit(p1))
    return _SamplerState(pi=pi, p0=p0, p1=p1, b=b)


def _pack_theta(state: _SamplerState) -> np.ndarray:
    return np.array([_safe_logit(state.p0), np.log(max(1e-12, state.b))], float)


def _unpack_theta(theta: np.ndarray, state: _SamplerState) -> None:
    u0, logb = map(float, theta)
    state.p0 = _clamp_prob(sigmoid(u0))
    state.b = float(np.exp(logb))
    state.p1 = _clamp_prob(sigmoid(u0 - state.b))


def _mh_step(
    state: _SamplerState, scores: np.ndarray, cfg: GraderConfig,
    rng: np.random.Generator, chol: np.ndarray,
    logp_cur, mu_cur, sigma_cur, logF_cur, x_cached,
):
    theta = _pack_theta(state)
    if logp_cur is None:
        logp_cur, mu_cur, sigma_cur, logF_cur = _logpost_theta(
            theta, scores, state.pi, cfg, x_cached=x_cached
        )
    z = rng.normal(size=2)
    theta_prop = theta + chol @ z
    logp_prop, mu_prop, sigma_prop, logF_prop = _logpost_theta(
        theta_prop, scores, state.pi, cfg, x_cached=x_cached
    )
    if np.log(rng.uniform()) < (logp_prop - logp_cur):
        _unpack_theta(theta_prop, state)
        return True, logp_prop, mu_prop, sigma_prop, logF_prop
    return False, logp_cur, mu_cur, sigma_cur, logF_cur


# ======================================================================
# Main sampler
# ======================================================================

def run_bag_sampler(
    scores: np.ndarray,
    cfg: Optional[GraderConfig] = None,
    seed: Optional[int] = 1704,
    progress_callback=None,
) -> Dict[str, np.ndarray]:
    """Run the Metropolis-within-Gibbs BAG sampler.

    Parameters
    ----------
    scores : array-like of length N
        Numerical scores on [0, 100].
    cfg : GraderConfig, optional
        Sampler configuration. If None, uses defaults + auto-calibration.
    seed : int, optional
        Master seed for reproducibility.
    progress_callback : callable, optional
        Called with (chain_idx, iter_idx, n_chains, iters_per_chain) during sampling.
        Use this to drive a Streamlit progress bar.

    Returns
    -------
    dict with keys:
        p0_samples, p1_samples, b_samples        : (T,) traces
        pi_samples, mu_samples, sigma_samples    : (T, 5) traces
        resp_mean                                 : (N, 5) posterior responsibilities
        expected_grade                            : (N,) E[grade value]
        mode_label                                : (N,) argmax responsibility index
        accept_rate                               : float
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    if np.any(scores < 0) or np.any(scores > 100):
        raise ValueError("scores must all lie in [0, 100]")

    if cfg is None:
        cfg = default_config_for_class(scores)

    rng_master = default_rng(seed)
    x_cached = scores[:, None]
    N = scores.shape[0]

    iters = cfg.iters_per_chain
    burn = int(cfg.burn_in_prop * iters)
    kept_per_chain = (iters - burn + cfg.thinning - 1) // cfg.thinning
    total_kept = cfg.n_chains * kept_per_chain

    kept_p0 = np.empty(total_kept, dtype=float)
    kept_p1 = np.empty(total_kept, dtype=float)
    kept_b = np.empty(total_kept, dtype=float)
    kept_pi = np.empty((total_kept, 5), dtype=float)
    kept_mu = np.empty((total_kept, 5), dtype=float)
    kept_sigma = np.empty((total_kept, 5), dtype=float)

    resp_accum = np.zeros((N, 5))
    kept_draws = 0
    keep_idx = 0

    accept_count = 0
    prop_count = 0

    d = 2
    base_s_d = (2.38 ** 2) / d
    s_d = base_s_d * cfg.am_scale
    base_diag0 = np.diag([cfg.prop_sd_u0, cfg.prop_sd_logb])
    diag0 = base_diag0 * cfg.am_diag_scale
    epsI = cfg.am_eps * np.eye(d)

    for ch in range(cfg.n_chains):
        rng = default_rng(rng_master.integers(1, 2 ** 63 - 1))
        state = _initialize_state(scores, cfg, rng)

        theta_cur = _pack_theta(state)
        m = theta_cur.copy()
        S = np.zeros((d, d), float)
        n = 1
        chol = diag0.copy()
        chol_frozen = None
        logp_cur = None
        mu_cur = sigma_cur = logF_cur = None

        # adaptive scalers (local copies)
        am_scale = cfg.am_scale
        am_diag_scale = cfg.am_diag_scale
        log_am_scale = np.log(max(1e-12, am_scale))
        log_am_diag_scale = np.log(max(1e-12, am_diag_scale))
        win_acc = 0
        win_props = 0

        for t in range(iters):
            # choose proposal covariance
            if cfg.use_block_am:
                if t < burn:
                    if (t + 1) >= max(1, cfg.am_t0) and n > 1:
                        Cov = (S / max(1, n - 1)) + epsI
                        try:
                            chol = np.linalg.cholesky(s_d * Cov)
                        except np.linalg.LinAlgError:
                            chol = np.linalg.cholesky(s_d * (Cov + 10.0 * epsI))
                    else:
                        chol = diag0
                else:
                    if chol_frozen is None:
                        if n > 1:
                            Cov = (S / max(1, n - 1)) + epsI
                            try:
                                chol_frozen = np.linalg.cholesky(s_d * Cov)
                            except np.linalg.LinAlgError:
                                chol_frozen = np.linalg.cholesky(s_d * (Cov + 10.0 * epsI))
                        else:
                            chol_frozen = diag0
                    chol = chol_frozen
            else:
                chol = diag0

            use_joint = cfg.use_block_am and (rng.random() < cfg.mix_prob_joint)
            proposal = chol if use_joint else diag0
            accepted, logp_cur, mu_cur, sigma_cur, logF_cur = _mh_step(
                state, scores, cfg, rng, proposal,
                logp_cur, mu_cur, sigma_cur, logF_cur, x_cached,
            )
            prop_count += 1
            accept_count += int(accepted)

            # reuse cached quantities for π update
            mu, sigma = mu_cur, sigma_cur
            R_old, _ = _responsibilities_from_logF(logF_cur, state.pi)

            # tempered π update
            if not cfg.fix_pi:
                do_update = (
                    (t + 1) % cfg.pi_update_period == 0
                    and (t + 1) > cfg.pi_update_warmup
                )
                if do_update:
                    n_soft = R_old.sum(axis=0)
                    n_tempered = cfg.pi_temper * n_soft
                    alpha = cfg.K_pi * np.asarray(cfg.pi0, float) + n_tempered
                    if cfg.pi_update_mode == "sample":
                        pi_new = rng.dirichlet(alpha)
                    else:
                        pi_new = alpha / alpha.sum()
                    lam = cfg.pi_ema
                    state.pi = (1.0 - lam) * state.pi + lam * pi_new
                    state.pi = state.pi / state.pi.sum()
                    logp_cur = None  # π changed

            # AM adaptation
            if t < burn and cfg.use_block_am:
                theta_cur = _pack_theta(state)
                n += 1
                delta = theta_cur - m
                m += delta / n
                S += np.outer(delta, theta_cur - m)
                if use_joint:
                    win_props += 1
                    win_acc += int(accepted)
                    if win_props >= cfg.am_adapt_window:
                        acc_rate = win_acc / max(1, win_props)
                        gamma = cfg.am_adapt_gain / np.sqrt(t + 1.0)
                        if (t + 1) < max(1, cfg.am_t0):
                            log_am_diag_scale += gamma * (acc_rate - cfg.am_target_acc)
                            am_diag_scale = float(np.clip(
                                np.exp(log_am_diag_scale),
                                cfg.am_diag_min_scale, cfg.am_diag_max_scale
                            ))
                            diag0 = base_diag0 * am_diag_scale
                        else:
                            log_am_scale += gamma * (acc_rate - cfg.am_target_acc)
                            am_scale = float(np.clip(
                                np.exp(log_am_scale),
                                cfg.am_min_scale, cfg.am_max_scale
                            ))
                            s_d = base_s_d * am_scale
                        win_props = 0
                        win_acc = 0

            # store
            if t >= burn and ((t - burn) % cfg.thinning == 0):
                kept_p0[keep_idx] = state.p0
                kept_p1[keep_idx] = state.p1
                kept_b[keep_idx] = state.b
                kept_pi[keep_idx, :] = state.pi
                kept_mu[keep_idx, :] = mu
                kept_sigma[keep_idx, :] = sigma
                resp_accum += R_old
                kept_draws += 1
                keep_idx += 1

            if progress_callback is not None and (t % 100 == 0 or t == iters - 1):
                progress_callback(ch, t, cfg.n_chains, iters)

    resp_mean = resp_accum / max(1, kept_draws)
    expected_grade = resp_mean @ GRADE_VALUES
    mode_label = np.argmax(resp_mean, axis=1)

    return {
        "p0_samples": kept_p0,
        "p1_samples": kept_p1,
        "b_samples": kept_b,
        "pi_samples": kept_pi,
        "mu_samples": kept_mu,
        "sigma_samples": kept_sigma,
        "resp_mean": resp_mean,
        "expected_grade": expected_grade,
        "mode_label": mode_label,
        "accept_rate": accept_count / max(1, prop_count),
    }


def run_bag_sampler_with_retry(
    scores: np.ndarray,
    cfg: Optional["GraderConfig"] = None,
    seed: int = 1704,
    rhat_thresh: float = 1.01,
    ess_min: float = 400.0,
    max_retries: int = 2,
    progress_callback=None,
) -> Dict:
    """Run the sampler. If convergence thresholds are not met, double the
    iteration count and try again, up to `max_retries` additional attempts.

    This mirrors the auto-escalation logic from the reference implementation:
    the user clicks Run once, and the sampler keeps doubling iters_per_chain
    until either (max_rhat < rhat_thresh and min_ess >= ess_min) or we hit
    the retry cap.

    Returns the sampler output dict with two extra keys:
      - 'iters_per_chain_used': final iters_per_chain actually run
      - 'attempts': number of attempts made (1 = first try succeeded)
      - 'converged': True if the final run met both thresholds

    When convergence fails after all retries, the last attempt is returned.
    """
    from dataclasses import replace

    if cfg is None:
        cfg = default_config_for_class(scores)

    base_iters = int(cfg.iters_per_chain)
    last_out = None
    last_cfg = cfg

    for attempt in range(1, max_retries + 2):
        iters_for_this = base_iters * (2 ** (attempt - 1))
        cfg_attempt = replace(cfg, iters_per_chain=int(iters_for_this))
        # ensure pi_update_warmup is rescaled to the new iter count
        cfg_attempt.pi_update_warmup = int(
            cfg_attempt.burn_in_prop * cfg_attempt.iters_per_chain
        )

        if progress_callback is not None:
            # Wrap the callback so it reports the attempt number too
            def _inner_cb(ch, t, n_ch, iters, _attempt=attempt,
                          _total_attempts=max_retries + 1):
                progress_callback(ch, t, n_ch, iters, _attempt, _total_attempts)
            cb = _inner_cb
        else:
            cb = None

        out = run_bag_sampler(
            scores, cfg=cfg_attempt, seed=seed + 1000 * (attempt - 1),
            progress_callback=cb,
        )
        diag = diagnose(out, cfg_attempt)

        out["iters_per_chain_used"] = int(cfg_attempt.iters_per_chain)
        out["attempts"] = attempt
        out["converged"] = bool(
            diag["max_rhat"] < rhat_thresh and diag["min_ess"] >= ess_min
        )
        last_out = out
        last_cfg = cfg_attempt

        if out["converged"]:
            return out, last_cfg

    return last_out, last_cfg


# ======================================================================
# Post-processing: labels, boundaries, diagnostics
# ======================================================================

def nearest_granulated_label(gval) -> np.ndarray:
    """Map a scalar or array of expected-grade values to granulated labels."""
    gval = np.atleast_1d(np.asarray(gval, dtype=float))
    idx = np.argmin(np.abs(gval - GRANULATED_GRID[:, None]), axis=0)
    return GRANULATED_LABELS[idx]


def compute_cutoffs(
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    a: float = 0.0, b: float = 100.0,
) -> np.ndarray:
    """Score cutoffs where P(grade k | x) == P(grade k+1 | x) for k=0..3."""
    from scipy.stats import truncnorm as tn
    eps = 1e-6
    boundaries = []
    for k in range(4):
        mu0, sd0, pi0 = float(mu[k]), float(max(sigma[k], eps)), float(pi[k])
        mu1, sd1, pi1 = float(mu[k + 1]), float(max(sigma[k + 1], eps)), float(pi[k + 1])
        a0, b0 = (a - mu0) / sd0, (b - mu0) / sd0
        a1, b1 = (a - mu1) / sd1, (b - mu1) / sd1

        def diff(x):
            return pi0 * tn.pdf(x, a0, b0, loc=mu0, scale=sd0) - \
                   pi1 * tn.pdf(x, a1, b1, loc=mu1, scale=sd1)

        try:
            f_a, f_b = diff(a), diff(b)
            if f_a * f_b > 0:
                grid = np.linspace(a, b, 51)
                vals = diff(grid)
                sgn = np.sign(vals)
                idx = np.where(np.diff(sgn) != 0)[0]
                if len(idx) > 0:
                    lo, hi = grid[idx[0]], grid[idx[0] + 1]
                else:
                    raise ValueError
            else:
                lo, hi = a, b
            boundaries.append(brentq(diff, lo, hi, maxiter=200))
        except Exception:
            boundaries.append(0.5 * (mu0 + mu1))
    return np.array(boundaries)


def _split_rhat(chains: np.ndarray) -> float:
    M, T = chains.shape
    if T % 2 == 1:
        chains = chains[:, :-1]
        T -= 1
    halves = chains.reshape(M * 2, T // 2)
    _, n = halves.shape
    B = n * halves.mean(axis=1).var(ddof=1)
    W = halves.var(axis=1, ddof=1).mean()
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else float("nan")


def _ess_bulk(chains: np.ndarray, max_lag: int = 1000) -> float:
    M, T = chains.shape
    x = chains - chains.mean(axis=1, keepdims=True)
    gamma = []
    for lag in range(max_lag + 1):
        v = 0.0
        for m in range(M):
            xm = x[m]
            if lag == 0:
                v += np.dot(xm, xm) / T
            else:
                v += np.dot(xm[:-lag], xm[lag:]) / (T - lag)
        gamma.append(v / M)
    gamma = np.array(gamma)
    if gamma[0] <= 0:
        return float("nan")
    rho = gamma / gamma[0]
    t = 1
    s = 0.0
    while 2 * t < len(rho) and (rho[2 * t - 1] + rho[2 * t]) > 0:
        s += rho[2 * t - 1] + rho[2 * t]
        t += 1
    return float(max(1.0, M * T / (1 + 2 * s)))


def diagnose(out: Dict[str, np.ndarray], cfg: GraderConfig) -> Dict:
    """Compute convergence diagnostics (split-Rhat, ESS, acceptance) and
    return a dict suitable for UI rendering."""
    n_ch = int(cfg.n_chains)
    p0, p1, b = out["p0_samples"], out["p1_samples"], out["b_samples"]
    pi = out["pi_samples"]

    def _rhat_ess(trace: np.ndarray) -> Tuple[float, float]:
        T_total = trace.shape[0]
        if T_total % n_ch != 0:
            return float("nan"), float("nan")
        T = T_total // n_ch
        if trace.ndim == 1:
            ch = trace.reshape(n_ch, T)
            return _split_rhat(ch), _ess_bulk(ch)
        rhats, esss = [], []
        for k in range(trace.shape[1]):
            ch = trace[:, k].reshape(n_ch, T)
            rhats.append(_split_rhat(ch))
            esss.append(_ess_bulk(ch))
        return float(np.nanmax(rhats)), float(np.nanmin(esss))

    rhat_p0, ess_p0 = _rhat_ess(p0)
    rhat_p1, ess_p1 = _rhat_ess(p1)
    rhat_b, ess_b = _rhat_ess(b)
    rhat_pi, ess_pi = _rhat_ess(pi)

    max_rhat = float(np.nanmax([rhat_p0, rhat_p1, rhat_b, rhat_pi]))
    min_ess = float(np.nanmin([ess_p0, ess_p1, ess_b, ess_pi]))

    if np.isnan(max_rhat):
        status = "unknown"
        message = "Could not compute convergence diagnostics."
    elif max_rhat <= 1.01 and min_ess >= 400:
        status = "good"
        message = "The statistical engine converged cleanly. Results are trustworthy."
    elif max_rhat <= 1.05 and min_ess >= 100:
        status = "ok"
        message = ("Some minor instability in the posterior fit. "
                   "Results are usable but you may want to increase iterations.")
    else:
        status = "poor"
        message = ("The sampler did not converge well. "
                   "Increase iterations or number of chains and re-run.")

    return {
        "status": status,
        "message": message,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "accept_rate": float(out["accept_rate"]),
        "rhat_detail": {"p0": rhat_p0, "p1": rhat_p1, "b": rhat_b, "pi": rhat_pi},
        "ess_detail": {"p0": ess_p0, "p1": ess_p1, "b": ess_b, "pi": ess_pi},
    }
