"""The statistical model underlying BAG."""

import streamlit as st

st.set_page_config(page_title="Statistical model — BAG Grader", page_icon="📊", layout="wide")

st.markdown("# The statistical model")
st.caption("The formal specification. For the plain-English version, see the previous page.")

st.markdown(r"""
### Generative model

Let $x_i \in [0, 100]$ be student $i$'s score, and let $z_i \in \{A, B, C, D, E\}$
be student $i$'s latent type. Let $\pi = (\pi_A, \pi_B, \pi_C, \pi_D, \pi_E)$ be
the mixing proportions.

$$
\pi \sim \text{Dirichlet}(K_\pi \cdot \pi_0)
$$

$$
z_i \mid \pi \sim \text{Categorical}(\pi)
$$

$$
x_i \mid z_i = \theta \sim \text{TruncNormal}\big(\mu(\theta),\ \sigma^2(\theta);\ 0, 100\big)
$$

where $\pi_0$ is the prior grade distribution (set by the user) and $K_\pi$
is the prior concentration, auto-calibrated so that the prior-predictive
probability of seeing at least one A in a class of size $N$ is at least
$1 - \text{TAIL\_PI}$ (default TAIL_PI = 0.05, i.e. 95% chance of at least
one A).

### The exam microfoundation (EDSE)

The means and variances of each type's score distribution are **not free
parameters**. They are derived from a microfoundation of the exam structure.

We assume the exam is an **Exponential Decay Separating Exam (EDSE)**: items
are partitioned into four difficulty bands corresponding to types D, C, B, A
(type-E items are absent, since there is no band below E). A type-$\theta$
student answers an item calibrated for type $\hat\theta$ correctly with
probability

$$
p(\theta \mid \hat\theta) = \text{sigmoid}\big(\text{logit}(p_0) + b \cdot d(\theta, \hat\theta)\big),
\qquad d(\theta, \hat\theta) = g(\theta) - g(\hat\theta)
$$

where $g$ is the grading transform ($g(A)=4,\ g(B)=3,\ \ldots$), $p_0$ is the
probability a well-calibrated item is answered correctly (so $1-p_0$ is a
type-1 error), and $b = \text{logit}(p_0) - \text{logit}(p_1)$ is the slope.
$p_1$ is the probability that a type-$\theta$ student correctly answers an
item calibrated for the adjacent-higher type.

Given $(p_0, p_1, Q)$, the moments of each type's score distribution are
deterministic:

$$
\mu(\theta) = 25 \sum_{h = 0}^{3} p(d = h - \theta)
$$

$$
\sigma^2(\theta) = \frac{100^2}{4Q} \sum_{h = 0}^{3} p(d)(1 - p(d))
$$

where $Q$ is the effective number of exam items.

### Prior on $(p_0, p_1)$

We place a Dirichlet prior on the three-component vector $(p_1,\ p_0-p_1,\ 1-p_0)$:

$$
(p_1,\ p_0-p_1,\ 1-p_0) \sim \text{Dirichlet}\big(\alpha_1, \alpha_2, \alpha_3\big)
$$

with $(\alpha_1, \alpha_2, \alpha_3) = \text{mass} \cdot (p_1^\star,\ \text{gap},\ \text{type1\_error})$
and $\text{gap} = 1 - p_1^\star - \text{type1\_error}$. This keeps the
three probabilities summing to one and handles the non-negativity constraints.

The mass is calibrated from two user-friendly constraints:

- $\mathbb{E}[p_0] = 1 - \text{type1\_error}$
- $\Pr(p_1 \ge 0.9 \cdot p_0) = \text{tail}$, ensuring meaningful separation

### Inference via Metropolis-within-Gibbs

The parameters $(p_0, b, \pi)$ are updated in alternation:

1. **Update $(p_0, b)$ given $\pi$.** We work on the unconstrained scale
   $\theta = (\text{logit}(p_0),\ \log b)$ and propose from an adaptive
   multivariate normal. Specifically, a joint Haario-style adaptive Metropolis
   step is used with probability $0.8$, mixed with an independent random-walk
   step with probability $0.2$ for robustness. The proposal covariance adapts
   to the empirical covariance of past draws during burn-in and is frozen
   thereafter. The target acceptance rate is 0.25.

2. **Update $\pi$ given $(p_0, b)$.** Using Rao-Blackwellized soft assignments
   $R_{ik} = \Pr(z_i = k \mid x_i,\ \text{current parameters})$, the
   posterior for $\pi$ is Dirichlet with parameters
   $K_\pi \cdot \pi_0 + \sum_i R_{i\cdot}$. We either sample from it or take
   its mean.

3. **Compute responsibilities.** Given the current parameters, the
   probability that student $i$ is of type $k$ is

   $$r_i(k) = \frac{\pi_k \cdot f(x_i;\ \mu(k), \sigma(k))}{\sum_j \pi_j \cdot f(x_i;\ \mu(j), \sigma(j))}$$

   where $f$ is the truncated-normal density. These are averaged across
   post-burn-in draws to form the posterior mean responsibilities.

### Grade assignment: Bayes actions under different losses

Once we have the posterior $r_i = \Pr(z_i = \cdot \mid x_i)$, the final
letter is a Bayes action. The paper discusses three loss functions:

- **0-1 loss:** assign the modal grade $\arg\max_k r_i(k)$
- **Absolute error loss:** assign the weighted-median grade
- **Quadratic loss:** assign $\sum_k r_i(k) \cdot g(k)$ and round to the
  nearest granulated letter

BAG Grader uses **quadratic loss** by default, because it gives a natural
interpretation of the "+" and "-" modifiers as measures of posterior
uncertainty, rather than as separate types.

### Boundary cutoffs

Model-based cutoffs between adjacent grades are the score values where
$\Pr(k \mid x) = \Pr(k+1 \mid x)$. They are a consequence of the fitted
model, not a user-set quantity. We solve for them via Brent's method.

### Convergence diagnostics

We report:

- **Split-$\hat R$** across chains for $p_0$, $p_1$, $b$, and $\pi$. Target: $\le 1.01$.
- **Bulk effective sample size (ESS)** via the initial-positive-sequence estimator. Target: $\ge 400$.
- **Acceptance rate** of the Metropolis proposal. Target: $\approx 0.25$.

The home page surfaces these through a three-state status badge; the JSON
and PDF downloads contain the raw numbers.

**Auto-retry.** If the first run fails the R̂ or ESS targets, the tool
automatically doubles the iteration count and re-runs, up to a
user-configurable cap (default 2 retries, i.e. up to 3 total attempts:
5000 → 10000 → 20000 iters per chain). The "Iters / chain" metric in the
results panel shows the actual iteration count used, so users always see
a receipt of how much work was done. Users who want a single,
deterministic run can set retries to 0 in the Advanced expander.

### Identification

The model is identified up to label-switching via the constraint
$\mu(A) > \mu(B) > \mu(C) > \mu(D) > \mu(E)$, which is automatically
satisfied by the EDSE parametrization (type means are monotone in $\theta$
whenever $b > 0$, i.e., whenever $p_0 > p_1$). No post-hoc relabeling is
required.

### What's not in this tool

The paper also develops a comparative framework benchmarking BAG against
Fixed Scale Grading and curving under various misspecification regimes.
That comparison is a research artifact, not an instructor-facing feature.
The code for it lives in the GitHub repository as a separate script for
anyone who wants to reproduce the paper's Monte Carlo results.
""")
