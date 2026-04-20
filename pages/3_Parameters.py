"""Parameter reference — explains every knob in the tool."""

import streamlit as st

st.set_page_config(page_title="Parameters — BAG Grader", page_icon="📊", layout="wide")

st.markdown("# Parameter reference")
st.caption(
    "Every knob the tool exposes, what it does, when to change it, and "
    "what happens if you get it wrong."
)

st.markdown("## Basic tier — the one dropdown")

st.markdown("""
### Expected grade distribution (π)

Your prior belief about the proportion of students in each grade category,
before seeing any scores. The data will then update this prior.

The tool provides six presets:

- **Baseline business-school prior (default)** — `[0.165, 0.378, 0.373, 0.074, 0.010]`.
  The default preset, matching the baseline prior used in the paper's example.
  A mildly bell-shaped distribution with few A's and very few E's. Reasonable
  for a well-behaved mid-level class.
- **Alternative A/B-heavy prior** — `[0.27, 0.65, 0.056, 0.016, 0.008]`. Heavy
  top of the distribution. Useful for selective or upper-division classes where
  most students are expected to earn an A or B.
- **Alternative symmetric bell prior** — `[0.0668, 0.24173, 0.38292, 0.24173, 0.0667]`.
  A textbook symmetric bell-shaped distribution centered on C.
- **Let the data decide (near-uniform prior)** — `[0.2, 0.2, 0.2, 0.2, 0.2]`.
  The most agnostic option. With this preset, the data dominates the posterior
  more strongly, so the final grade distribution is genuinely data-driven.
- **Skewed high (honors class)** — `[0.30, 0.40, 0.22, 0.06, 0.02]`. Similar
  in spirit to the A/B-heavy prior, milder tail.
- **Skewed low (intro / weed-out)** — `[0.06, 0.18, 0.38, 0.28, 0.10]`. For
  first-year intro classes or courses designed to separate out students who
  shouldn't continue in the major.

**What if I get the prior wrong?** The data will pull the posterior away from
a bad prior, but only so far. How far depends on the concentration mass K_π
(see Advanced section below). With default settings, the data dominates, so
moderate prior errors self-correct. If your prior is very far off and your
class is small, you'll see it in the posterior. When in doubt, pick "Let the
data decide."
""")

st.markdown("## Customize tier — exam structure and quotas")

st.markdown("""
### Effective number of exam items (Q)

The number of scorable "units" your exam represents. This directly controls
the **variance** of each component in the mixture: smaller Q means wider
component distributions, so BAG is less willing to draw sharp lines between
adjacent types.

**How to compute it:**
- A 60-question multiple-choice exam: Q = 60.
- A 12-question exam graded in fifth-point increments: Q = 12 × 5 = 60
  (this is the default wizard setting).
- A 10-question exam graded in half-point increments: Q = 10 × 2 = 20.
- A single essay graded out of 100 in 5-point increments: Q = 20.
- Two exams averaged together, each Q = 30: Q = 60 (see paper §2.3).

The built-in wizard in the Customize expander computes Q from "number of
questions" and "grading increments" — it defaults to 12 questions in fifths,
which gives Q = 60 (the default Q).

**What if I get Q wrong?** If you set Q too high (pretending the exam is
more precise than it is), the model will be overconfident in drawing sharp
boundaries. If you set Q too low, the model will be too timid and you'll
get large uncertainty bands. The safe direction to err is low.

### Custom π sliders

Five sliders for the grade proportions. They auto-normalize so you don't
need to make them sum to exactly 1. The sliders reflect whichever preset
is currently selected; moving them overrides the preset.

### Enforce this distribution exactly (fix_pi)

If checked, the tool does **not** update π from the data. It locks π to
the distribution you set. Use this only when:

- Your institution mandates a strict grade distribution (quotas), or
- You have very strong prior information about the class composition.

**The trade-off:** you lose BAG's adaptive advantage. If the data strongly
contradicts your fixed π, the model still has to honor it, which can
produce poor fits and misclassification. Most instructors should leave
this unchecked.
""")

st.markdown("## Advanced tier — you probably don't need this")

st.markdown("""
These exist for reproducibility of published results and for users who
want to poke at the MCMC internals. If you don't know what MCMC is,
leave them alone.

### Prior calibration knobs

These three knobs *together* determine the full prior on the exam
parameters `(p₀, p₁)`. You do not set `p₁` directly — it's derived from
these three, and the Advanced expander shows you the derived `p₁` prior
mean live as you move the sliders.

#### Type-1 error

The probability that a student of type θ misses a question calibrated for
their own type. Equivalently, `1 - p₀`. **Default: 0.10.**

This captures how well-calibrated your exam items are. A value of 0.10
means even students who *should* get a given item correct will miss it
10% of the time due to noise. Lower values assume sharper, more
discriminating items.

#### TAIL — tail tolerance for the `(p₀, p₁)` prior

Controls concentration of the Dirichlet prior on `(p₁, p₀-p₁, 1-p₀)`.
Specifically, TAIL sets two tail probabilities:

- `P(p₀ ≤ 0.5) = TAIL` — how unlikely you consider the case where a
  well-calibrated item has less than 50% chance of being answered right.
- `P(p₁ ≥ 0.9 · p₀) = TAIL` — how unlikely you consider the case where
  the next-type-up student answers a question nearly as well as the
  matched-type student (i.e., poor separation).

**Default: 0.01.** Lower = tighter prior (you're more confident that
`p₀` is high and that there's meaningful separation). Higher = more
diffuse prior. Moving this slider changes the implied `p₁` prior mean
and the overall Dirichlet concentration.

#### TAIL_PI — tail tolerance for π concentration

Controls how strong the prior on π is. Specifically, the concentration
`K_π` is auto-calibrated so that under the prior, the probability of
seeing at least one A in a class of size N is at least `1 - TAIL_PI`.

**Default: 0.05.** Lower TAIL_PI → larger K_π → prior dominates the
data more. With TAIL_PI = 0.05, you're saying "it would be very
unusual for a class of this size to have no A-level students at all."

### MCMC knobs

#### MCMC chains

Number of independent Markov chains to run in parallel. **Default: 4.**
More chains improve convergence diagnostics (split-R̂ needs multiple
chains) at a roughly linear cost in runtime. Don't go below 2.

#### Iterations per chain (starting)

Iterations per chain in the first attempt. **Default: 5000.**

If the diagnostic thresholds (R̂ and ESS) aren't met, the tool will
*automatically* double the iteration count and re-run, up to the retry
cap below. So "5000 starting" means the first attempt runs 5000 iters
per chain; if that fails convergence and retries are enabled, subsequent
attempts run 10000, then 20000, etc.

For 4 chains × 5000 iters, budget 30–60 seconds on the free hosting
tier. With retries escalating to 20000 iters, budget 2–3 minutes for
the worst case.

#### Retry doublings if diagnostics are poor

How many times the sampler doubles the iteration count when R̂ or ESS
fails the targets. **Default: 2.** This means up to 3 total attempts:
5000 → 10000 → 20000 iters per chain.

- **0 = no retry.** The sampler runs once with whatever iteration count
  you set. Most transparent; you see exactly what the first attempt did.
- **1, 2, or 3.** Auto-escalation. The final "Iters / chain" diagnostic
  tile in the results shows how many iterations actually ran, so the
  receipt is still visible.

If you want reproducibility across runs, set this to 0 and manually
increase iterations if needed.

#### Max R̂ target

Convergence threshold. If R̂ ≤ this value (across `p₀`, `p₁`, `b`, and
each `π` component), the chain is considered converged. **Default: 1.01.**
Standard Bayesian practice is 1.01 for "clean" convergence and 1.05
for "acceptable."

#### Min ESS target

Effective sample size threshold. If the minimum ESS across parameters is
≥ this value, the chain has enough independent samples. **Default: 400.**

### Where to find the derived values

After you set the Advanced knobs above, the Advanced expander displays:

> **Derived from the above:** implied p₁ prior mean = **X.XXX**  •
> implied concentration (p01_mass) = **X.XX**

This is so you can see what the tail conditions imply for the full
`(p₀, p₁)` prior without having to compute it yourself.
""")

st.markdown("## Small-class warning")

st.markdown("""
If your class has fewer than 15 students, the posterior is heavily
influenced by your prior choices — especially the grade distribution and
the concentration masses (K_π and the `(p₀, p₁)` prior mass). You can
still use BAG, but changing the π preset will visibly change the output
in a way it wouldn't for a larger class.

For tiny classes (under 10 students), consider whether you have enough
information to pick a defensible prior at all, or whether the honest
thing to do is report raw scores.
""")

st.markdown("## Quick summary table")

st.markdown("""
| Parameter | Default | When to change | Risk of changing |
|---|---|---|---|
| π preset | Baseline business-school | Strong prior on class composition | Minor (data updates it) |
| Q | 60 | Whenever you know your exam structure | Affects variance noticeably |
| fix_pi | Off | Mandatory quotas | Loses adaptivity |
| Type-1 error | 0.10 | Very well/poorly calibrated exams | Small; within uncertainty |
| TAIL | 0.01 | Stronger/weaker separation prior | Changes implied p₁ |
| TAIL_PI | 0.05 | Larger/smaller classes, different A-rarity | Changes K_π |
| Chains | 4 | Never reduce; may increase to 6–8 | More = slower, more stable |
| Iterations (start) | 5000 | If you want more up-front precision | More = slower, more accurate |
| Retries | 2 | 0 for reproducibility, 3 for hardest cases | More = auto-escalates silently |
| Max R̂ target | 1.01 | 1.05 for more permissive convergence | Looser → can declare convergence prematurely |
| Min ESS target | 400 | Lower only for demo purposes | Lower → noisier estimates |
""")
