"""Parameter reference — explains every knob in the tool."""

import streamlit as st

st.set_page_config(page_title="Parameters — BAG Grader", page_icon="📊", layout="wide")

st.markdown("# Parameter reference")
st.caption(
    "Every knob the tool exposes, what it does, when to change it, and "
    "what happens if you get it wrong."
)

st.markdown("## Basic tier — almost always the right thing")

st.markdown("""
### Expected grade distribution (π)

Your prior belief about the proportion of students in each grade category.
The tool provides four presets:

- **Let the data decide (near-uniform prior)** — `[0.2, 0.2, 0.2, 0.2, 0.2]`.
  Recommended default. With very little concentration mass, the data dominates,
  and you get a genuinely data-driven grade distribution.
- **Roughly normal (default)** — `[0.165, 0.378, 0.373, 0.074, 0.010]`. A
  mildly bell-shaped distribution with few A's and E's. Reasonable for a
  well-behaved mid-level class.
- **Skewed high (honors class)** — `[0.30, 0.40, 0.22, 0.06, 0.02]`. For
  selective classes where most students are expected to perform well.
- **Skewed low (intro / weed-out)** — `[0.06, 0.18, 0.38, 0.28, 0.10]`. For
  first-year intro classes, or courses that are designed to separate out
  students who shouldn't continue in the major.

**What if I get it wrong?** With the default concentration mass, the data
will pull the posterior away from a bad prior, but only so far. If your prior
is very badly wrong and your class is small, you'll see it in the posterior.
When in doubt, pick "Let the data decide."
""")

st.markdown("## Customize tier — when you have specifics")

st.markdown("""
### Effective number of exam items (Q)

The number of scorable "units" your exam represents. This directly controls
the **variance** of each component in the mixture: smaller Q means wider
component distributions, so BAG is less willing to draw sharp lines.

**How to compute it:**
- A 60-question multiple-choice exam: Q = 60.
- A 10-question exam graded in half-point increments on a 100-point scale:
  Q = 10 × 2 = 20 (because there are 20 possible "points of progress").
- A single essay graded out of 100 in 5-point increments: Q = 20.
- Two exams averaged together, each Q = 30: Q = 60 (see paper §2.3).

The built-in wizard in the Customize expander will compute Q from two inputs.

**What if I get it wrong?** If you set Q too high (pretending the exam is
more precise than it is), the model will be overconfident in drawing sharp
boundaries. If you set Q too low, the model will be too timid and you'll
get large uncertainty bands. The safe direction to err is low.

### Custom π (sliders)

Sliders for the five grade proportions. They auto-normalize so you don't
need them to sum to exactly 1. Override the preset if none of the four
captures what you want.

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
want to poke at the MCMC internals. Changing them can produce slightly
different numbers — usually within the posterior uncertainty. If you
don't know what MCMC is, leave these alone.

### Type-1 error

The probability that a student of type θ misses a question calibrated for
their own type. Equivalently, $1 - p_0$. Default: **0.10**.

This captures how well-calibrated your exam items are. A value of 0.10
means that even students who *should* get a given item correct will miss
it 10% of the time due to noise. Lower values assume sharper, more
discriminating items.

### p₁ prior mean

The probability that a type-θ student correctly answers an item calibrated
for the **next higher** type. Default: auto-calibrated from the type-1
error.

Lower p₁ means the five component distributions are better separated
(less overlap between A-level and B-level scores). This should
generally be lower than $1 - \\text{type1\\_error}$.

### MCMC chains

Number of independent Markov chains to run in parallel. Default: **4**.
More chains improve convergence diagnostics (R̂ requires multiple chains)
at a roughly linear cost in runtime. Don't go below 2.

### Iterations per chain

Total iterations in each chain (includes 25% burn-in). Default: **5000**.
Longer runs reduce Monte Carlo error and improve ESS; they also take
longer.

For 4 chains × 5000 iters, expect 30–60 seconds on the free hosting tier.
For 4 chains × 15000 iters, budget 2–3 minutes.

If the convergence diagnostic shows yellow ("minor instability") or
red ("poor convergence"), try doubling the iterations and re-running.
""")

st.markdown("## Small-class warning")

st.markdown("""
If your class has fewer than 15 students, the posterior is heavily influenced
by your prior choices — especially the grade distribution and the concentration
mass. You can still use BAG, but be aware that changing the π preset will
visibly change the output. Consider whether you have enough information to
pick a defensible prior, or whether the honest thing to do for a small class
is just use scores directly.
""")

st.markdown("## Quick summary table")

st.markdown("""
| Parameter | Default | When to change | Risk of changing |
|---|---|---|---|
| π preset | Let the data decide | Strong prior on class composition | Minor (data updates it) |
| Q | 60 | Whenever you know your actual exam structure | Affects variance noticeably |
| fix_pi | Off | Mandatory quotas | Loses adaptivity |
| Type-1 error | 0.10 | Very well/poorly calibrated exams | Small; within uncertainty |
| p₁ mean | auto | Very unusual exams | Usually auto is fine |
| Chains | 4 | Never reduce; may increase to 6-8 | More = slower, more stable |
| Iterations | 5000 | If diagnostics say "minor" or "poor" | More = slower, more accurate |
""")
