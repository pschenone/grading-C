"""How BAG works — plain-English explanation."""

import streamlit as st

st.set_page_config(page_title="How it works — BAG Grader", page_icon="📊", layout="wide")

st.markdown("# How BAG works")
st.caption("A plain-English walkthrough. No equations. Those live on the next page.")

st.markdown("""
### The problem

You have a class of students. Each student has a numerical score between 0 and 100.
You have to assign each of them a letter grade A, B, C, D, or E.

The usual ways of doing this have problems. If you pre-commit to cutoffs like
"90 and above is an A, 80–90 is a B," you're deciding what a B is **before**
seeing the actual class. Maybe your exam turned out harder than you thought.
Maybe easier. The cutoffs don't adjust.

If you "curve" — assign the top 10% an A regardless of raw scores — you're
essentially saying every student came from the same distribution and differences
between them are luck. That's rarely what you actually believe.

### The BAG idea

**Bayesian Adaptive Grading (BAG)** takes a different view: the students in
your class come from five latent groups — A-level, B-level, C-level, D-level,
E-level — and your job as a grader is to figure out which group each student
belongs to, based on the score they got.

This is a classification problem. BAG treats it as one.

### The five-group model

BAG assumes that:

1. **There are five student types.** A student's type is a reflection of
   their knowledge stock and ability, and it's fixed before the exam.
2. **Each type tends to produce scores in a particular range.** Type-A students
   tend to score high, type-E students tend to score low, the others in between.
3. **But there's noise.** A good student can have a bad day. A lucky guess can
   bump a weaker student higher than usual. So we never see types directly —
   we only see scores, which are a noisy signal.

Your job is to infer types from scores. That's Bayesian inference.

### What BAG actually computes

For each student, BAG produces a **full probability distribution over grades** —
not just "this student is a B" but "this student is 65% B, 25% C, and 10% A."

This is the central thing BAG does differently. Most grading methods give you
one letter. BAG gives you a posterior over all five, and lets you see when
the evidence is strong (95% B, 5% everything else) versus when it's
genuinely on the fence (51% B, 49% C).

### How the letter grade gets assigned

Once we have the full posterior, we need to collapse it back to a single letter
for the transcript. BAG does this by computing an **expected grade value**
(on a 0–4 scale where A=4, B=3, etc.) and mapping it to the nearest granulated
letter (A, A-, B+, B, B-, ...).

This makes "+" and "-" modifiers **honest**: they carry information about
uncertainty. A solid B means "B with high confidence." A B- means "pretty
likely B but there's meaningful C mass." That's a real difference you can
defend to a student who asks.

### How BAG "learns" from your class

BAG has a **prior belief** about how many students of each type there are.
You can pick from six presets — including a baseline business-school
prior (the default), two alternative shapes from the paper, a near-uniform
"let the data decide" option, and honors/intro-class skewed presets.

The sampler then looks at the actual class scores and **updates** that belief.
If your prior said 10% A's but the data strongly suggests 20%, the posterior
shifts toward 20%. If you want to enforce a strict quota, you can tell BAG
not to update — but that's usually not what you want.

### What if the sampler doesn't converge on the first try?

The tool runs a convergence check (split-R̂ and ESS) after each run. If the
thresholds aren't met, by default the tool automatically doubles the
iteration count and re-runs, up to 2 extra attempts (so 3 total: 10000 →
20000 → 40000 iters per chain). You'll see a receipt in the results panel
showing the actual iterations used and the number of attempts.

If you want fully transparent, single-attempt runs, set "Retry doublings"
to 0 in the Advanced expander.

### What you get at the end

- A per-student grade table with the letter, the expected-grade number, the
  probability of the top grade, and a traffic-light confidence flag.
- Four figures showing the fitted mixture components, expected grades with
  uncertainty, expected grade versus score, and the grade-probability curves.
- A per-student probability table with confidence flags and A–E posterior bars.
- Downloadable CSV, JSON, PDF report, and PNG figures.

### When should you not use BAG?

- **Very small classes.** With fewer than 15 students, the posterior is
  heavily prior-dominated. You'll get a warning in that case.
- **Assessments where all students took different exams.** BAG assumes all
  students took the same assessment(s) averaged into the same 0–100 scale.
- **When you have strong reasons to deviate from what the model suggests.**
  BAG is a statistical tool. You remain the instructor. If the model assigns
  a student a grade that contradicts your direct knowledge of them, use your
  judgment — but you now have the full posterior to consult before overriding.

### Where the method comes from

BAG is introduced in Schenone (2025), *The Hitchhiker's Guide to the Grading
Galaxy*. The paper walks through the model formally, proves properties about
it, and benchmarks it against Fixed Scale Grading and curving on synthetic
data. Across a wide range of scenarios, BAG reduces mean misclassification by
55–72% relative to Fixed Scale Grading and 45–68% relative to curving.

See the **Statistical model** page for the math and the **Paper** link in
the sidebar for the full article.
""")
