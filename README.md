# BAG Grader

**A free, open-source web tool that turns numerical student scores into letter grades using Bayesian Adaptive Grading.**

Based on Schenone (2025), *The Hitchhiker's Guide to the Grading Galaxy*.

**→ [Live demo](https://bag-grader-pschenone.streamlit.app)**

---

## What does it do?

You paste a list of numerical scores. It runs a Bayesian mixture-model MCMC inference. You get back:

- A per-student letter grade (A, A-, B+, B, B-, ..., E) with uncertainty quantification
- The probability your student belongs to each grade category
- Model-based boundary cutoffs
- Four publication-quality figures
- A one-click PDF grade-justification report
- CSV, JSON, and PNG downloads

No account needed for the public app. Works in your browser.

## Quick start — no terminal required

If you've never used a terminal and just want to deploy your own copy, follow these steps in your browser:

### 1. Create a GitHub account (~2 min)

Go to [github.com](https://github.com). Click **Sign up** in the top-right. Use your email address, pick a password, pick a username (this will be part of your tool's URL). Choose the **Free** plan when prompted.

### 2. Create a Streamlit Community Cloud account (~1 min)

Open a new tab at [share.streamlit.io](https://share.streamlit.io). Click **Continue with GitHub**, then click **Authorize streamlit**. Fill in your name and email when asked.

### 3. Fork this repository

Go to the repository for this project (the page you're reading this README on). In the top-right of the page, click the **Fork** button. On the page that appears, click the green **Create fork** button. You now have your own copy of the code.

### 4. Deploy to Streamlit

Go back to your Streamlit tab at [share.streamlit.io](https://share.streamlit.io). Click the blue **Create app** button, then **Deploy a public app from GitHub**. Fill in the three dropdowns:

- **Repository:** `YOUR-USERNAME/bag-grader`
- **Branch:** `main`
- **Main file path:** `Home.py`

Change the app URL to something memorable (e.g., `smith-grader`). Click the blue **Deploy** button.

Wait 2–5 minutes. The page will show a scrolling log while it installs Python packages. When it finishes, your app is live at the URL shown at the top.

**You're done.** Share the URL with anyone.

---

## Changing things later

Every setting that matters is in a file in your GitHub repo. To change anything:

1. Go to your repo at `github.com/YOUR-USERNAME/bag-grader`
2. Click the file you want to edit
3. Click the ✏️ pencil icon in the top-right
4. Make your edit in the browser
5. Scroll down, type a short description in "Commit changes", click the green **Commit changes** button

Streamlit auto-detects the change and redeploys within 60 seconds.

### Things you might want to change

- **Default grade distribution:** edit `Home.py`, search for `PI_PRESETS`, change the numbers.
- **Tool name / branding:** edit `Home.py`, search for `st.set_page_config` and the hero `<div class="hero-title">`.
- **Theme colors:** edit `.streamlit/config.toml`.
- **Paper link:** edit `pages/4_About.py` and look for the `TODO` comment.
- **GitHub link:** edit `pages/4_About.py` and `CITATION.cff` — search for `YOUR-USERNAME`.

---

## How the tool is organized

```
bag-grader/
├── Home.py                    # main tool (paste scores, run, download)
├── pages/
│   ├── 1_How_it_works.py      # plain-English explanation
│   ├── 2_Statistical_model.py # formal generative model + math
│   ├── 3_Parameters.py        # reference for every knob
│   └── 4_About.py             # paper, citation, license
├── bag/
│   ├── __init__.py            # public API
│   ├── sampler.py             # MCMC sampler (MwG with adaptive Metropolis)
│   ├── visualize.py           # figures
│   └── report.py              # PDF report generator
├── assets/
│   └── sample_scores.csv      # 33-student demo from paper
├── .streamlit/
│   └── config.toml            # theme
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT
├── CITATION.cff               # machine-readable citation
└── README.md                  # this file
```

## Using the Python package directly

If you prefer to run BAG in Python without the web UI:

```python
import numpy as np
from bag import run_bag_sampler, default_config_for_class, diagnose, nearest_granulated_label

scores = np.array([25, 32.5, 35, 45, 55, 65, 75, 82.5, 94])
cfg = default_config_for_class(scores)
out = run_bag_sampler(scores, cfg=cfg)

print("Expected grades:", out["expected_grade"])
print("Letter grades:", nearest_granulated_label(out["expected_grade"]))
print("Diagnostics:", diagnose(out, cfg))
```

## FAQ

**Q: My app says "Your app has gone to sleep."**
A: Streamlit's free tier sleeps apps after 7 days of no traffic. Click the "Wake up" button. Takes ~30 seconds.

**Q: Can I use a custom domain (e.g. `grader.mylab.edu`)?**
A: Streamlit Community Cloud's free tier does not support custom domains. You can embed the Streamlit URL in an iframe on your own site, or pay for a Streamlit plan that supports custom domains.

**Q: Can I use this with real student names?**
A: Only if you are comfortable with your hosting arrangement and institutional rules. For sensitive data, a school-approved deployment is safer than a generic public-hosting setup.

**Q: How long does a run take?**
A: For a typical class (15–40 students) with default settings (4 chains × 10000 iterations), expect roughly 1–2 minutes on Streamlit's free tier.

**Q: The convergence diagnostic is yellow. Should I worry?**
A: Yellow means your results are usable but not perfect. By default, the tool auto-retries with doubled iterations (up to 2 extra attempts) when diagnostics don't meet the strict targets — so if you see yellow at the end, the sampler has already tried escalating. You can raise the retry cap to 3 in the Advanced expander, or manually bump iterations per chain. If it goes red, something's off with your data (e.g., a very unusual score distribution or a class too small for the prior).

**Q: What about the research experiments comparing BAG to Fixed Scale Grading and curving?**
A: Those are a research artifact, not part of the web tool. They'd need to live in a separate `experiments/` directory in this repo, portable from the `Final.py` reference code.

## License

MIT. See [LICENSE](LICENSE) for details.

## Citation

If you use BAG in research, please cite:

```bibtex
@article{schenone2025hitchhiker,
  title   = {The Hitchhiker's Guide to the Grading Galaxy},
  author  = {Schenone, Pablo},
  year    = {2025},
  note    = {Working paper}
}
```
