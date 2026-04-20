"""About BAG Grader — paper, citation, code, license."""

import streamlit as st

st.set_page_config(page_title="About — BAG Grader", page_icon="📊", layout="wide")

st.markdown("# About BAG Grader")

st.markdown("""
BAG Grader is a web-based implementation of **Bayesian Adaptive Grading**,
a statistical method for turning numerical student scores into letter grades
with quantified uncertainty.

It is open-source, free to use, and free to self-host. No account required,
no data stored. It is intended for instructors and researchers who want
a defensible, reproducible grade-assignment process backed by published
methodology.
""")

st.markdown("## The paper")

st.markdown("""
> **Schenone, P. (2025).** *The Hitchhiker's Guide to the Grading Galaxy.*

<!-- TODO: Replace with the actual paper URL (arXiv / SSRN / journal) -->
[Placeholder link to paper](https://example.com/hitchhiker-grading-galaxy)

The paper develops the full theoretical framework: a microfoundation for
student types based on effort cost and productivity, a formal model of
exams that induce score distributions, the Bayesian Adaptive Grading
algorithm, and comparative Monte Carlo simulations against Fixed Scale
Grading and curving.
""")

st.markdown("## Citation")

st.code("""@article{schenone2025hitchhiker,
  title   = {The Hitchhiker's Guide to the Grading Galaxy},
  author  = {Schenone, Pablo},
  year    = {2025},
  note    = {Working paper}
}""", language="bibtex")

st.markdown("""
A machine-readable `CITATION.cff` file ships in the source repository,
so GitHub auto-generates a "Cite this repository" button.
""")

st.markdown("## Source code")

st.markdown("""
<!-- TODO: replace YOUR-USERNAME with the actual GitHub handle -->
The source code for this tool lives at
[github.com/YOUR-USERNAME/bag-grader](https://github.com/YOUR-USERNAME/bag-grader).

- The full Python package implementing the sampler is in `bag/`.
- The Streamlit web UI is in `Home.py` and `pages/`.
- The research-grade experimental harness comparing BAG to Fixed Scale
  Grading and curving (Cells 4-5 of the original paper code) ships as a
  separate script in the repo under `experiments/` for reproducibility,
  but is not part of the web interface.

To report a bug, request a feature, or discuss the methodology, open an
issue on GitHub.
""")

st.markdown("## License")

st.markdown("""
BAG Grader is released under the **MIT License**. You are free to use,
modify, fork, redistribute, and build on it — including for commercial
purposes — provided you preserve the copyright notice. See `LICENSE` in
the repository for the full text.

The research methodology is the intellectual work of Pablo Schenone. If
you use BAG in your own work, please cite the paper.
""")

st.markdown("## Privacy")

st.markdown("""
BAG Grader does not store your data.

- Scores and any IDs you enter are held in your browser session only.
- Nothing is transmitted to an external server beyond the hosting
  platform's normal operation.
- Refreshing the page or closing the tab clears everything.
- The downloadable outputs (CSV, JSON, PDF) contain only what you put in,
  plus the model output.

This makes BAG Grader FERPA-safe by default: no personally identifiable
student data persists anywhere. You can safely use it with real student
names if you wish, though the default "Student 1, Student 2, ..." anonymous
labeling is a good practice.
""")

st.markdown("## Acknowledgments")

st.markdown("""
The web interface was built by combining the research methodology in
Schenone (2025) with the Python implementation that accompanies the paper.
The statistical core (the Metropolis-within-Gibbs sampler with adaptive
Metropolis proposals, the Dirichlet-prior parametrization, and the
convergence diagnostics) is a direct port of the paper's reference code.
The UI is original.
""")

st.markdown("## Version")

st.markdown("""
This instance of BAG Grader is at version **0.2.0**.
""")
