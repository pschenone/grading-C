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

[Paper page](https://sites.google.com/site/pschenone/home#h.te52tm9oem0y)

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
The source code for this tool lives at
[github.com/pschenone/grading-C](https://github.com/pschenone/grading-C).

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


st.markdown("## Data handling note")

st.markdown("""
BAG Grader is a hosted web application. If you plan to use real student names
or other sensitive data, use a hosting arrangement that fits your institutional
requirements. The app is designed for transient interactive use, but the precise
data-handling properties depend on how and where you deploy it.
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
