# Convex Optimization in Computer Graphics

Code and notebooks for the Eurographics 2026 tutorial Convex Optimization in Computer Graphics, presented by [Leticia Mattos Da Silva](https://www.lmattos.com/).

## What this tutorial covers

A number of tasks in computer graphics can be written as critical-point conditions of an optimization problem. Many of those problems are nonlinear or nonconvex, which makes them painful to solve reliably. This tutorial walks through *convex relaxation*, the strategy of replacing a difficult non-convex problem with a convex one. The notebooks cover the basics of convex optimization in the first half and a guided tour of a number of graphics applications in the second half. By the end you should be comfortable picking up a convex modeling tool like CVXPY, recognizing the standard problem classes, and porting an idea from a paper to a working prototype.

The tutorial is aimed at graduate students. We assume working knowledge of linear algebra and multivariate calculus. Familiarity with basic optimization or geometry processing is helpful, but not required.

## Repository layout

```
convex-optimization-graphics.github.io/
├── notebooks/      python notebooks
├── src/            core functions
├── utility/        helpers
├── data/           meshes and other data
├── external/       git submodules for official code implementation
└── docs/           course syllabus and slides
```

The notebook order matches the slide deck:

1. The fundamentals
2. Convexity
3. Why convexity?
4. Standard convex problems (LP, QP, QCQP, SOCP, SDP)
5. Viscosity solutions
6. Mapping problems
7. Optimal transport
8. SOS relaxation

The first four notebooks are the basics. The last four revisit each section of the slides with a code example, and sometimes demos, based on one or more of the paper covered in the tutorial.

## Getting started

We recommend you create a conda environment:

```bash
conda env create -f environment.yml
conda activate convex-graphics
jupyter lab
```

Either path will install CVXPY the other dependencies the notebooks rely on. If you want to use Mosek (recommended) you will also have to download and save a license. The full list is in `environment.yml`.

### Submodules

The notebooks in Part II reference official code implementation from other repositories. To pull those in, run the following:

```bash
git submodule update --init --recursive
```

If you cloned this repo without `--recurse-submodules`, the command above is what you want.

### Optional: Mosek

Several of the SDP examples in Part II are noticeably faster with [Mosek](https://www.mosek.com/), which offers a free academic license. The notebooks fall back to other solvers if Mosek is not available, so the tutorial runs end-to-end either way.

## How to follow along

Each notebook is self-contained. We recommend going through them in order, since later notebooks use ideas (and sometimes helpers) introduced in earlier ones.

Notebooks also have some simple optional exercises indicated by **▶ Your turn.**

## Acknowledgements

Thanks to Ahmed Mahmoud, Chris Scarvelis and Justin Solomon for advising on the SGP 2025 version of this tutorial, and to the authors of all the papers featured in Part II for making their code available!

## Citing this tutorial

If you use these notes in your own teaching or research, please cite the course materials:

```bibtex
@misc{mattosdasilva2026convex,
  title  = {Convex Optimization in Computer Graphics},
  author = {Mattos Da Silva, Leticia},
  year   = {2026},
  note   = {Eurographics Tutorial},
  url    = {https://convex-optimization-graphics.github.io/}
}
```
