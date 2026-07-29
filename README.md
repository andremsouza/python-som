# python-som

[![CI](https://github.com/andremsouza/python-som/actions/workflows/ci.yml/badge.svg)](https://github.com/andremsouza/python-som/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-som.svg)](https://pypi.org/project/python-som/)
[![Python versions](https://img.shields.io/pypi/pyversions/python-som.svg)](https://pypi.org/project/python-som/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/andremsouza/python-som/blob/master/LICENSE)

Implementation of Kohonen's 2-D self-organizing map, built on NumPy, with scikit-learn for PCA and
standardization. Accepts NumPy arrays, pandas DataFrames and plain lists.

**[Documentation](https://andremsouza.github.io/python-som/)** ·
**[Changelog](https://github.com/andremsouza/python-som/blob/master/CHANGELOG.md)**

## Install

```bash
pip install python-som              # requires Python 3.10+
pip install "python-som[cli]"       # adds tqdm progress bars
```

## Quick start

```python
import numpy as np
import python_som

rng = np.random.default_rng(0)
data = rng.normal(size=(150, 4))

som = python_som.SOM(x=20, y=None, input_len=4, data=data, random_seed=42)
som.weight_initialization(mode="linear", data=data)
error = som.train(data, n_iteration=len(data), mode="batch")

umatrix = som.distance_matrix()
winner = som.winner(data[0])
```

A full worked example with plots is in [examples/iris.py](https://github.com/andremsouza/python-som/blob/master/examples/iris.py) and in the
[getting-started guide](https://andremsouza.github.io/python-som/getting-started/).

![U-matrix of a SOM trained on Iris](https://raw.githubusercontent.com/andremsouza/python-som/master/docs/assets/iris.png)

## Features

* Stepwise and batch training
* Random, random-sampling and linear (PCA) weight initialization
* Automatic selection of the map size ratio, from PCA
* Cyclic arrays, for toroidal maps
* Gaussian, bubble and Mexican hat neighborhood functions
* Custom decay functions
* Visualization support: U-matrix, activation matrix
* Supervised labelling, via the label map
* Fully type-annotated, with a `py.typed` marker

## Neighborhood functions

All three are functions of the distance between two nodes in the grid, `sqdist(c, i)` in Eq. (5) of
Kohonen (2013).

| Name | Shape | Notes |
| --- | --- | --- |
| `'gaussian'` | `exp(-r² / 2σ²)` | Strictly positive, monotonically decreasing. The default. |
| `'bubble'` | `1` for `max(dx, dy) ≤ σ`, else `0` | The truncated inner lobe of the Mexican hat. Uses the Chebyshev metric, so the region is a square. |
| `'mexicanhat'` | `(1 - u)·exp(-u)`, `u = r² / 2σ²` | Excitatory near the winner, inhibitory beyond it. Zero at `r = √2·σ`, minimum `-e⁻²` at `r = 2σ`. |

The Mexican hat takes negative values, so it **cannot be used with `mode='batch'`**: the batch
update of Kohonen Eq. (8) is a weighted mean whose denominator is not sign-definite for a signed
neighborhood function. Use `mode='random'` or `mode='sequential'`; `mode='batch'` raises a
`ValueError`.

See [Neighborhood functions](https://andremsouza.github.io/python-som/neighborhood-functions/) for
the derivations, including why the Mexican hat is not an outer product of two 1-D wavelets.

## Behavior changes in 0.3.0

0.3.0 corrects several methodology defects. Each changes numerical output, so results are not
comparable across the boundary.

**Seeds no longer reproduce older results.** The random generator is now per-instance rather than a
call to `np.random.seed` on NumPy's global state. That fixes a real side effect, since constructing
a SOM used to reseed NumPy for the whole host program, but it means `random_seed=42` gives a
different map. **To reproduce figures made with an earlier version, pin `python-som==0.2.0`.**

Also changed:

* **Batch training no longer destroys models** whose neighborhood contains no data. They previously
  became the zero vector; on a 30×30 map with 20 samples, 282 of 900 models were wiped in one step.
* **Linear initialization now spans the principal-component plane.** It used the eigen*values* where
  it needed the eigen*vectors*, so every model came out as a constant vector. PCA is also now fitted
  on raw data rather than standardized data, so the models share the input space.
* **`sequential` mode honours `n_iteration`.** It previously ran one pass over the dataset whatever
  you asked for.
* **The automatic map-size ratio uses `√(λ₁/λ₂)`** rather than the raw eigenvalue ratio, and rounds
  rather than floor-dividing, so a side length can no longer come out as zero.
* **The neighborhood radius is floored** at `min_neighborhood_radius` (default 0.5), per Kohonen
  Section 4.2.
* **`get_shape()` returns `tuple[int, int]`** instead of NumPy integers, so `figsize=som.get_shape()`
  works.
* **`requires-python` is `>=3.10`**, which is what the code actually needs.

The full list, with citations, is in the [changelog](https://github.com/andremsouza/python-som/blob/master/CHANGELOG.md).

## Development

```bash
uv sync --all-extras
uv run pytest --cov          # tests and coverage
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # type-check
uv run mkdocs serve          # docs, locally
pre-commit install           # optional, run the gates on commit
```

If you use the SonarQube for IDE (SonarLint) VS Code extension, it will also apply Sonar's Python
rules locally; the ruff configuration is set up to cover most of the same ground.

## References

Based on:

Teuvo Kohonen,
Essentials of the self-organizing map,
Neural Networks,
Volume 37,
2013,
Pages 52-65,
ISSN 0893-6080,
<https://doi.org/10.1016/j.neunet.2012.09.018>

The Mexican hat neighborhood follows the lateral-interaction formulation in:

O. J. Vrieze,
Kohonen network,
in: Artificial Neural Networks: An Introduction to ANN Theory and Practice,
Lecture Notes in Computer Science, Volume 931,
Springer, Berlin, Heidelberg,
1995,
Pages 83-100,
<https://doi.org/10.1007/BFb0027024>

## License

MIT. See [LICENSE](https://github.com/andremsouza/python-som/blob/master/LICENSE).
