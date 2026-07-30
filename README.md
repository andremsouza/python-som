# python-som

[![CI](https://github.com/andremsouza/python-som/actions/workflows/ci.yml/badge.svg)](https://github.com/andremsouza/python-som/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/python-som.svg)](https://pypi.org/project/python-som/)
[![Python versions](https://img.shields.io/pypi/pyversions/python-som.svg)](https://pypi.org/project/python-som/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/andremsouza/python-som/blob/master/LICENSE)

Implementation of Kohonen's 2-D self-organizing map. NumPy is the only dependency. Accepts NumPy
arrays, pandas DataFrames, polars, pyarrow, and anything else implementing the `__array__` protocol.

[Documentation](https://andremsouza.github.io/python-som/) ·
[Changelog](https://github.com/andremsouza/python-som/blob/master/CHANGELOG.md)

## Install

```bash
pip install python-som                  # requires Python 3.10+; NumPy is the only dependency
pip install "python-som[cli]"           # adds tqdm progress bars
pip install "python-som[sklearn]"       # adds the scikit-learn estimator adapter
pip install "python-som[examples]"      # adds matplotlib and seaborn, for the plots
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

The same map through the estimator interface, which `Pipeline` and `GridSearchCV` also understand:

```python
som.fit(data, n_iteration=len(data), mode="batch")
labels = som.predict(data)  # (n_samples,) flat node index
distances = som.transform(data)  # (n_samples, x*y)

som.save_npz("map.npz")  # models plus provenance, no pickle
som = python_som.SOM.load_npz("map.npz")
```

A full worked example with plots is in [examples/iris.py](https://github.com/andremsouza/python-som/blob/master/examples/iris.py) and in the
[getting-started guide](https://andremsouza.github.io/python-som/tutorial/first-map/).

![U-matrix of a SOM trained on Iris](https://raw.githubusercontent.com/andremsouza/python-som/master/docs/assets/iris.png)

## Features

* NumPy is the only runtime dependency; a fresh install is 69 MB across one package
* Stepwise and batch training
* Random, random-sampling and linear (PCA) weight initialization
* Automatic selection of the map size ratio, from PCA
* Cyclic arrays, for toroidal maps
* Gaussian, bubble and Mexican hat neighborhood functions
* Custom decay functions
* Save and load a trained map without `pickle`, so loading one cannot execute code
* Provenance: every run records its seed, iteration count, error and library versions
* Works as a scikit-learn estimator: `fit`, `transform`, `predict`, and an adapter for `Pipeline`,
  `GridSearchCV` and `cross_val_score`
* Options accepted as plain strings or enums, with typos caught by a type checker
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

See [Neighborhood functions](https://andremsouza.github.io/python-som/reference/neighborhood-functions/) for
the derivations, including why the Mexican hat is not an outer product of two 1-D wavelets.

## Upgrading

Two releases change numerical results. If you are reproducing a figure, pin the version that made
it.

**0.3.0** corrects several methodology defects, so results are not comparable with earlier versions.
In particular **`random_seed` no longer reproduces pre-0.3.0 maps**: the generator is now
per-instance rather than a call to `np.random.seed` on NumPy's global state. To reproduce older
figures, pin `python-som==0.2.0`.

**0.4.0** corrects linear initialization for data far from the origin. Its PCA previously went
through scikit-learn's `auto` solver, which forms a covariance matrix and loses precision when the
mean is large relative to the spread. On data offset by 1e7 the second explained variance was wrong
by 5.8%.
Near the origin the difference is floating-point noise. Timestamps, coordinates and absolute sensor
readings are the cases that were affected.

0.4.0 also **removed pandas and scikit-learn as runtime dependencies**. If you imported either
transitively through this package, depend on them directly, or install `python-som[examples]`.

**0.5.0** briefly made plain-string options emit a `DeprecationWarning`. **0.6.0 withdrew that**:
strings are permanent and 1.0.0 will not remove them. If you saw that warning, you can stop
migrating.

Each change and the passage of Kohonen (2013) behind it is in the
[changelog](https://github.com/andremsouza/python-som/blob/master/CHANGELOG.md).

## Development

```bash
uv sync --all-extras
uv run pytest --cov          # tests and coverage
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # type-check
uv run bandit -c pyproject.toml -r src/   # security checks
uv run pip-audit             # known vulnerabilities in the resolved set
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
