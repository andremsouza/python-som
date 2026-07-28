# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-28

### Added

- Mexican hat neighborhood function, selectable with `neighborhood_function='mexicanhat'`
  (contributed by [@Arne49](https://github.com/Arne49) in
  [#2](https://github.com/andremsouza/python-som/pull/2)).

  Implemented as the isotropic 2-D form `(1 - u) * exp(-u)` over `u = sqdist(c, i) / (2 * sigma^2)`,
  normalized so that `h(c, c) = 1`. It crosses zero at a radius of `sqrt(2) * sigma` and reaches its
  minimum of `-exp(-2)` at a radius of `2 * sigma`.

  Note that a neighborhood function must depend on the grid distance alone, not on the two axis
  offsets separately, per `sqdist(c, i)` in Kohonen (2013) Eq. (5) and the "lateral distance"
  abscissa of Vrieze (1995) Fig. 3. The gaussian happens to be separable into a product of per-axis
  terms, but that is a property of the exponential rather than of neighborhood functions in general.

- `## Neighborhood functions` section in the README, comparing all three and documenting the
  batch-mode restriction below.
- Test suite for the neighborhood functions, asserting closed-form analytic properties rather than
  golden values.

### Fixed

- **Cyclic (toroidal) neighborhoods were only correct in one direction.** The fold-back handled
  offsets above `+length/2` but left those below `-length/2` alone, so a winner in the upper half of
  an axis did not wrap. On a 10x10 toroidal map with sigma=1, a winner at row 0 gave `h[9] = 0.6065`
  while a winner at row 9 gave `h[0] = 0.0` instead of the same 0.6065. Now uses the minimum-image
  convention, which folds both tails.

  This changes results for any map built with `cyclic_x=True` or `cyclic_y=True`.

- A neighborhood radius of zero divided by zero and produced `inf`/`nan` weights instead of raising.
  `_gaussian` and `_mexicanhat` now reject a non-finite or non-positive radius; `_bubble` still
  accepts a radius of zero, which selects the winner alone.

- An unknown `neighborhood_function` raised a bare `KeyError`. It now raises `ValueError` listing the
  valid names, matching the behavior of `weight_initialization`.

### Changed

- **`requires-python` is now `>=3.10`.** The declared floor of `>=3.9` was never accurate: the module
  uses PEP 604 unions (`int | None`) in runtime-evaluated annotations without
  `from __future__ import annotations`, so importing it on Python 3.9 raises `TypeError`. Python 3.9
  reached end of life in October 2025. Installers will now decline the package on 3.9 rather than
  install something that cannot be imported.

- `train(mode='batch')` combined with `neighborhood_function='mexicanhat'` now raises `ValueError`.
  The batch update of Kohonen (2013) Eq. (8) is a weighted mean whose denominator `sum_j n_j * h_ji`
  is only well defined for a non-negative neighborhood function. The mexican hat is signed by
  construction, so the denominator can vanish or turn negative: on a 12x12 grid, 49 of 144
  denominators come out negative, which inverts or explodes the update. This is a property of the
  function itself rather than of this implementation. Use `mode='random'` or `mode='sequential'`.

### Packaging

- Restored the `cli` extra (`pip install python-som[cli]`, which pulls in `tqdm` for training
  progress bars). This extra shipped in 0.1.3 on PyPI but was never committed to the repository, so
  releasing from the repository as-is would have silently removed it.

### Note on 0.1.3

There is no changelog entry for 0.1.3 because it was published to PyPI from an uncommitted working
tree and has no corresponding commit. Its `python_som/__init__.py` is byte-identical to the 0.1.2
tag; the only differences are the version string and the `cli` extra restored above.

## [0.1.3] - 2025-01-14

Published to PyPI only. See the note above.

## [0.1.2] - 2025-01-14

Earlier releases predate this changelog. See the
[commit history](https://github.com/andremsouza/python-som/commits/master) for details.

[0.2.0]: https://github.com/andremsouza/python-som/releases/tag/v0.2.0
[0.1.3]: https://pypi.org/project/python-som/0.1.3/
[0.1.2]: https://pypi.org/project/python-som/0.1.2/
