# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-07-31

0.7.0 shipped optional numba acceleration that a user had no way to find or to check. This makes it
discoverable and adds one public function to inspect it. No behaviour changes and no numerical
changes.

### Added

- **`python_som.accelerated()`**, which reports whether training will use the compiled kernel:

  ```python
  import python_som

  python_som.accelerated()  # True once numba is installed
  ```

  A function rather than a constant because a constant would have to resolve at import time, which
  forces the numba import on every `import python_som`. That import costs about 80 ms, and 0.7.0
  deferred it deliberately; a function defers it to the caller who asked. Named for the capability
  rather than the backend, so a future change of backend is not a rename.

  Calling it imports numba but does not compile the kernel: `njit` is lazy, so the roughly 500 ms
  compile happens on the first call that trains.

- **[Speed up training](https://andremsouza.github.io/python-som/how-to/speed-up-training/)**, a
  how-to covering the four levers in order of effect: batch mode, the default distance function,
  numba, and map size.

### Fixed

- **The numba acceleration is now findable.** 0.7.0's Install section listed four `pip install`
  lines and numba was not among them; the only user-facing mention was one bullet in the feature
  list, and the documentation site mentioned numba just once, in a caveat about MiniSom. A published
  PyPI description cannot be edited, so correcting it takes a release. The README's Install section
  now carries the command, its NumPy consequence, and a link to the how-to, and a packaging test
  asserts that section of the built metadata rather than the description as a whole.

## [0.7.0] - 2026-07-31

Batch training is 20x to 40x faster and now beats both comparable libraries by a wide margin. The
arithmetic is Kohonen's, reorganised; results move by about 1e-15, which makes this a minor release
rather than a patch.

### Changed

- **Batch training is 20x to 40x faster.** Measured against 0.6.1, and against MiniSom and SOMPY
  under the fairness protocol from the comparison page:

  | map | 0.6.1 | 0.7.0 | vs MiniSom | vs SOMPY |
  | --- | --- | --- | --- | --- |
  | 20x20 | 234.2 ms | 12.1 ms | 23.1x faster | 26.2x faster |
  | 40x40 | 992.4 ms | 22.7 ms | 30.9x faster | 70.3x faster |
  | 60x60 | 2780.2 ms | 54.0 ms | 30.2x faster | 94.4x faster |
  | 100x100 | 13972.8 ms | 340.1 ms | | |

  Through 0.6.1 batch training was 1.3x to 1.6x *slower* than MiniSom. Stepwise training is
  unchanged and remains about 10% faster than MiniSom's.

  Two changes. Eq. (8) sums over every pair of nodes, and because the neighborhood depends only on
  the offset between two nodes that sum is a convolution; both neighborhoods batch training admits
  are separable, so it contracts to two matrix products instead of one pass per node. And the
  best-matching-unit search expands the Euclidean norm into a single matrix product rather than one
  full-grid norm per sample. Kohonen derives Eq. (8) from Eq. (7) on the same grounds, that "the
  same addends occur a great number of times" (Section 4.4).

  Peak memory did not rise to pay for it: 3.7 MB against 4.6 MB at 100x100.

- **Results are not bit-identical to 0.6.1.** The contraction sums the same terms in a different
  order, so trained weights differ by 5.9e-16 to 8.5e-16 relative. Below anything a result depends
  on, and enough to break an exact-equality check. Pin `python-som==0.6.1` to reproduce an older
  figure exactly.

### Added

- **Optional numba acceleration** for the winner search, used automatically when numba is present:

  ```bash
  pip install numba
  ```

  Worth 1.0x to 2.4x on top of the above, with bit-identical results, and uneven: where the
  neighborhood update dominates it changes little. Deliberately neither a dependency nor an extra.
  numba 0.66 requires `numpy<2.5` while this package releases against 2.5, so requiring it would cap
  every user's NumPy, and declaring it as an extra caps the lockfile, since uv resolves every extra
  together. A plain `pip install python-som` is still NumPy and nothing else, which a CI job checks.
  numba is imported on first use, so installing it does not slow `import python_som`.

- **Two explanation pages**: how batch training is computed, and why linear initialization is an
  SVD. Both hold derivations that used to sit in docstrings.

### Removed

- **The neighborhood kernel machinery in `python_som._core`**: `gaussian_kernel`, `bubble_kernel`,
  `mexican_hat_kernel`, `kernel_view`, `NEIGHBORHOOD_KERNELS`, `resolve_kernel` and `offset_span`.
  The axis-matrix contraction replaced them and they had no remaining caller. Private names, so no
  supported API changes.

### Deprecated

- **`KernelFunction`**, which is in `python_som.__all__` and no longer describes anything the
  package produces. It still works and is removed at 1.0.0.

### Fixed

- **The winner search is exact for data far from the origin.** Expanding the Euclidean norm is
  faster and cancels catastrophically when the models sit far from zero: at an offset of 1e9,
  `||w||^2` is around 1e18 while the differences between models are of order 1, and a naive
  expansion gives a different node for 499 of 500 samples. Centring both sides on the models' mean
  is exact, costs 1%, and removes it at every offset tested up to 1e12. This shipped correct; the
  entry is here because it is the same failure mode 0.4.0 fixed in linear initialization, and
  because a regression test now pins it.

## [0.6.1] - 2026-07-30

### Fixed

- **The package description on PyPI now describes the current package.** It had not been updated
  since 0.3.0, so it listed none of the features added in 0.4.0, 0.5.0 or 0.6.0: not the drop to
  NumPy as the only dependency, not `save_npz`/`load_npz`, not the training reports, and not the
  scikit-learn estimator interface that 0.6.0 exists to add.

  A PyPI description cannot be edited after upload, so correcting it takes a release. That is the
  whole of this one: no code changed, and results are identical to 0.6.0.

  The upgrade notes were also short of two things worth knowing. 0.4.0 changes
  linear-initialization results for data far from the origin, where the previous PCA path was wrong
  by up to 5.8%, and it removed pandas and scikit-learn as runtime dependencies, which matters to
  anyone who was importing either transitively. Both are now in the README, alongside a note that
  0.5.0's string deprecation was withdrawn so anyone who saw the warning knows to stop migrating.

## [0.6.0] - 2026-07-30

A map can now be used as a scikit-learn estimator, and the plain-string deprecation from 0.5.0 is
withdrawn. Nothing breaks and no numerical results change: `train` is untouched, both option
spellings work, and scikit-learn remains optional.

### Added

- **`python_som.sklearn.SOMEstimator`**, a scikit-learn estimator, behind a new `sklearn` extra:

  ```bash
  pip install "python-som[sklearn]"
  ```

  The methods on `SOM` are enough when you call them yourself. They are not enough when scikit-learn
  does the calling: since 1.7, `Pipeline.predict`, `GridSearchCV` and `cross_val_score` all require
  `__sklearn_tags__`, which in practice means inheriting `BaseEstimator`. So the integration lives
  in its own module and scikit-learn stays optional. Importing `python_som` still pulls in nothing
  but NumPy, which a test asserts in a subprocess.

  `input_len` is absent from its constructor, since scikit-learn infers the feature count from `X`.
  Unlike `SOM.fit`, the estimator's `fit` starts over rather than continuing, because `GridSearchCV`
  fits one cloned estimator fold after fold, and carrying weights across folds would leak one fold
  into the next.

  All five integration points have tests that call the real library rather than asserting about it:
  `clone`, `Pipeline.fit`, `Pipeline.predict`, `GridSearchCV`, `cross_val_score`.

- **An estimator interface on `SOM`**: `fit`, `transform`, `predict`, `fit_transform`, `score`,
  `get_params` and `set_params`, modelled on `KMeans`. `train` remains the primary way to train and
  is unchanged, so nothing existing is affected.

  `predict` returns a **flat** node index rather than `(row, column)`, because a 1-D label array is
  what scorers, `confusion_matrix` and `cross_val_score` assume;
  `np.unravel_index(som.predict(X), som.get_shape())` recovers the grid position, and `winner()`
  still returns coordinates. `score` is the *negated* quantization error, since `GridSearchCV`
  maximises and without the sign a search would select the worst map.

  Fitted attributes follow the convention: `weights_` for `cluster_centers_`,
  `quantization_error_` for `inertia_`, plus `n_features_in_`.

  Two deliberate differences from scikit-learn, both documented: `fit` *continues* rather than
  resetting, because a SOM's models are its state; and the models exist before training, since
  initialization is a separate step.

  `set_params` changes the rates, radii, decays and distance function. It refuses to change the grid
  shape or `input_len`, which would leave a map whose models no longer match its own description.
  It is what `set_learning_rate` and `set_neighborhood_radius` become; both still work.

  These names alone are **not** enough for `Pipeline.predict` or `GridSearchCV`, which since
  scikit-learn 1.7 require `__sklearn_tags__`. A separate `python_som.sklearn` adapter follows.

### Changed

- **The plain-string deprecation introduced in 0.5.0 is withdrawn.** Strings are permanently
  supported, the `DeprecationWarning` is gone, and 1.0.0 will not remove them. If you migrated to
  the enums while 0.5.0 was current, nothing you wrote breaks: the enums are staying too.

  0.5.0 was wrong, and the reason is worth stating rather than quietly reverting. Every comparable
  library passes options as plain strings and none export enums: scikit-learn
  (`KMeans(init="k-means++")`), numpy (`np.pad(mode="constant")`), scipy
  (`linkage(method="single")`), and both SOM libraries, minisom and sompy. Removing the string form
  would have made this the only library in its ecosystem to reject `mode="batch"`.

  The benefit originally claimed for enums was that a type checker catches typos. That is already
  delivered by the `Literal` unions added in 0.4.0, with strings still working: `mode="bacth"` is a
  type error while `mode="batch"` is not. The deprecation bought nothing that was not already had.

  The decision was made without checking what comparable libraries do. That check is now part of
  planning any future API change.

## [0.5.0] - 2026-07-30

One change: the plain-string options now warn. Nothing else moves, and no result changes.

This is the release 1.0.0 was waiting on. The policy is that anything removed in a major release
warns for at least one minor release first, so strings could not be removed until this shipped.

### Changed

- **Plain strings now emit `DeprecationWarning`.** 0.4.0 deprecated them in writing only, because
  `mode="batch"` was what the documentation showed at the time. This is the warning that follows the
  written notice, and it is the last step before 1.0.0 removes strings entirely.

  The message names the exact replacement, so migrating is a substitution you read off it:

  ```
  Passing mode='batch' as a plain string is deprecated and will stop working in 1.0.0.
  Use TrainingMode.BATCH instead.
  ```

  Two things it deliberately does not do. It does not fire for a spelling that was never valid, so
  `mode="stochastic"` still raises its `ValueError` instead of burying the real mistake under a
  deprecation notice. And it does not fire when a map is loaded from a `.npz`: the name in an
  artifact is a serialisation detail, since JSON has no enums, so the loader converts it rather than
  warning about a string the caller never wrote.

  To silence it while migrating, filter `DeprecationWarning` for the `python_som` module.

### Fixed

- Error messages showed an enum's repr when one was passed. `weight_initialization(mode=WeightInit
  .LINEAR)` reported `<WeightInit.LINEAR: 'linear'> initialization requires ...`. Both spellings now
  produce identical text.

## [0.4.0] - 2026-07-30

`numpy` is now the only runtime dependency, and a trained map can be saved without `pickle`.

**Results change in two places.** Linear initialization is corrected for data far from the origin,
where 0.3.0 was wrong by up to 5.8%; and anyone who imported pandas or scikit-learn transitively
through this package must now depend on them directly. Everything else is additive: no public name
or signature is removed, and `mode="batch"` and the other plain strings keep working.

To reproduce a figure made with an earlier version, pin that version.

### Added

- **`save_npz` and `load_npz`**, so a trained map can be kept without reaching for `pickle`. One
  `.npz` holds the models plus a JSON block with the configuration, the seed, the generator state
  and
  the last training report. No pickle on either side, so loading one cannot execute code.

  A reloaded map continues the same random stream, so training it further gives what never
  stopping would have given. Both the seed and the generator's current state are saved: the seed
  alone
  would restart the stream and a resumed run would quietly diverge from an uninterrupted one.

  Strategies are saved by name and resolved through registries. A map built from the shipped
  functions
  round-trips completely; one built with your own function records the name for provenance and
  raises
  `ArtifactError` on load, naming the argument to pass it back through. A `functools.partial` is
  treated as custom, so `partial(exponential_decay, factor=3.0)` cannot silently reload as
  `factor=2.0`.

  On safety: `allow_pickle=False` is passed explicitly, names resolve only through the registries,
  and
  the member list, JSON, format version and weight shape are all validated before a map is built.
  What
  that buys is "cannot execute code", not "safe to load anything". An `.npz` is a zip, so a hostile
  file can still attempt resource exhaustion. `docs/save-and-load.md` states both halves.

- **`SOM.last_report`**, a frozen `TrainingReport`: mode, iterations, sample count, seed, the final
  learning rate and radius, quantization error, the `python_som` and NumPy versions, and wall time.
  `train()` still returns the quantization error, so nothing that used the return value changes.
  `final_learning_rate` is `None` for batch training, which has no step size: Eq. (8) is a weighted
  mean, and reporting the unused initial value would read as though it had been applied.

- **`SOMConfig`**, the serialisable description of a map, available as `som.config()`.

- **`DECAY_FUNCTIONS` and `DISTANCE_FUNCTIONS`** registries with `resolve_decay` and
  `resolve_distance`, mirroring `NEIGHBORHOOD_FUNCTIONS`. These names are written into artifacts, so
  they are public API from 0.4.0.

- An `analysis` extra with pinned `bandit`, `pip-audit` and `pylint`, for the deeper review passes.
  Not wired into any gate; ruff and mypy remain the enforced checks. CI names the extras it needs
  rather than using `--all-extras`, so these are not installed on every matrix entry.

- **Enums for every string-valued option**: `TrainingMode`, `Neighborhood`, `WeightInit` and
  `SampleMode`. Each member *is* a `str`, so `TrainingMode.BATCH` and `"batch"` are
  interchangeable: equal, same hash, same JSON, usable as the same dictionary key. Nothing that
  accepted a string stops accepting one.

  The parameters are typed as the enum *or* the exact set of valid strings, so a type checker now
  rejects `mode="bacth"` while accepting both `mode="batch"` and `mode=TrainingMode.BATCH`. If you
  build an option from configuration, annotate it with the matching `TrainingModeStr` alias or call
  `TrainingMode(value)` to validate it at runtime.

  **Plain strings are deprecated but do not warn yet.** They warn in 0.5.0 and are removed in 1.0.0.
  Warning now would fire on `mode="batch"`, which is what the documentation showed until this
  release, so the written notice comes first. See `docs/options-and-types.md`.

- **`Protocol`s for the three replaceable strategies**: `NeighborhoodFunction`, `DecayFunction` and
  `DistanceFunction`, plus `KernelFunction` for the batch kernel. Structural, so nothing inherits
  from them and every existing callable already satisfies its protocol; their parameters are
  positional-only, so your own parameter names are your own. A type checker now checks a
  user-supplied strategy against a named contract instead of a bare `Callable`.

### Changed

- **`learning_rate` is validated**, having been unchecked. A non-positive or non-finite rate raises
  `ValueError`: `0` freezes every model so training completes and changes nothing, and `-1` moves
  models away from the samples they match, taking the quantization error from 0.0 to 11.7. Both were
  previously accepted in silence.

  A rate above 1 emits `UserWarning` rather than raising. It overshoots and oscillates but does not
  necessarily diverge: at `alpha = 5` with decay disabled the largest weight stayed at 3.61, because
  the neighborhood damps the correction away from the winner. Kohonen gives no upper bound, so
  rejecting it would invent a limit the sources do not.

- The package is now a pure functional core with a thin shell around it. `python_som._core` holds
  every numeric decision as functions over NumPy arrays and imports nothing but NumPy;
  `python_som._convert` adapts pandas at the boundary; `python_som._som` keeps the state, the
  validation and the training loops. No public name, signature or numerical result changes:
  trained weights are bit-identical to 0.3.0 for every combination of training mode and neighborhood
  function, which is asserted rather than assumed.

  The boundary is machine-checked, not merely documented: ruff's `TID251` bans pandas and
  scikit-learn outside the two shell modules that exist to adapt them, and reports the reason at the
  point of violation. `architecture-profile.toml` records the style.

- The two update rules are now pure functions returning new arrays. An earlier note claimed this
  was also faster. It is not, and the claim is withdrawn. Measured with interleaved arms and
  medians, the pure form is about 10% slower on small maps and indistinguishable above roughly
  100x100. The cost buys functions testable without constructing a network, and is single-digit
  milliseconds over a 10,000-iteration run on a 20x20 map. `benchmarks/bench_update.py` is the
  corrected measurement.

### Performance

- **Batch training is 1.2x to 1.5x faster**, with identical results. Eq. (8) needs the neighborhood
  between every pair of nodes, and the previous implementation got it by evaluating the neighborhood
  function once per node, once per iteration: 1600 full-grid evaluations per iteration on a 40x40
  map. Profiling made that 42% of batch training, more than the contraction it feeds.

  It is unnecessary, because a neighborhood depends only on the offset between two nodes and never
  on where the winner sits. One kernel over every reachable offset serves the whole grid, and
  each node's neighborhood is a slice of it, as a view rather than a copy. The kernel is `4xy`
  floats, 111 KB on a 60x60 map, against the 800 MB a full `(x, y, x, y)` tensor would need.

  The offset-only dependence holds on a torus as well as a flat grid, because making the fold depend
  on the offset alone is exactly what the minimum-image convention does. It holds per axis, so mixed
  maps (one axis wrapping, one not) need no special case.

  Results do not change: trained weights are bit-identical for all 72 working combinations of
  training mode, neighborhood function, initializer and cyclic setting, and the kernel is checked
  against per-node evaluation across 40,832 combinations at exactly `0.0`. The gaussian gains more
  than the bubble, whose per-node form is cheaper to begin with. `benchmarks/bench_batch.py` is the
  measurement.

  Stepwise training is unaffected: it already evaluates the neighborhood once per iteration, so
  there is nothing to amortize.

### Fixed

- **Linear initialization was inaccurate for data far from the origin.** Through 0.3.0 its PCA went
  to scikit-learn's `auto` solver, which since 1.5 selects `covariance_eigh` when samples outnumber
  features. It forms the covariance matrix, and squaring the data squares its condition number. On
  `(150, 4)` data offset by 1e7, the second explained variance was wrong by 5.8%, and the
  resulting models differed from the correct ones by 2.43 against a total model spread of 2.0: an
  error larger than the structure being initialized.

  The NumPy implementation decomposes the centred matrix directly and agrees with a `longdouble`
  reference to ~1e-15. Data offset far from the origin is not exotic: timestamps, easting/northing
  coordinates and absolute sensor readings all look like it.

  This changes results. For data near the origin the difference is floating-point noise (~1e-15,
  and `auto_dimensions` is unaffected because it standardizes first). Far from the origin it is
  large, and 0.4.0 is the correct one.

- `_train_stepwise` looked up `array[index]` twice per iteration, allocating the sample twice on the
  hot loop. Found by profiling the refactor rather than by reading it.

### Removed

- **pandas and scikit-learn are no longer required.** `numpy` is the only runtime dependency.
  Measured on Linux/CPython 3.12, a fresh install drops from 333 MB across 10 packages to 69 MB
  across 1, a 264 MB reduction and 79% of the payload, since the two also pulled in scipy, joblib,
  narwhals, python-dateutil, six and threadpoolctl.

  Nothing is lost, and input support is wider. pandas was used for one `isinstance` check before
  calling `.to_numpy()`; `np.asarray` already does that via the `__array__` protocol. Because
  that is a protocol rather than a library, polars, pyarrow, xarray and CuPy objects now work too,
  without this package knowing they exist.

  scikit-learn supplied one PCA and one z-score, now about twenty lines of `np.linalg.svd`. Both
  libraries remain in the `dev` extra, where `tests/test_linalg_matches_sklearn.py` re-derives every
  fit both ways on every CI run rather than trusting a one-off measurement. `pip install
  python-som[examples]` restores the previous install set for anyone relying on it.

  If you imported pandas or scikit-learn *transitively* through this package, add them to your own
  dependencies. That reliance was always an accident of packaging.

- The batch denominator's `1e-12` tolerance, replaced by `> 0`. The constant had no source, and it
  guarded a condition that cannot occur: every term of `sum_j n_j h_ji` is non-negative, because
  batch training rejects signed neighborhoods and only registered names resolve, so a sum of
  non-negative floats is zero exactly when every term is zero. The 0.3.0 entry below describes it as
  "compared against a tolerance rather than exact zero"; that was the right instinct applied to a
  problem that was not there.

## [0.3.0] - 2026-07-29

Modernizes the packaging and corrects four methodology defects. **Numerical results differ from
0.2.0.** The changes are listed with the passage of Kohonen (2013) each one follows.

### Changed, and results differ

- **Seeds no longer reproduce 0.1.x/0.2.0 results.** The random generator is now a per-instance
  `np.random.Generator`. Previously the constructor called `np.random.seed`, which reseeded NumPy's
  *global* generator as a side effect of building a SOM, disturbing the whole host program. The
  side effect is gone, but `random_seed=42` now yields a different map.

  To reproduce figures made with an earlier version, pin `python-som==0.2.0`.

- Batch training no longer destroys models whose neighborhood contains no data. `_train_batch`
  built its result from a zeroed array and assigned it wholesale, so every model it did not touch
  became the zero vector. On a 30x30 map with 20 samples and a small radius, that wiped 282 of 900
  models in a single step, and drove the quantization error from 0.0 to 0.196. Untouched models now
  keep their previous value, and the denominator is compared against a tolerance rather than exact
  zero. (Kohonen Eq. 8.)

- Linear initialization now spans the principal-component plane. It used
  `pca.explained_variance_`, the two eigen*values*, where it needed `pca.components_`, the
  eigen*vectors*. Every model came out as a scalar broadcast across all features: a constant vector
  on the all-ones diagonal, rank 1 rather than a plane. PCA is also now fitted on the raw data
  instead of standardized data, so the models live in the same space as the inputs they are compared
  against during training. (Kohonen Section 4.3.)

- `random` training now samples i.i.d. with replacement. It previously drew with
  `replace=(n_iteration > len(data))`, so it was a random permutation whenever the iteration count
  did not exceed the sample count, and independent draws only beyond it. The character of the
  sampling therefore depended on how many iterations you asked for. It is now uniformly i.i.d.,
  which is the stochastic approximation of Robbins and Monro (1951) that Kohonen cites in
  Section 4.1.

- `sequential` training honours `n_iteration`. It iterated the dataset exactly once regardless
  of the requested count, so asking for 500 iterations over 30 samples ran 30 steps while the decay
  functions still used 500 as their horizon, leaving the learning rate almost unchanged. It now
  cycles the dataset until the requested number of steps has run.

- The automatic map-size ratio uses `sqrt(lambda_1 / lambda_2)` rather than the raw eigenvalue
  ratio, rounds instead of floor-dividing, and clamps to at least 1 so a side can no longer come out
  as zero. Kohonen Section 3.5 asks for side lengths matching "the lengths of the two largest
  principal components"; a component is a unit direction, so its length in the data is the standard
  deviation. For Iris with `x=20`, `y` moves from 6 to 11.

- The neighborhood radius is floored at the new `min_neighborhood_radius` parameter, default
  0.5. Kohonen Section 4.2: "the final value of sigma shall not go to zero, because otherwise the
  process loses its ordering power. It should always remain, say, above half of the grid spacing."
  This also stops `linear_decay` from reaching a division by zero at its horizon.

- `get_shape()` returns `tuple[int, int]` rather than NumPy integers, so
  `plt.figure(figsize=som.get_shape())` works without the `float()` workaround the old example
  needed.

### Added

- `mexican_hat` is accepted as an alias for `mexicanhat`.
- A test suite, where there was none: 190 tests at 100% coverage, including one regression test
  per defect above carrying its measured numbers, and property-based tests via Hypothesis over
  generated grid sizes, radii and centres.
- Type annotations are now exported, via a `py.typed` marker. The library was already annotated,
  but without the marker PEP 561 requires downstream type checkers to ignore it.
- Documentation site at <https://andremsouza.github.io/python-som/>, built with Material for
  MkDocs, including the derivations behind the neighborhood functions.
- Continuous integration across Python 3.10 to 3.13, with ruff, mypy `--strict`, coverage
  gating, and a build that installs the wheel into a clean environment and imports it.
- Releases publish through PyPI Trusted Publishing, so no long-lived API token exists.

### Performance

- Batch training is about 30x faster. The nested Python loop over every pair of nodes is
  replaced by a NumPy contraction. Measured on a 20x20 map with 150 samples over 5 iterations: 1.89
  s
  to 0.06 s. A regression test asserts the new result matches the old formulation to 1e-12.

### Internal

- The package moves to a `src/` layout and splits into `_som`, `_neighborhood`, `_decay` and
  `_distance`. `import python_som; python_som.SOM(...)` is unchanged, and the pre-0.3.0 private
  names
  (`_asymptotic_decay`, `_euclidean_distance`, and the rest) still resolve.
- `test_iris.ipynb` is removed; its narrative moved into the documentation. `SOM_atualizado.py`, an
  untracked experimental variant that the sdist had been sweeping up, is gone.
- `print` calls become `logging`; `verbose=True` still drives the tqdm progress bar.

### Fixed

- The `bubble` neighborhood is documented as using the Chebyshev metric, so its region is a
  square rather than a disc. The behavior is unchanged and follows Vrieze's appendix pseudo-code
  (`b = MAX(ABS(i - w_i), ABS(j - w_j))`), but it was previously unstated, and it means the bubble
  is
  *not* isotropic under the Euclidean metric. The test suite shipped in 0.2.0 asserted that it was;
  that assertion passed only because no counterexample exists on a grid that small. The smallest is
  at radius `sqrt(50)`, where `(5, 5)` is inside a `sigma = 5` square and `(7, 1)` is outside.

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

- Cyclic (toroidal) neighborhoods were only correct in one direction. The fold-back handled
  offsets above `+length/2` but left those below `-length/2` alone, so a winner in the upper half of
  an axis did not wrap. On a 10x10 toroidal map with sigma=1, a winner at row 0 gave `h[9] = 0.6065`
  while a winner at row 9 gave `h[0] = 0.0` instead of the same 0.6065. Now uses the minimum-image
  convention, which folds both tails.

  This changes results for any map built with `cyclic_x=True` or `cyclic_y=True`.

- A neighborhood radius of zero divided by zero and produced `inf`/`nan` weights instead of raising.
  `_gaussian` and `_mexicanhat` now reject a non-finite or non-positive radius; `_bubble` still
  accepts a radius of zero, which selects the winner alone.

- An unknown `neighborhood_function` raised a bare `KeyError`. It now raises `ValueError` listing
  the
  valid names, matching the behavior of `weight_initialization`.

### Changed

- `requires-python` is now `>=3.10`. The declared floor of `>=3.9` was never accurate: the module
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

[0.3.0]: https://github.com/andremsouza/python-som/releases/tag/v0.3.0
[0.2.0]: https://github.com/andremsouza/python-som/releases/tag/v0.2.0
[0.1.3]: https://pypi.org/project/python-som/0.1.3/
[0.1.2]: https://pypi.org/project/python-som/0.1.2/
