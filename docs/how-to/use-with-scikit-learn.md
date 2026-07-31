# Use the estimator interface

`SOM` provides `fit`, `transform`, `predict`, `score`, `get_params` and `set_params`, so a map can
be used the way a scikit-learn estimator is used. `train` remains the primary way to train and is
not going away.

```python
som = python_som.SOM(x=10, y=10, input_len=4, random_seed=42)
som.weight_initialization(mode=WeightInit.LINEAR, data=X)

som.fit(X, n_iteration=100, mode=TrainingMode.BATCH)  # returns the map, so calls chain
labels = som.predict(X)  # (n_samples,)  flat node index
distances = som.transform(X)  # (n_samples, x*y)
som.score(X)  # negated quantization error
```

## What maps onto what

Modelled on `KMeans`, which is the right precedent: a SOM is a topologically-constrained k-means.

| `KMeans` | `SOM` | |
| --- | --- | --- |
| `fit(X, y=None)` | same | returns the estimator |
| `transform(X)` → `(n, n_clusters)` | `(n, x*y)` | distance to every node |
| `predict(X)` → `(n,)` | `(n,)` | **flat** node index |
| `score(X)` | same | negated, so larger is better |
| `cluster_centers_` | `weights_` | |
| `inertia_` | `quantization_error_` | `None` before training |
| `n_features_in_` | same | |

## predict returns a flat index

Not `(row, column)`. A 1-D array of labels is what scorers, `confusion_matrix` and `cross_val_score`
all assume, so returning coordinates would read better for a grid and compose with nothing.

```python
rows, columns = np.unravel_index(som.predict(X), som.get_shape())
```

`winner(x)` still returns `(row, column)` for a single sample.

## Two differences from scikit-learn

**`fit` continues rather than resetting.** scikit-learn estimators conventionally discard their
fitted state on a second `fit`. This one does not: a SOM's models *are* its state, and `train` has
always continued from wherever they were. Calling `fit` twice trains twice.

**The models exist before training.** Initialization is a separate step
(`weight_initialization`), so `weights_` is available from construction. There is no
"not fitted" state to raise about.

## set_params, and what it will not change

```python
som.set_params(learning_rate=0.1, neighborhood_radius=3.0)
```

The rates, the radii, the decays and the distance function can be changed. The grid shape and
`input_len` cannot, and asking raises:

```python
som.set_params(x=20)
# ValueError: 'x' cannot be changed after construction: the models would no longer match it.
```

Silently rebuilding would discard trained weights, which is the kind of help nobody asks for.

`set_params` is what `set_learning_rate` and `set_neighborhood_radius` become. Both still work and
are removed in 1.0.0.

## Full scikit-learn integration

These method names alone are **not** enough for `Pipeline.predict`, `GridSearchCV` or
`cross_val_score`. Since scikit-learn 1.7 those paths require `__sklearn_tags__`, which in practice
means inheriting `BaseEstimator`:

| approach | `clone` | `Pipeline.fit` | `Pipeline.predict` | `GridSearchCV` | `cross_val_score` |
| --- | --- | --- | --- | --- | --- |
| these methods alone | ok | ok | fails | fails | fails |
| `python_som.sklearn.SOMEstimator` | ok | ok | ok | ok | ok |

So a proper adapter ships separately, and scikit-learn stays an optional dependency:

```bash
pip install "python-som[sklearn]"
```

```python
from python_som.sklearn import SOMEstimator

GridSearchCV(SOMEstimator(x=10, y=10), {"x": [8, 10, 12]}, cv=3).fit(X)
```

Use the methods on `SOM` when you are calling them yourself; use the adapter when scikit-learn is
doing the calling.
