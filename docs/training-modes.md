# Training modes

Three modes, selected with `mode=` on [`SOM.train`][python_som.SOM.train].

## Batch (recommended)

```python
som.train(data, n_iteration=100, mode="batch")
```

Every model is recomputed at once from the data in its neighborhood, then all are replaced together.
Kohonen (2013) Section 3.1 is unambiguous about preferring it:

> It should be emphasized that only the batch-learning version of the SOM is recommendable for
> practical applications, because it does not involve any learning-rate parameter, and its
> convergence is an order of magnitude faster and safer.

Eq. (8) gives the update:

$$m_i^* = \frac{\sum_j n_j h_{ji} \bar{x}_{m,j}}{\sum_j n_j h_{ji}}$$

where $n_j$ is the number of samples whose best match is node $j$ and $\bar{x}_{m,j}$ is their mean.
`learning_rate` is ignored. Because each iteration touches the whole dataset, far fewer are needed:
the default is 10 per sample against 1000 for the stepwise modes.

!!! warning "Batch mode rejects signed neighborhood functions"

    The denominator $\sum_j n_j h_{ji}$ is only sign-definite when $h \ge 0$. With the mexican hat it
    can vanish or go negative, which inverts or explodes the update, so that combination raises a
    `ValueError`. This is a property of the weighted mean itself, not of this implementation.

### Two departures from a naive transcription

**Models with no data in their neighborhood keep their previous value.** Building the new weights
from a zeroed array instead destroys them. On a 30×30 map with 20 samples and a small radius, that
wiped 282 of 900 models in a single step.

**The per-node sums are contracted with NumPy rather than looped over in Python.** The neighborhood
is evaluated once per node and contracted against the per-node sums and counts. On a 20×20 map with
150 samples this runs about 30× faster than the nested Python loop it replaces, and the two agree
to $10^{-12}$.

The full $(x, y, x, y)$ tensor would be faster still, but it costs $(xy)^2$ floats, roughly 800 MB
for a 100×100 map, so it is not materialised.

## Random

```python
som.train(data, n_iteration=10_000, mode="random")
```

Draws samples at random with replacement and applies Eq. (3) one at a time:

$$m_i(t+1) = m_i(t) + h_{ci}(t)\,[x(t) - m_i(t)]$$

Uses `learning_rate` and `learning_rate_decay`. The original formulation, kept because it is the
one most SOM literature describes and because it accepts signed neighborhood functions.

## Sequential

```python
som.train(data, n_iteration=10_000, mode="sequential")
```

The same stepwise update, but walking the dataset in order and wrapping around until `n_iteration`
steps have run.

## Choosing the number of iterations

Leave `n_iteration` unset and the default is 10 per sample for batch, 1000 per sample for the
stepwise modes. Kohonen Section 4.1 on the radius schedule:

> The true mathematical form of $\sigma(t)$ is not crucial, as long as its value is fairly large in
> the beginning of the process, say, on the order of half of the diameter of the grid, whereafter it
> is gradually reduced to a fraction of it in about 1000 steps.

So a sensible starting point is `neighborhood_radius` at roughly half the shorter side of the grid,
decaying from there, with the floor described in
[Neighborhood functions](neighborhood-functions.md#the-radius-floor) keeping it away from zero.

## Reproducibility

Pass `random_seed` and the run is deterministic. The generator is per-instance, so constructing a
SOM has no effect on NumPy's global random state.

!!! warning "Seeds do not carry across the 0.3.0 boundary"

    Versions before 0.3.0 called `np.random.seed` on the global generator. 0.3.0 uses a private
    `np.random.Generator`, so the same seed produces a different map. To reproduce figures made with
    an older version, pin it: `pip install python-som==0.2.0`.
