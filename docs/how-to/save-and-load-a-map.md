# Saving and loading a map

```python
som.save_npz("iris-map.npz")

som = python_som.SOM.load_npz("iris-map.npz")
som.train(more_data, n_iteration=50)  # continues where it left off
```

One file holds the models and everything needed to describe how they were produced. No `pickle` is
involved on either side, so loading one cannot execute code.

## What is in the file

An `.npz` with two members:

| member | contents |
| --- | --- |
| `weights` | the models, `float64`, shape `(x, y, n_features)` |
| `metadata` | a JSON string |

The metadata is deliberately plain text, so an artifact can be inspected without this package
installed:

```python
import json, numpy as np

with np.load("iris-map.npz", allow_pickle=False) as archive:
    print(json.loads(str(archive["metadata"]))["report"])
```

```json
{
  "format_version": 1,
  "python_som_version": "0.7.0",
  "numpy_version": "2.5.1",
  "config": {
    "shape": [10, 10], "input_len": 4,
    "learning_rate": 0.5, "neighborhood_radius": 1.0, "min_neighborhood_radius": 0.5,
    "cyclic": [false, false],
    "neighborhood_function": "gaussian",
    "learning_rate_decay": "asymptotic_decay",
    "neighborhood_radius_decay": "asymptotic_decay",
    "distance_function": "euclidean_distance"
  },
  "rng": { "seed": 42, "state": { "bit_generator": "PCG64", "state": {"state": 1, "inc": 2} } },
  "report": {
    "mode": "batch", "n_iteration": 100, "n_samples": 150, "random_seed": 42,
    "final_learning_rate": null, "final_neighborhood_radius": 0.5,
    "quantization_error": 0.3142,
    "python_som_version": "0.7.0", "numpy_version": "2.5.1", "wall_time_seconds": 0.42
  }
}
```

`final_learning_rate` is `null` for batch training. Eq. (8) is a weighted mean, so there is no step
size to report; recording the unused initial value would read as though it had been applied.

## Resuming training

A reloaded map continues the *same* random stream, so training it further gives what never stopping
would have given. That is why both the seed and the generator's current state are saved: the seed
alone would restart the stream, and a resumed run would quietly diverge from an uninterrupted one.

```python
# these two end up with identical weights
a = fit(data, iterations=200)

b = fit(data, iterations=100)
b.save_npz("checkpoint.npz")
python_som.SOM.load_npz("checkpoint.npz").train(data, n_iteration=100)
```

## The training report

`som.last_report` describes the most recent `train()` call, and `None` before the first one. It is
saved with the map, so a figure can be traced back to the run that produced it: the seed, the
iteration count, the error, and the versions of this package and NumPy that computed it.

`train()` still returns the quantization error, so nothing that used the return value changes.

## Custom functions

A map's neighborhood, decays and distance are *callables*, and a callable cannot go into a file
without `pickle`. What gets saved is the **name**, resolved on load through the registries.

A map built from the shipped functions round-trips completely. One built with your own function
records the name for provenance and refuses to load silently:

```python
som = python_som.SOM(x=10, y=10, input_len=4, distance_function=cosine)
som.save_npz("custom.npz")

python_som.SOM.load_npz("custom.npz")
# ArtifactError: this map was saved with custom function(s) that cannot be restored from a name
# (distance_function='cosine'). ... Pass them back explicitly:
# load_npz(path, distance_function=...)

python_som.SOM.load_npz("custom.npz", distance_function=cosine)  # works
```

The same applies to a `functools.partial`: it has no name to record, so it is treated as custom.
That is the correct outcome. Resolving `partial(exponential_decay, factor=3.0)` by the wrapped
function's name would silently restore a different decay from the one that trained the map.

The names that *do* resolve are the keys of `NEIGHBORHOOD_FUNCTIONS`, `DECAY_FUNCTIONS` and
`DISTANCE_FUNCTIONS`, each of which is simply the function's own name.

## Is it safe to load a file from someone else

Safer than a pickle, and categorically so: an artifact cannot execute code. It can still be
malformed or oversized. [Artifact safety](../explanation/artifact-safety.md) sets out exactly what is
and is not guaranteed, and why the guarantee stops where it does.

## Version differences

Loading an artifact written by a different **major** version is allowed and warns. A major version
is where this package permits numerical results to change, so a map trained under one and reloaded
under another may not reproduce the run its own report describes. Refusing outright would make old
artifacts unreadable, which is the opposite of what provenance is for.

An artifact written in a newer *format* version is refused, naming the version that wrote it, rather
than being half-understood.
