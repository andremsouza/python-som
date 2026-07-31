# Use a custom strategy

Three things can be replaced with your own implementation: the neighborhood, the decay applied to
the learning rate and radius, and the distance.

Three things can be replaced with your own implementation: the neighborhood, the decay applied to
the learning rate and radius, and the distance. Each has a `Protocol` describing what it must accept
and return, so a type checker verifies yours against a named contract rather than a bare `Callable`.

```python
from python_som import DecayFunction


def linear_to_zero(value: float, step: int, total: int) -> float:
    return value * (1.0 - step / total)


som = python_som.SOM(x=10, y=10, input_len=4, learning_rate_decay=linear_to_zero)
```

The protocols are **structural**: nothing needs to inherit from them, and every callable that
already worked still does. Their parameters are positional-only, so your parameter names are your
own. `linear_to_zero(value, step, total)` and `linear_to_zero(a, b, c)` both satisfy
`DecayFunction`.

One contract is worth reading before writing a neighborhood of your own:
[`NeighborhoodFunction`][python_som.NeighborhoodFunction] must be a function of the grid distance
between two nodes, not of the two axis offsets separately. See
[Why isotropy matters](../explanation/why-isotropy-matters.md) for what goes wrong otherwise.

## What a custom strategy costs you

A callable cannot be written to a file without `pickle`, so `save_npz` records only its **name**. A
map trained with your own function will not reload on its own.

```python
python_som.SOM.load_npz("custom.npz")
# ArtifactError: ... Pass them back explicitly: load_npz(path, distance_function=...)

python_som.SOM.load_npz("custom.npz", distance_function=my_distance)  # works
```

That is deliberate. The alternative is guessing, and a map that silently reloads with a different
distance from the one that trained it is worse than one that refuses.

If you want a strategy that round-trips without the extra argument, the ones this package ships all
do: their names are the keys of `NEIGHBORHOOD_FUNCTIONS`, `DECAY_FUNCTIONS` and
`DISTANCE_FUNCTIONS`.

## Before writing a neighborhood

Read [Why isotropy matters](../explanation/why-isotropy-matters.md) first. A neighborhood must be a
function of the grid distance between two nodes, not of the two axis offsets separately, and the
construction that tempts you into getting this wrong is the same one that works correctly for the
gaussian.
