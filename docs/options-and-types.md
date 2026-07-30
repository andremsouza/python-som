# Options and types

Every option that is spelled as a string has an enum member too, and the two are interchangeable.

```python
import python_som
from python_som import Neighborhood, TrainingMode, WeightInit

som = python_som.SOM(x=10, y=10, input_len=4, neighborhood_function=Neighborhood.GAUSSIAN)
som.weight_initialization(mode=WeightInit.LINEAR, data=data)
som.train(data, n_iteration=100, mode=TrainingMode.BATCH)
```

is the same call as

```python
som = python_som.SOM(x=10, y=10, input_len=4, neighborhood_function="gaussian")
som.weight_initialization(mode="linear", data=data)
som.train(data, n_iteration=100, mode="batch")
```

Each enum member **is** a `str`, so `TrainingMode.BATCH == "batch"` is `True`, it hashes the same, it
serialises to `"batch"` in JSON, and it works as a dictionary key. Nothing that accepted a string
before stops accepting one.

## Why bother

A type checker cannot tell `"batch"` from `"bacth"`, so the misspelling survives until the
`ValueError` at runtime — after however long it took to get there. The parameters are typed as the
enum *or* the exact set of valid strings, so both spellings pass and a typo does not:

```python
som.train(data, mode="batch")  # fine
som.train(data, mode=TrainingMode.BATCH)  # fine
som.train(data, mode="bacth")  # error: incompatible type "Literal['bacth']"
```

The trade is that a variable of plain `str` type no longer satisfies the parameter. If you build a
mode from configuration, narrow it or annotate it:

```python
from python_som import TrainingModeStr

mode: TrainingModeStr = config["mode"]  # or TrainingMode(config["mode"]) to validate at runtime
som.train(data, mode=mode)
```

`TrainingMode(config["mode"])` raises `ValueError` on an unknown value, which is often what you want
when the value comes from outside the program.

## The options

| Parameter | Enum | Values |
| --- | --- | --- |
| `SOM(neighborhood_function=...)` | `Neighborhood` | `gaussian`, `bubble`, `mexican_hat` |
| `SOM.train(mode=...)` | `TrainingMode` | `random`, `sequential`, `batch` |
| `SOM.weight_initialization(mode=...)` | `WeightInit` | `random`, `linear`, `sample` |
| `weight_initialization(sample_mode=...)` | `SampleMode` | `standard_normal`, `uniform` |

The legacy spelling `"mexicanhat"` still resolves and is still accepted, but has no enum member:
one canonical spelling per option is most of the point of having an enum.

## Deprecation timetable for plain strings

Strings are deprecated as of **0.4.0**, and nothing warns yet.

| Version | What happens |
| --- | --- |
| **0.4.0** | Enums added. Strings accepted, no warning. |
| **0.5.0** | Strings emit `DeprecationWarning`. |
| **1.0.0** | Strings removed; the enums become the only accepted form. |

No warning in 0.4.0 is deliberate. `mode="batch"` is what this documentation showed until this page
existed, so a minor release that warned on it would be scolding people for following its own
instructions. The written notice comes first and the warning follows in 0.5.0, which also means
0.5.0 has to ship before 1.0.0 can remove anything — the policy is that a removal in a major release
is preceded by at least one minor release that warns.

Migrating early costs nothing and is a mechanical substitution: `"batch"` becomes
`TrainingMode.BATCH`, and so on down the table above.

## Custom strategies

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
own — `linear_to_zero(value, step, total)` and `linear_to_zero(a, b, c)` both satisfy
`DecayFunction`.

One contract is worth reading before writing a neighborhood of your own:
[`NeighborhoodFunction`][python_som.NeighborhoodFunction] must be a function of the grid distance
between two nodes, not of the two axis offsets separately. See
[Neighborhood functions](neighborhood-functions.md) for what goes wrong otherwise.

## Learning rate

`learning_rate` is validated from 0.4.0, having been unchecked before.

- **Rejected**: zero, negative, `nan`, `inf`. A rate of `0` freezes every model so that training
  completes and changes nothing; `-1` moves models *away* from the samples they match, taking the
  quantization error from 0.0 to 11.7 in one run. Both used to be accepted in silence.
- **Warned about**: anything above 1. Eq. (3) moves a model a fraction `alpha * h` of the way to the
  sample, so above 1 it overshoots and oscillates. It does not necessarily diverge — at `alpha = 5`
  with decay disabled the largest weight stayed at 3.61, because the neighborhood damps the
  correction away from the winner — and Kohonen gives no upper bound, so it is a warning rather than
  an error.
