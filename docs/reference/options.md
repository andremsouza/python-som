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

Each enum member **is** a `str`, so `TrainingMode.BATCH == "batch"` is `True`, it hashes the same,
it serialises to `"batch"` in JSON, and it works as a dictionary key. Nothing that accepted a string
before stops accepting one.

## Why bother

A type checker cannot tell `"batch"` from `"bacth"`, so the misspelling survives until the
`ValueError` at runtime, after however long it took to get there. The parameters are typed as the
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

## Both spellings are permanent

Neither form is deprecated and neither is going away.

0.5.0 briefly made plain strings emit `DeprecationWarning`, announcing removal in 1.0.0. **0.6.0
withdrew that.** If you migrated in between, your code still works and nothing is wasted beyond the
time; the enums are staying too.

The reason for the reversal is that every comparable library passes options as strings:

| library | how options are passed |
| --- | --- |
| scikit-learn | `KMeans(init="k-means++", algorithm="lloyd")` |
| numpy | `np.pad(mode="constant")`, `np.linalg.norm(ord="fro")` |
| scipy | `linkage(method="single")`, `minimize(method="BFGS")` |
| minisom | `neighborhood_function="gaussian"`, `topology="rectangular"` |
| sompy | `normalization="var"`, `neighborhood="gaussian"` |

None of them export enums at all. Removing the string form would have made this the only library in
its ecosystem to reject `mode="batch"`, a worse trade than any tidiness it bought.

The argument originally made for enums was that a type checker catches typos. That benefit is real
and already delivered by the `Literal` unions above, without removing anything: `mode="bacth"` is a
type error today while `mode="batch"` is not.

## Learning rate

`learning_rate` is validated from 0.4.0, having been unchecked before.

- **Rejected**: zero, negative, `nan`, `inf`. A rate of `0` freezes every model so that training
  completes and changes nothing; `-1` moves models *away* from the samples they match, taking the
  quantization error from 0.0 to 11.7 in one run. Both used to be accepted in silence.
- **Warned about**: anything above 1. Eq. (3) moves a model a fraction `alpha * h` of the way to the
  sample, so above 1 it overshoots and oscillates. It does not necessarily diverge: at `alpha = 5`
  with decay disabled the largest weight stayed at 3.61, because the neighborhood damps the
  correction away from the winner. Kohonen gives no upper bound, so it is a warning rather than
  an error.
