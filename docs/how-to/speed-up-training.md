# Speed up training

Four changes, ordered from largest effect to smallest. The first two cost nothing.

## 1. Use batch training

```python
som.train(data, n_iteration=100, mode="batch")
```

It is the default, it is the mode the optimization work targets, and it needs far fewer iterations:
10 per sample against 1000 for the stepwise modes. If you are passing `mode="random"` or
`mode="sequential"` for no particular reason, change it.

See [Batch vs stepwise](../explanation/batch-vs-stepwise.md) for what the modes do differently.

## 2. Keep the default distance function

The winner search has a fast path that applies only to the Euclidean distance. Supplying your own
`distance_function` falls back to one evaluation per sample, several times slower. If you need a
custom distance, see [Use a custom strategy](use-a-custom-strategy.md); if you do not, leave it
alone.

## 3. Install numba

```bash
pip install numba
```

Nothing else to do. The compiled kernel is detected and used automatically, and results are
identical to the NumPy path. Check it took effect:

```python
import python_som

python_som.accelerated()  # True once numba is installed
```

Three things to expect:

- **NumPy is downgraded in that environment.** numba 0.66 requires `numpy<2.5`. Install it into a
  virtual environment you are willing to hold below that.
- **The gain is uneven.** Measured on batch training: 2.4x on a 60x60 map, about 1.0x on a 20x20
  or a 100x100 one. Measure your own case before accepting the NumPy constraint.
- **The first run is slower.** About 80 ms to import numba and 500 ms to compile, once per process.

## 4. Size the map for the data

Cost grows with the node count, faster than linearly. Kohonen's rule of thumb is about 50 samples
per node on average; well past that, you are paying for resolution the data does not support.

Let the constructor pick the ratio from the data rather than choosing both sides yourself:

```python
som = python_som.SOM(x=20, y=None, input_len=4, data=data)
```

The [measured timings](../explanation/comparison-with-som-libraries.md#against-minisom) show what
map size costs across a range of shapes.

## Why these and not others

[How batch training is computed](../explanation/how-batch-training-is-computed.md) covers what the
library does internally, including the approaches that were measured and rejected.
