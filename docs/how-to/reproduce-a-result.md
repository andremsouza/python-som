# Reproduce a result

To get the same map twice, or to let someone else get the map you got.

## Pin the seed

```python
som = python_som.SOM(x=10, y=10, input_len=4, random_seed=42)
```

Without `random_seed` the constructor draws one from the OS, so every run differs. The seed it used
either way is available afterwards:

```python
som.get_random_seed()
```

The generator is per-instance. Constructing a map does not disturb NumPy's global random state, so
two maps built in the same program do not interfere with each other and nothing else in your program
is affected by building one.

## Pin the version

Numerical results are allowed to change between minor versions before 1.0.0, and between major
versions after it. A seed alone does not pin a result across an upgrade.

```
python-som==0.7.0
```

Three releases change results. If you are reproducing an older figure:

- **0.3.0** replaced the global RNG with a per-instance generator, so `random_seed=42` gives a
  different map from 0.2.0 and earlier. Pin `python-som==0.2.0` to reproduce those.
- **0.4.0** fixed an accuracy defect in linear initialization for data far from the origin. Near the
  origin the difference is floating-point noise; far from it, it is large, and 0.4.0 is the correct
  one.
- **0.7.0** made batch training 20x to 40x faster by reorganising the same arithmetic, which sums in
  a different order. Weights differ from 0.6.1 by about 1e-15 relative: far below anything a result
  depends on, and still not bit-identical. Pin `python-som==0.6.1` if you need an exact match.

## Record what you ran

`som.last_report` describes the run that just finished:

```python
error = som.train(data, n_iteration=100, mode=TrainingMode.BATCH)
print(som.last_report)
```

```
TrainingReport(mode='batch', n_iteration=100, n_samples=150, random_seed=42,
               final_learning_rate=None, final_neighborhood_radius=0.5,
               quantization_error=0.3142, python_som_version='0.7.0',
               numpy_version='2.5.1', wall_time_seconds=0.42)
```

That is enough to reconstruct the run: the seed, the mode, the iteration count, and the versions of
this package and NumPy that computed it. `wall_time_seconds` is excluded from equality, so two
identical runs produce equal reports.

## Save the map itself

Better than reconstructing a run is not having to:

```python
som.save_npz("figure-3.npz")
```

The file carries the models, the configuration, the seed, the generator state and the report. A
reloaded map also continues the same random stream, so training it further gives what never stopping
would have given.

See [Save and load a map](save-and-load-a-map.md) for the details, including what happens when the
map used a function this package cannot look up by name.

## Check that you succeeded

Reproducibility is a claim worth testing rather than assuming:

```python
first = build_and_train(seed=42)
second = build_and_train(seed=42)
assert np.abs(first.get_weights() - second.get_weights()).max() == 0.0
```

Exact equality, not `allclose`. Two runs with the same seed on the same version perform the same
arithmetic in the same order, so any difference at all means something is not pinned that you think
is: an unpinned dependency, a different BLAS, or data that is not identical.
