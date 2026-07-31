# How this compares to MiniSom and SOMPY

Three Python packages implement Kohonen's self-organizing map. This page measures python-som against
the other two, and spends most of its length on why that measurement is harder than it looks.

The short version: on batch training python-som is **1.3x to 2.0x faster than SOMPY** and **1.3x to
1.6x slower than MiniSom**, with the gap against MiniSom growing as the map grows. On stepwise
training python-som and MiniSom are within about 10% of each other. SOMPY has no stepwise mode.

## Why a naive comparison is wrong

The three packages look interchangeable. python-som's method names deliberately mirror MiniSom's,
so `activate`, `winner`, `quantization`, `quantization_error` and `get_weights` all line up.
Underneath, some of them do different things.

**MiniSom's `train_batch` is not batch training.** It is stepwise training in sequential sample
order, the counterpart of `mode="sequential"` here. Kohonen's Eq. (8) lives in
`train_batch_offline`.
Reaching for the name that matches would compare a weighted mean against per-sample gradient steps
and produce a number that looks plausible.

**Only the gaussian is the same function in all three.** MiniSom writes it as an outer product of
two one-dimensional gaussians, which is valid because the exponential is separable; SOMPY applies
it to an already-squared grid distance. They agree to one unit in the last place. The other two
neighborhoods do not agree at all:

| | python-som | MiniSom 2.3.6 | SOMPY @6aca604 |
| --- | --- | --- | --- |
| gaussian | $e^{-r^2/2\sigma^2}$ | separable product, same function | same function |
| mexican hat | $(1-u)e^{-u}$, zero at $\sqrt{2}\sigma$ | $(1-2u)e^{-u}$, zero at $\sigma$ | not implemented |
| bubble | Chebyshev, $\max(\|dx\|,\|dy\|) \le \mathrm{round}(\sigma)$ | Chebyshev, strict, unrounded | Euclidean disc |

At $\sigma = 1$ the bubble selects 9 nodes here and 1 node in MiniSom. That is not a rounding
difference, and a benchmark using the bubble at equal $\sigma$ would compare a neighbourhood against
a point update. So every timing below uses the gaussian.

None of these are defects. They are convention choices, and each package is internally consistent.
Worth stating plainly in one direction: **MiniSom's mexican hat is isotropic**, so it never had the
separability bug that python-som shipped in its first contributed version and fixed in 0.2.0. See
[Why isotropy matters](why-isotropy-matters.md) for what that bug was.

## The controls

With the differences known, the libraries can be driven into computing the same thing. Every control
below is asserted before any timing is reported, and the MiniSom ones are also checked on every CI
run by `tests/test_minisom_agreement.py`.

1. **The same initial models are injected into all arms.** Seeding cannot do this: the three use
   different generators, and their PCA initializations are three different functions.
2. **The gaussian only.**
3. **The same radius schedule.** `asymptotic_decay` is character-for-character identical in
   python-som and MiniSom. SOMPY can only run `linspace(start, end, n)`, so python-som matches that
   instead.
4. **Cases where python-som's radius floor never engages**, since neither peer has one.
5. **MiniSom's batch learning rate pinned to a constant 1.0.** `train_batch_offline` relaxes toward
   the Eq. (8) mean by a decaying $\eta$ where Eq. (8) assigns it outright.
6. **Sequential rather than random sample order**, because `arange(T) % n` and
   `np.resize(arange(n), T)` are the same sequence. The random modes differ genuinely and no seed
   reconciles them.
7. **SOMPY's normalization turned off.** At its `'var'` default it would train on z-scored data.
8. **The quantization error recomputed under one definition for every arm.** This is not optional
   for SOMPY: its `calculate_quantization_error` returns `mean(|x - m|)` over every feature, an
   elementwise mean absolute error, where the standard definition is the mean Euclidean distance per
   sample. SOMPY disagrees with itself here, since the value it logs during training *is* an L2
   distance.

With those in place the trained models agree, which is what makes the timings mean anything:

| against | agreement | why not exact |
| --- | --- | --- |
| MiniSom, stepwise | 2.8e-16 relative | floating-point association order |
| MiniSom, batch | 1.3e-15 relative | the two accumulate Eq. (8) in a different order |
| SOMPY, batch | 1.4e-07 relative | SOMPY rounds its codebook to six decimals every iteration |

## The numbers

Medians of 9 interleaved repeats. python-som 0.6.1, MiniSom 2.3.6 on NumPy 2.5.1, SOMPY @6aca604 on
NumPy 1.26.4, CPython 3.12.13, Linux, Intel Core Ultra 9 275HX.

### Against MiniSom

| map | samples | features | mode | python-som | MiniSom | result |
| --- | --- | --- | --- | --- | --- | --- |
| 20x20 | 200 | 4 | batch | 232.5 ms | 239.2 ms | 1.03x faster |
| 40x40 | 300 | 6 | batch | 1016.6 ms | 758.7 ms | **1.34x slower** |
| 60x60 | 400 | 8 | batch | 2885.2 ms | 1763.0 ms | **1.64x slower** |
| 20x20 | 200 | 4 | sequential | 1.2 ms | 1.4 ms | 1.12x faster |
| 40x40 | 300 | 6 | sequential | 2.4 ms | 2.5 ms | 1.06x faster |
| 60x60 | 400 | 8 | sequential | 4.4 ms | 4.7 ms | 1.08x faster |

**python-som's batch training is slower, and the reason is structural rather than incidental.**
`batch_update` walks every node in Python: on a 60x60 map over 30 iterations that is 108,000
`einsum` calls. MiniSom's node-side update is a single vectorised divide, and its Python loop runs
over the 400 samples instead. So the ratio tracks nodes against samples, which is why the gap is
absent at 20x20 and 1.64x at 60x60. Vectorising the node loop is the obvious fix, and has not been
done.

### Against SOMPY

| map | samples | features | python-som | SOMPY | result |
| --- | --- | --- | --- | --- | --- |
| 20x20 | 200 | 4 | 145.4 ms | 188.3 ms | 1.30x faster |
| 40x40 | 300 | 6 | 727.7 ms | 1264.6 ms | 1.74x faster |
| 60x60 | 400 | 8 | 1906.4 ms | 3796.6 ms | 1.99x faster |

Batch only, since SOMPY implements nothing else: `SOMFactory.build` accepts `training='seq'` and
ignores it.

Cases stop at 60x60 because SOMPY materialises the full node-by-node neighborhood, a `(3600, 3600)`
matrix at 104 MB. At 100x100 it would be 800 MB. python-som stores one `(2X-1, 2Y-1)` kernel
instead, 198 KB at 80x80, so its neighborhood memory grows linearly in node count where SOMPY's
grows quadratically.

### Initialization

The one default the three do not share, isolated by holding the algorithm and every hyperparameter
fixed and letting each library seed its own models. Quantization error after training, on data
scaled by 10 and offset 100 from the origin:

| map | python-som | MiniSom | better |
| --- | --- | --- | --- |
| 20x20 | 1.6797 | 1.1157 | MiniSom |
| 40x40 | 0.2855 | 0.6265 | python-som |
| 60x60 | 0.0937 | 0.5009 | python-som |

The three initializers place models on the plane of the first two principal components and differ in
how far apart. python-som uses $\bar{x} + c_1\sqrt{\lambda_1}v_1 + c_2\sqrt{\lambda_2}v_2$, scaling
each direction by the data's actual extent along it, from an SVD of the centred matrix. MiniSom uses
unit eigenvectors, so the initial sheet is one unit across whatever the data's spread. SOMPY scales
by $\lambda$ rather than $\sqrt{\lambda}$ and uses a randomized PCA, which makes its initialization
non-deterministic.

## Reproducing this

```bash
uv sync --extra dev --extra bench
uv run python benchmarks/bench_vs_minisom.py
```

SOMPY needs an interpreter of its own and you have to build it by hand, because it cannot be
installed alongside python-som at all: it uses `np.Inf` at class-definition time, and that was
removed in NumPy 2.0. `benchmarks/bench_vs_sompy.py` prints this command if the environment is
missing, and installs nothing itself.

```bash
uv venv .venv-sompy --python 3.12
uv pip install --python .venv-sompy \
    "numpy==1.26.4" "scipy==1.13.1" "scikit-learn==1.5.2" "scikit-image==0.24.0" \
    "numexpr==2.10.1" "joblib==1.4.2" "matplotlib==3.9.2" \
    "SOMPY @ git+https://github.com/sevamoo/SOMPY@6aca604b06e5eea1391ecf507810c7aabafc3f8b"
uv run python benchmarks/bench_vs_sompy.py
```

!!! warning "`pip install sompy` gets a different package"

    The SOMPY compared here is [sevamoo/SOMPY](https://github.com/sevamoo/SOMPY), which was never
    published to PyPI. The PyPI project named `sompy` is version 0.1.1 by a different author, two
    releases in 2016, and is unrelated. The pin above is a commit SHA because the repository has no
    tags.

To track python-som against its own history rather than against a peer, which is a different
question and wants a different tool:

```bash
cd asv_benchmarks
uv run --extra bench asv machine --yes
uv run --extra bench asv continuous master HEAD
```

!!! note "`asv machine` records your hardware"

    It writes your host's name, CPU model, core count, RAM and kernel version to
    `.asv/results/<machine>/machine.json`. That directory is gitignored.

## Caveats

Timings come from one machine and are not portable; the ratios are more durable than the
milliseconds, and both are worth re-measuring rather than quoting. Nothing here is a CI gate,
because shared runners cannot measure anything: an earlier version of this comparison swung from
2.56x to 1.01x between two runs on an otherwise idle laptop.

The peers are pinned, and MiniSom's development branch has already moved past its release with a
numba-compiled batch path that would change these numbers substantially.

Both peers are worth using. MiniSom is faster at batch training on larger maps and has hexagonal
topologies, which python-som does not. SOMPY has clustering and visualization helpers built in. What
python-som offers against them is NumPy as its only runtime dependency, a scikit-learn estimator
interface, pickle-free artifacts with provenance, and type information.
