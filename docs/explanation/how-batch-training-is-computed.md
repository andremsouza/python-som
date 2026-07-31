# How batch training is computed

Kohonen's Eq. (8) says what a batch update is. It does not say how to evaluate it, and the
difference is a factor of thirty.

$$m_i^* = \frac{\sum_j n_j h_{ji} \bar{x}_{m,j}}{\sum_j n_j h_{ji}}$$

Read literally, that is a sum over every pair of nodes, evaluated once per node. On a 60x60 map
over 30 iterations, the literal reading is 108,000 evaluations of the neighborhood. This package
does it in two matrix products.

## The sum is a convolution

$h_{ji}$ depends only on the offset between nodes $j$ and $i$, never on where either sits. That is
what makes the map translation-invariant, and it means the numerator is a convolution of the
per-node sums with the neighborhood, and the denominator a convolution of the per-node counts.

A convolution can be evaluated many ways. The one that wins here depends on a second property.

## Both batch neighborhoods are separable

Batch training admits the gaussian and the bubble, and each factors into a product of per-axis
terms:

$$e^{-(dx^2 + dy^2) / 2\sigma^2} = e^{-dx^2 / 2\sigma^2} \cdot e^{-dy^2 / 2\sigma^2}$$

$$\max(|dx|, |dy|) \le r \iff (|dx| \le r) \land (|dy| \le r)$$

The first is a property of the exponential. The second is a property of the Chebyshev metric, which
is the metric this package's bubble uses; a Euclidean disc would not factor. Neither is a property
of neighborhood functions in general, and the mexican hat has no such factorisation, which is one
of two reasons batch training rejects it.

Given the factors as matrices $H^x_{ac} = f(a-c)$ and $H^y_{bd} = g(b-d)$, the whole update is:

```python
numerator = np.einsum("ac,bd,cdf->abf", hx, hy, sums, optimize=True)
denominator = np.einsum("ac,bd,cd->ab", hx, hy, counts, optimize=True)
```

$H^x$ is $X \times X$ and $H^y$ is $Y \times Y$, so the memory is $X^2 + Y^2$ floats: 58 KB on a
60x60 map, against the 104 MB a full node-by-node matrix would need and the 800 MB it would need at
100x100.

Measured against evaluating the neighborhood per node:

| map | per node | axis matrices | |
| --- | --- | --- | --- |
| 20x20 | 1.79 ms | 0.054 ms | 33x |
| 40x40 | 14.64 ms | 0.093 ms | 158x |
| 60x60 | 65.11 ms | 0.135 ms | 482x |

## This is not the separability mistake

The distinction matters, because the two look identical from a distance and this package shipped
the wrong one once.

An **axis profile** is a way of evaluating a neighborhood that is already defined as a function of
$\mathrm{sqdist}$. The definition does not change; only the order of the arithmetic does, and a test
asserts the outer product of the two factors equals the isotropic function node by node.

A **separably defined neighborhood** is a different function. Building a mexican hat as an outer
product of two one-dimensional Ricker wavelets gives $+0.165$ on the diagonal at $2\sigma$ where the
correct value is $-0.055$: an excitatory lobe exactly where the function must inhibit. That was a
real defect here, and [Why isotropy matters](why-isotropy-matters.md) covers it.

The guard is a registry. A neighborhood has an axis profile only where the factorisation is an
identity, and a test asserts the registry holds exactly the unsigned neighborhoods. A future
neighborhood that is unsigned but not separable fails that test rather than being approximated.

## Finding every winner at once

The update is now a small part of the cost. The larger part is Eq. (4), the search for each sample's
best-matching model:

$$c = \arg\min_i \lVert x - m_i \rVert$$

Expanding the norm gives $\lVert x \rVert^2 - 2\,x \cdot w + \lVert w \rVert^2$, and the first term
is the same for every model, so it cannot change which one wins. What remains is a matrix product
against all the models at once, plus a per-node constant.

**This is not Kohonen's dot-product map.** Section 4.5 defines a genuinely different algorithm,
$c = \arg\max_i \mathrm{dot}(x, m_i)$, which requires the models to be renormalized to constant
length after every cycle and picks a different node when they are not. The expansion above is exact
for the Euclidean distance and needs no normalization.

### The expansion cancels, and the fix is one line

$\lVert w \rVert^2$ grows with the square of the data's distance from the origin, while the
differences between models do not. With models offset by $10^9$, that term is around $10^{18}$ and
the subtraction loses every significant digit:

| offset | samples given the wrong node |
| --- | --- |
| origin, 1e3, 1e6 | 0 of 500 |
| **1e9** | **500 of 500** |
| **1e12** | **500 of 500** |

Subtracting a common shift from both sides is exact in $\lVert x - w \rVert$, costs 1%, and removes
it at every offset tested. Data far from the origin is not exotic: timestamps, easting and northing
coordinates and absolute sensor readings all look like this. It is the same failure mode that
[linear initialization](why-linear-initialization-is-an-svd.md) had before 0.4.0.

A custom `distance_function` keeps the exact per-sample loop, because the expansion is an identity
for the Euclidean norm and nothing else.

### Small blocks beat large ones

The search runs in blocks so the score matrix never grows with the dataset. The block size is
tuned rather than chosen, on a 60x60 map with 2000 samples:

| budget | time | peak |
| --- | --- | --- |
| **512 KB** | **7.62 ms** | **1.07 MB** |
| 2 MB | 7.50 ms | 2.57 MB |
| 8 MB | 11.14 ms | 8.56 MB |

A block that fits in cache is read back by `argmin` for free. One that does not is read back from
memory, which is why the largest budget is both the slowest and the heaviest.

## What Kohonen says about all this

Reorganising the arithmetic is not a departure from the paper. Section 4.4 derives Eq. (8) from
Eq. (7) on exactly these grounds, that "the same addends occur a great number of times", and
Section 5.2 notes that Eq. (8) "allows for a very efficient implementation" and that "the winner
search can be partly parallelized by dividing the data".

One requirement does constrain the implementation. Section 4.4 closes: the old values "are replaced
by the respective means, **in one concurrent computing operation over all nodes of the grid**".
Every node must be computed from the models as they stood at the start of the iteration. The
contraction satisfies this structurally, since there is no loop to get wrong, and a test asserts it
directly.

Two further optimizations the paper suggests are **not** implemented here. Section 5.2 proposes
confining the winner search to the neighborhood of the previous winner, which is an approximation
that can miss a better match, and reducing the models to eight-bit precision, which changes results
by far more than round-off. Both are recorded rather than adopted.

## What this cost

Trained weights differ from 0.6.1 by about $10^{-15}$ relative. The contraction sums the same terms
in a different order, so the results are not bit-identical to earlier versions, and
[Reproduce a result](../how-to/reproduce-a-result.md) says which version to pin to reproduce an
older figure exactly.
