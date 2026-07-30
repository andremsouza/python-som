# Why isotropy matters

A neighborhood function has to depend on *one* number: the distance between two nodes on the grid.
Not on the horizontal offset and the vertical offset separately. This page is about why that
distinction is load-bearing rather than pedantic. Getting it wrong produces a function that looks
plausible and inverts the behaviour the mexican hat exists to provide.

## What the sources require

Kohonen (2013) Eq. (5) writes the neighborhood as a function of `sqdist(c, i)`, "the square of the
geometric distance between the nodes c and i in the grid". One scalar.

Vrieze (1995) is more explicit still. Figure 3, *"The 'Mexican-hat' function of lateral
interaction"*, plots interaction against an abscissa labelled **"Lateral distance"**. The text:

> When a neuron is firing, near by situated neurons are stimulated, diminishing with increasing
> distance to the firing neuron. From a certain distance on inhibition will take place that
> gradually vanishes when the distance becomes very large.

The coefficient is written $h_{i i_c} = 1/\lVert i_c - i \rVert$, the neighborhood as
$N_i = \{i' \mid d(i,i') \le \rho\}$, and Vrieze notes that both spaces are assumed to be metric.
Every formulation in both papers reduces the two axis offsets to a single distance *first*, then
applies the profile.

## The mistake that looks right

The gaussian is **multiplicatively separable**:

$$e^{-(dx^2 + dy^2)/2\sigma^2} = e^{-dx^2/2\sigma^2} \cdot e^{-dy^2/2\sigma^2}$$

So you can compute it as an outer product of two 1-D gaussians and get exactly the isotropic 2-D
gaussian. That works, it is a normal optimization, and it tempts you into thinking the same trick
generalizes.

It does not. Separability is a property of the exponential, not of neighborhood functions. Apply the
same construction to the Ricker wavelet and you get something else entirely:

```python
ax = (1 - dx**2 / sigma**2) * exp(-(dx**2) / (2 * sigma**2))
ay = (1 - dy**2 / sigma**2) * exp(-(dy**2) / (2 * sigma**2))
h = outer(ax, ay)  # not a mexican hat
```

In the diagonal quadrants both 1-D factors are negative, so their product is **positive**. The
function stops inhibiting exactly where it is supposed to inhibit most.

Measured on a 21×21 grid with $\sigma = 3$, centre at (10, 10):

| quantity | outer product | correct isotropic form |
| --- | --- | --- |
| $h$ at $(c + 2\sigma,\, c + 2\sigma)$ | +0.165 | −0.055 |
| $h$ at $(c + 3\sigma,\, c + 3\sigma)$ | +0.008 | −0.001 |
| global minimum | −0.443 | −0.135 (= $-e^{-2}$, at $r = 2\sigma$) |
| zero-crossing locus | a **cross** ($dx = \pm\sigma \cup dy = \pm\sigma$) | a **circle**, $r = \sqrt{2}\sigma$ |

The separable version puts an excitatory lobe worth 16.5% of the winner's strength where the
function should be pushing models away, and along the diagonals it never becomes negative at all.
Its value is not a function of any metric on the grid, which is precisely the property Eq. (5)
requires.

## How the package enforces it

Every neighborhood function is built from `squared_grid_distance`, which reduces the two offsets to
one number before any profile is applied. The three shipped functions share a single implementation
of each formula, so the per-node form and the batch kernel cannot drift apart.

The test suite asserts the property directly rather than checking golden values: equal grid distance
must give equal $h$. That assertion fails against the separable construction and passes against the
isotropic one.

## The one deliberate exception

The **bubble** is a Chebyshev ball, so it is *not* isotropic under the Euclidean metric, and that is
intentional. Vrieze's own appendix computes `b = MAX(ABS(i - w_i), ABS(j - w_j))`, a square region
rather than a disc. Kohonen's phrasing ("up to a certain radius from the winner") reads as Euclidean,
so the two sources genuinely differ; this package follows Vrieze and says so rather than quietly
picking one.

The consequence is worth stating because it is easy to assume otherwise: on a large enough grid,
nodes at equal Euclidean distance can fall on opposite sides of the boundary. The smallest case is a
radius of $\sqrt{50}$, where $(5, 5)$ lies inside a $\sigma = 5$ square and $(7, 1)$ lies outside.

## Further reading

- [Neighborhood functions](../reference/neighborhood-functions.md) for the formulas and constants.
- [Batch vs stepwise](batch-vs-stepwise.md) for why a signed neighborhood cannot be used with batch
  training at all.
