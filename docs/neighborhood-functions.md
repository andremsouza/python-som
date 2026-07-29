# Neighborhood functions

The neighborhood function decides how a winner's correction spreads to the rest of the grid. It is
the part of the SOM that turns vector quantization into a *topology-preserving* map, and it is where
this library's subtlest bug lived, so it is worth setting out properly.

## The rule that governs all of them

Kohonen (2013) Eq. (5) writes the neighborhood as

$$h_{ci}(t) = \alpha(t) \exp\left[-\frac{\mathrm{sqdist}(c, i)}{2\sigma^2(t)}\right]$$

where $\mathrm{sqdist}(c, i)$ is *"the square of the geometric distance between the nodes $c$ and $i$
in the grid"*.

Vrieze (1995) says the same thing from the biological side. Figure 3 of that chapter, captioned
*"The 'Mexican-hat' function of lateral interaction"*, plots interaction strength against a single
axis labelled **"Lateral distance"**. The update coefficient is written $h_{i i_c} = 1/\|i_c - i\|$,
the neighborhood set as $N_i = \{i' \mid d(i, i') \le \rho\}$, and the text states that the grid is
assumed to be a metric space.

So the rule is: **collapse the two axis offsets into one distance first, then apply the profile.**
A neighborhood function is a function of $r$, not of $(dx, dy)$ separately.

## Gaussian

```python
som = python_som.SOM(x=20, y=20, input_len=4, neighborhood_function="gaussian")
```

$$h(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right)$$

Strictly positive, monotonically decreasing, normalized so $h(0) = 1$. This is the default and the
form Kohonen writes out.

The gaussian has a special property that causes trouble elsewhere: it is **multiplicatively
separable**.

$$\exp\left(-\frac{dx^2 + dy^2}{2\sigma^2}\right) = \exp\left(-\frac{dx^2}{2\sigma^2}\right)\cdot\exp\left(-\frac{dy^2}{2\sigma^2}\right)$$

You can therefore compute it as an outer product of two 1-D gaussians and get exactly the right
answer. That works because of a property of $\exp$, not because of anything about neighborhood
functions.

## Mexican hat

```python
som = python_som.SOM(x=20, y=20, input_len=4, neighborhood_function="mexicanhat")
```

$$h(r) = (1 - u)\,e^{-u}, \qquad u = \frac{r^2}{2\sigma^2}$$

Also called the Ricker wavelet, or the Laplacian of Gaussian. Normalized so $h(0) = 1$. It:

- **crosses zero** at $r = \sqrt{2}\,\sigma$
- reaches its **minimum of $-e^{-2} \approx -0.135$** at $r = 2\sigma$
- decays back to zero beyond that

This is the shape Vrieze describes: *"When a neuron is firing, near by situated neurons are
stimulated, diminishing with increasing distance to the firing neuron. From a certain distance on
inhibition will take place that gradually vanishes when the distance becomes very large."*

!!! warning "It is not an outer product of two 1-D wavelets"

    The Ricker wavelet is **not** separable, so the trick that works for the gaussian fails here.
    The outer product of two 1-D Ricker wavelets is positive in the diagonal quadrants, where both
    1-D factors are negative and their product flips sign.

    Measured on a 21×21 grid with $\sigma = 3$:

    | quantity | outer product | correct isotropic form |
    | --- | --- | --- |
    | $h$ at $(c + 2\sigma,\, c + 2\sigma)$ | **+0.165** | −0.055 |
    | $h$ at $(c + 3\sigma,\, c + 3\sigma)$ | **+0.008** | −0.001 |
    | global minimum | −0.443 | −0.135 |
    | zero-crossing locus | a **cross** | a **circle**, $r = \sqrt{2}\sigma$ |

    The separable version puts an excitatory lobe worth 16.5% of the winner's strength exactly
    where the function is supposed to inhibit, and along the diagonals it never becomes negative at
    all. Its value is not a function of any metric on the grid.

    This is tested directly: `test_mexican_hat_is_not_the_separable_product` and `test_is_isotropic`.

### It cannot be used with batch training

Batch training computes a weighted mean whose denominator is $\sum_j n_j h_{ji}$ (Kohonen Eq. 8).
That is only well defined when $h$ is non-negative. On a 12×12 grid, 49 of 144 denominators come out
negative, which inverts or explodes the update. Passing `mode="batch"` with the mexican hat raises a
`ValueError`. Use `mode="random"` or `mode="sequential"`.

## Bubble

```python
som = python_som.SOM(x=20, y=20, input_len=4, neighborhood_function="bubble")
```

$$h(dx, dy) = \begin{cases} 1 & \max(|dx|, |dy|) \le \rho \\ 0 & \text{otherwise}\end{cases}$$

This is the truncated inner, excitatory lobe of the mexican hat. Vrieze p. 85: *"In Kohonen networks
usually only the inner stimulation area is used"*, and the flat choice is *"just as effective and
sometimes even better"* than a distance-dependent one.

!!! note "The bubble uses the Chebyshev metric, so the region is a square"

    A node is included when $\max(|dx|, |dy|) \le \rho$, which makes the region a square rather than
    a disc. That follows Vrieze's appendix pseudo-code, which computes
    `b = MAX(ABS(i - w_i), ABS(j - w_j))`, although Kohonen's phrase "up to a certain radius from the
    winner" reads as Euclidean. The two sources genuinely differ, and this library keeps Vrieze's
    reading so that existing results stay reproducible.

    A consequence worth knowing, because it is easy to assume otherwise: **a Chebyshev ball is not
    isotropic under the Euclidean metric**. Two nodes the same Euclidean distance from the winner can
    fall on opposite sides of the boundary. The smallest case is $r = \sqrt{50}$, where $(5, 5)$ is
    inside a $\sigma = 5$ square and $(7, 1)$ is outside.

Unlike the other two, a radius of zero is allowed here: it selects the winner alone, which is well
defined for an indicator function, where for the gaussian and the mexican hat it would be a division
by zero.

## Cyclic grids

With `cyclic_x=True` or `cyclic_y=True` the grid becomes a torus and offsets use the minimum-image
convention, $(d + L/2) \bmod L - L/2$, so the shortest way around is used.

Both tails have to be folded. Folding only offsets above $+L/2$, as this library did before 0.2.0,
leaves a winner in the upper half of an axis unable to wrap: on a 10×10 map with $\sigma = 1$, a
winner at row 0 gave $h[9] = 0.6065$ while a winner at row 9 gave $h[0] = 0$.

## The radius floor

Kohonen Section 4.2 warns that *"the final value of $\sigma$ shall not go to zero, because otherwise
the process loses its ordering power. It should always remain, say, above half of the grid
spacing."*

The `min_neighborhood_radius` parameter enforces exactly that, defaulting to 0.5. It matters in
practice because `linear_decay` reaches zero at the horizon, which would otherwise be a division by
zero.
