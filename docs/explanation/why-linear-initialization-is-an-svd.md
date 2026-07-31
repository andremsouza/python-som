# Why linear initialization is an SVD

Kohonen recommends starting the models on the plane of the data's two largest principal components
rather than at random, because "much faster convergence follows" (Section 4.3). Computing those
components is the only linear algebra this package needs, and how it is computed turned out to
matter more than expected.

Through 0.3.0 it was `sklearn.decomposition.PCA`. Since 0.4.0 it is about twenty lines of
`np.linalg.svd`. The change removed a dependency and, unexpectedly, fixed a real accuracy defect.

## Two ways to find the same components

For centred data $X$, the principal components are the eigenvectors of the covariance matrix
$X^\top X / (n-1)$. There are two ways to get them.

**Eigendecompose the covariance matrix.** Form $X^\top X$, then decompose it. Cheap when there are
far more samples than features.

**Decompose the data directly.** For $X = U S V^\top$, the rows of $V^\top$ are the components and
$S^2/(n-1)$ the variance along each. No covariance matrix is ever formed.

They agree in exact arithmetic. They do not agree in floating point, because forming $X^\top X$
**squares the condition number**. Every digit of precision in the data becomes half a digit in the
result, and when the mean is large relative to the spread there are not many digits to start with.

## The defect this exposed

Linear initialization fits its PCA on **raw** data by design, so the models live in the same space
as the inputs they will be compared against. Since scikit-learn 1.5 the default solver picks
`covariance_eigh` when samples comfortably outnumber features, which is exactly the squaring path.

On `(150, 4)` data offset by $10^7$, the second explained variance was wrong by **5.8%**, and the
models it produced differed from the correct ones by 2.43 against a total model spread of 2.0. The
error was larger than the structure being initialized.

Measured against a reference centred in `longdouble` before decomposing:

| solver | relative error |
| --- | --- |
| scikit-learn `auto` (covariance path) | 1.4e-06 to 5.5e-06 |
| scikit-learn `svd_solver="full"` | ~1e-15 |
| this package | ~1e-15 |

Data offset far from the origin is not a corner case. Timestamps, easting and northing coordinates
and absolute sensor readings all look like this, and none of them announce themselves.

The same failure mode appears again in the best-matching-unit search, where expanding
$\lVert x - w \rVert^2$ squares the magnitudes in the same way.
[How batch training is computed](how-batch-training-is-computed.md) covers it and the one-line fix.

## Why the reimplementation is trusted

Replacing a widely-used library's numerics with twenty lines of your own is the change in this
package a reviewer should be least willing to take on faith, so it is not asked for on faith.

scikit-learn remains a **test** dependency, and `tests/test_linalg_matches_sklearn.py` re-derives
every fit both ways on every CI run and compares them. The claim under test is not "close enough"
but "the same numbers": the tolerances are at the scale of double-precision round-off.

The comparison is against `svd_solver="full"`, not against the default. That is deliberate, and the
first version of the test got it wrong: it compared against `auto`, failed, and the *reference* was
what was inaccurate. There is also a check against a `longdouble` reference, which depends on no
library's solver choice and would survive scikit-learn changing its defaults again.

## Two details that are easy to get wrong

**The sign convention is v-based.** An SVD fixes each component only up to sign, so a convention is
needed for a fit to be reproducible. scikit-learn's PCA calls `svd_flip` with
`u_based_decision=False`, orienting each component so its largest-magnitude loading is positive.
That is the less common of the two settings in that helper. Taking the default would still give a
valid PCA, but a different one, and linear initialization would lay its models out reversed along
that axis. No test of orthonormality or explained variance would notice; only comparing signs does.

**A near-constant column is scaled by 1.** Dividing a column by its own standard deviation is the
obvious z-score and the wrong one when that deviation is zero. The guard is not `variance == 0`
either: a column built by arithmetic that should cancel exactly can retain a variance of about
$10^{-30}$, which passes an equality test and then divides the column by roughly $10^{-15}$. The
bound used is scikit-learn's, from Chan, Golub and LeVeque.

## What this costs

The reimplementation removed 264 MB of required install, 79% of the payload, taking python-som from
10 packages to 1. That was the reason for doing it. The accuracy improvement was a side effect, and
in retrospect the more valuable half.

Because 0.4.0 changed what linear initialization produces for data far from the origin, results are
not comparable with 0.3.0 for those datasets.
[Reproduce a result](../how-to/reproduce-a-result.md) has the version-pinning details.
