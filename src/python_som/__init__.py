"""Python implementation of Kohonen's 2-D self-organizing map.

The public surface is :class:`SOM`, plus the decay, distance and neighborhood functions that can be
passed to it or used directly.

>>> import numpy as np, python_som
>>> data = np.random.default_rng(0).normal(size=(150, 4))
>>> som = python_som.SOM(x=10, y=10, input_len=4, random_seed=0)
>>> som.weight_initialization(mode="linear", data=data)
>>> error = som.train(data, n_iteration=100, mode="batch")

Internally the package is a pure functional core with a thin shell around it:
:mod:`python_som._core` holds every numeric decision as functions over NumPy arrays, and imports
nothing but NumPy. :mod:`python_som._convert` adapts pandas and anything else array-like at the
boundary, and :mod:`python_som._som` holds the state and the training loops.

Reference:
Teuvo Kohonen, Essentials of the self-organizing map, Neural Networks 37 (2013) 52-65,
ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2012.09.018
"""

from __future__ import annotations

from ._artifact import ArtifactError, SOMConfig, TrainingReport
from ._core._decay import (
    asymptotic_decay,
    exponential_decay,
    inverse_decay,
    linear_decay,
)
from ._core._distance import euclidean_distance
from ._core._neighborhood import (
    NEIGHBORHOOD_FUNCTIONS,
    SIGNED_NEIGHBORHOODS,
    bubble,
    gaussian,
    mexican_hat,
)
from ._core._protocols import (
    DecayFunction,
    DistanceFunction,
    KernelFunction,
    NeighborhoodFunction,
)
from ._enums import (
    Neighborhood,
    NeighborhoodStr,
    SampleMode,
    SampleModeStr,
    TrainingMode,
    TrainingModeStr,
    WeightInit,
    WeightInitStr,
)
from ._som import SOM

# `as` rather than a plain import: the explicit re-export idiom, which tells a type checker
# this is public without putting a dunder into __all__, where the public API lives.
from ._version import __version__ as __version__

__all__ = [
    "NEIGHBORHOOD_FUNCTIONS",
    "SIGNED_NEIGHBORHOODS",
    "SOM",
    "ArtifactError",
    "DecayFunction",
    "DistanceFunction",
    "KernelFunction",
    "Neighborhood",
    "NeighborhoodFunction",
    "NeighborhoodStr",
    "SOMConfig",
    "SampleMode",
    "SampleModeStr",
    "TrainingMode",
    "TrainingModeStr",
    "TrainingReport",
    "WeightInit",
    "WeightInitStr",
    "asymptotic_decay",
    "bubble",
    "euclidean_distance",
    "exponential_decay",
    "gaussian",
    "inverse_decay",
    "linear_decay",
    "mexican_hat",
]


# Backwards-compatible aliases. Before 0.3.0 these functions lived in the top-level module under
# underscore-prefixed names; they were private by convention but reachable, and the README listed
# them. Keep them working rather than breaking imports silently.
_asymptotic_decay = asymptotic_decay
_linear_decay = linear_decay
_exponential_decay = exponential_decay
_inverse_decay = inverse_decay
_euclidean_distance = euclidean_distance
