"""Names for the string-valued options, so a typo is a type error rather than a runtime one.

Every option is also accepted as a plain string, permanently. ``mode=TrainingMode.BATCH`` and
``mode="batch"`` are interchangeable, compare equal, hash equal and serialise identically, because
each member *is* a ``str``. 0.5.0 briefly deprecated the string form and 0.6.0 withdrew that; the
changelog has the reasoning.

The type-checking benefit comes from the ``Literal`` unions below rather than from the enums:
``mode="bacth"`` is a type error while ``mode="batch"`` is not.

**On the base class.** ``enum.StrEnum`` needs Python 3.11 and this package supports 3.10, so
:class:`_StrEnum` reproduces it. A bare ``class X(str, Enum)`` is not equivalent: its ``str()``
returns ``'X.MEMBER'`` rather than the value, which would put the wrong text into any f-string or
filename built from a member.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

__all__ = [
    "Neighborhood",
    "NeighborhoodStr",
    "SampleMode",
    "SampleModeStr",
    "TrainingMode",
    "TrainingModeStr",
    "WeightInit",
    "WeightInitStr",
]


class _StrEnum(str, Enum):
    """A string enum whose members render as their value on every supported Python version."""

    def __str__(self) -> str:
        """Return the member's value, as ``enum.StrEnum`` does.

        :return: The string value.
        """
        return str(self.value)


class TrainingMode(_StrEnum):
    """How :meth:`~python_som.SOM.train` presents the data.

    ``RANDOM`` and ``SEQUENTIAL`` are the stepwise algorithm of Kohonen (2013) Eq. (3), differing
    only in the order samples arrive. ``BATCH`` is Eq. (8), which updates every model concurrently.
    """

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    BATCH = "batch"


class Neighborhood(_StrEnum):
    """Which neighborhood function spreads the winner's correction over the grid.

    ``MEXICAN_HAT`` takes negative values and so cannot be used with :attr:`TrainingMode.BATCH`; see
    :data:`~python_som.SIGNED_NEIGHBORHOODS`. The legacy spelling ``"mexicanhat"`` remains accepted
    as a plain string and resolves to the same function, but is not given a member of its own: one
    canonical spelling per option is the point of having an enum.
    """

    GAUSSIAN = "gaussian"
    BUBBLE = "bubble"
    MEXICAN_HAT = "mexican_hat"


class WeightInit(_StrEnum):
    """How :meth:`~python_som.SOM.weight_initialization` seeds the models.

    ``LINEAR`` and ``SAMPLE`` both need a dataset; ``RANDOM`` does not.
    """

    RANDOM = "random"
    LINEAR = "linear"
    SAMPLE = "sample"


class SampleMode(_StrEnum):
    """Which distribution :attr:`WeightInit.RANDOM` draws from."""

    STANDARD_NORMAL = "standard_normal"
    UNIFORM = "uniform"


# The literal spellings, so that passing a plain string is still checked. Annotating these
# parameters as bare ``str`` would accept ``mode="bacth"`` silently, which is most of what the enums
# are for; a union of the enum and a ``Literal`` catches the typo and keeps both spellings working.
#
# The cost is that code passing a variable of unannotated ``str`` type now needs a cast or a
# narrowing check. That is the intended trade: an unvalidated string reaching a mode parameter is
# exactly the case worth surfacing, and it is still accepted at runtime.

#: Accepted strings for :meth:`~python_som.SOM.train`.
TrainingModeStr = Literal["random", "sequential", "batch"]

#: Accepted strings for ``neighborhood_function``. Includes the legacy ``"mexicanhat"`` spelling,
#: which has no enum member but still resolves.
NeighborhoodStr = Literal["gaussian", "bubble", "mexican_hat", "mexicanhat"]

#: Accepted strings for :meth:`~python_som.SOM.weight_initialization`.
WeightInitStr = Literal["random", "linear", "sample"]

#: Accepted strings for the ``sample_mode`` argument of random initialization.
SampleModeStr = Literal["standard_normal", "uniform"]
