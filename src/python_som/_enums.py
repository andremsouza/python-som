"""Names for the string-valued options, so a typo is a type error rather than a runtime one.

Every option these cover is still accepted as a plain string, and will be for the whole 0.4.x and
0.5.x series. ``mode=TrainingMode.BATCH`` and ``mode="batch"`` are interchangeable, compare equal,
hash equal, and serialise to the same JSON, because each member *is* a ``str``.

**Deprecation runway.** Plain strings are deprecated as of 0.4.0 and emit ``DeprecationWarning``
from 0.5.0. 0.4.0 deliberately warned about nothing, because ``mode="batch"`` was what every
documentation page showed at the time, so the written notice came first and this is the warning that
follows it. 1.0.0 removes strings entirely, and :func:`warn_if_string` goes with them.

**On the base class.** ``enum.StrEnum`` arrived in Python 3.11 and this package supports 3.10, so
:class:`_StrEnum` reproduces it. A bare ``class X(str, Enum)`` is *not* equivalent: its ``str()``
returns ``'X.MEMBER'`` rather than the value, which would put the wrong text into any f-string,
filename or log line built from a member. Defining ``__str__`` explicitly makes the behaviour
identical on every supported version, which was checked on 3.10, 3.12 and 3.13 rather than assumed.
"""

from __future__ import annotations

import warnings
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
    "warn_if_string",
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


#: Names that are accepted but have no member of their own, mapped to the member that replaces them.
#: ``mexicanhat`` is the spelling the original contribution used; the enum carries one canonical
#: spelling per option, so this is where the other one is answered for.
_LEGACY_SPELLINGS = {"mexicanhat": "Neighborhood.MEXICAN_HAT"}


def warn_if_string(value: object, enum: type[Enum], parameter: str, *, stacklevel: int = 3) -> None:
    """Emit a ``DeprecationWarning`` when a plain string is passed instead of an enum member.

    An enum member *is* a ``str``, so the test is "a string that is not one of ours" rather than
    ``isinstance(value, str)``, which would fire on the enum too.

    Only *valid* spellings warn. A string naming nothing returns silently and is left to whatever
    validates it, so ``mode="stochastic"`` still raises its ``ValueError`` rather than surfacing a
    deprecation notice about a spelling that never worked.

    The message names the exact replacement rather than describing one. A deprecation warning that
    makes the reader work out the substitution is a deprecation warning people silence.

    :param value: What the caller passed.
    :param enum: The enum that should replace a string here.
    :param parameter: Name of the parameter, for the message.
    :param stacklevel: Frames to skip so the warning blames the caller's line rather than this one.
    """
    if isinstance(value, enum) or not isinstance(value, str):
        return

    try:
        replacement = f"{enum.__name__}.{enum(value).name}"
    except ValueError:
        replacement = _LEGACY_SPELLINGS.get(value, "")
        if not replacement:
            # Not a value this enum knows and not a legacy alias, so it is simply wrong. Whoever
            # validates it will say so; telling the caller to modernise a spelling that was never
            # valid would bury the actual error under a deprecation notice.
            return

    warnings.warn(
        f"Passing {parameter}={value!r} as a plain string is deprecated and will stop working in "
        f"1.0.0. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
