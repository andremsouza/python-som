"""Saving and loading a trained map, with the provenance needed to defend a result.

Wilson et al., *Best Practices for Scientific Computing*: a result should carry its inputs,
parameters and versions. Until 0.4.0 the only way to keep a trained map was ``pickle``, which is
arbitrary code execution on load. The point here is not to forbid that but to make the safe path the
obvious one.

**One file.** Everything lives in a single ``.npz``: the models as an array, and the metadata as a
JSON string stored alongside them. A separate sidecar was the first design and is worse, because
provenance that can be separated from its artifact will be.

**What cannot be saved, and what happens instead.** A map holds four callables: the neighborhood,
two decays and the distance. A callable cannot be written to a file without ``pickle``, so what is
stored is its *name*, resolved on load through the registries in :mod:`python_som._core`. A map
built entirely from the shipped functions round-trips completely. One built with a caller's own
function records the name for provenance and refuses to load silently: the loader raises and names
the argument to pass it back through.

**Security.** ``allow_pickle=False`` is passed explicitly on load, so a crafted file containing an
object array is refused by NumPy rather than executed; strategies resolve only through the
registries, so no name from the file is ever imported or evaluated; the metadata is parsed with
``json.loads``. What that buys is "cannot execute code", not "safe to load anything": an ``.npz`` is
a zip, so a hostile file can still attempt resource exhaustion through decompression. Treat one from
an untrusted source the way you would a JPEG, not the way you would a signed archive.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from ._core._decay import DECAY_FUNCTIONS, resolve_decay
from ._core._distance import DISTANCE_FUNCTIONS, resolve_distance
from ._core._neighborhood import NEIGHBORHOOD_FUNCTIONS
from ._enums import Neighborhood

if TYPE_CHECKING:  # pragma: no cover
    import os
    from collections.abc import Callable, Mapping

    import numpy.typing as npt

    from ._core._protocols import DecayFunction, DistanceFunction

__all__ = [
    "ArtifactError",
    "SOMConfig",
    "Strategies",
    "TrainingReport",
    "load_arrays",
    "metadata_json",
]

#: Version of the on-disk layout. Bumped when the schema changes in a way an older reader could
#: misread. A file written by a *newer* version is refused rather than half-understood.
FORMAT_VERSION = 1

#: The two members every artifact contains. Checked on load, so a file with unexpected contents is
#: reported rather than silently partly read.
_MEMBERS = ("weights", "metadata")

#: A weight array is ``(x, y, n_features)``, by definition.
_WEIGHT_DIMENSIONS = 3

#: Spellings that resolve but have no enum member of their own. An artifact written before the enums
#: existed can carry one, and it means the same function.
_LEGACY_NEIGHBORHOOD_ALIASES = {"mexicanhat": Neighborhood.MEXICAN_HAT}

#: Which config fields name a callable, and the registry that restores each. The loader walks this
#: rather than repeating four near-identical branches.
_STRATEGIES: Mapping[str, Mapping[str, Any]] = {
    "neighborhood_function": NEIGHBORHOOD_FUNCTIONS,
    "learning_rate_decay": DECAY_FUNCTIONS,
    "neighborhood_radius_decay": DECAY_FUNCTIONS,
    "distance_function": DISTANCE_FUNCTIONS,
}


class ArtifactError(ValueError):
    """Raised when a file cannot be read as a python-som artifact.

    A subclass of :class:`ValueError` so that ``except ValueError`` in existing code still catches
    it, and distinct so that a caller can tell "this file is not ours" from "this map used a custom
    function".
    """


@dataclass(frozen=True, slots=True)
class SOMConfig:
    """Everything needed to rebuild a map, with strategies recorded by name.

    Frozen, because it describes a map that has already been trained: changing it after the fact
    would describe something that never ran.
    """

    #: Grid shape.
    shape: tuple[int, int]
    #: Number of input features.
    input_len: int
    #: Initial learning rate.
    learning_rate: float
    #: Initial neighborhood radius.
    neighborhood_radius: float
    #: Floor applied to the decayed radius.
    min_neighborhood_radius: float
    #: Whether each axis wraps around.
    cyclic: tuple[bool, bool]
    #: Name of the neighborhood function.
    neighborhood_function: str
    #: Name of the learning-rate decay function.
    learning_rate_decay: str
    #: Name of the neighborhood-radius decay function.
    neighborhood_radius_decay: str
    #: Name of the distance function.
    distance_function: str

    def unresolvable(self) -> dict[str, str]:
        """Return the strategies whose names are not in a registry.

        These are the ones a caller has to pass back in by hand, because a name is all that was
        saved and it does not correspond to anything this package can look up.

        :return: Field name to the saved function name, for each strategy that cannot be restored.
        """
        return {
            attribute: getattr(self, attribute)
            for attribute, registry in _STRATEGIES.items()
            if getattr(self, attribute) not in registry
        }


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What one training run did, recorded so a figure can be traced back to it.

    ``wall_time_seconds`` is excluded from equality: it is provenance rather than an input, and two
    identical runs should compare equal.
    """

    #: Training mode used.
    mode: str
    #: Number of iterations run.
    n_iteration: int
    #: Number of samples presented.
    n_samples: int
    #: Seed the map was constructed with.
    random_seed: int
    #: Learning rate at the final step, or None for batch training, which has no step size: Eq. (8)
    #: is a weighted mean.
    final_learning_rate: float | None
    #: Neighborhood radius at the final step.
    final_neighborhood_radius: float
    #: Mean quantization error after training.
    quantization_error: float
    #: Version of this package that ran it.
    python_som_version: str
    #: Version of NumPy it ran against.
    numpy_version: str
    #: Wall-clock duration. Excluded from equality; see the class docstring.
    wall_time_seconds: float = field(compare=False, default=0.0)


def metadata_json(
    config: SOMConfig,
    seed: int,
    rng_state: Mapping[str, Any],
    report: TrainingReport | None,
    version: str,
) -> str:
    """Serialise everything that is not an array into the JSON blob stored in the artifact.

    The seed and the generator state are separate keys and are not interchangeable. The seed says
    how the stream began, which is provenance; the state says where it has reached, which is what
    resumes it. Collapsing both under one ``rng`` key conflates them, and assigning an inner PCG
    state where a full ``bit_generator.state`` belongs fails with "state must be for a PCG64 RNG".

    :param config: The map's configuration.
    :param seed: The seed the map was constructed with.
    :param rng_state: ``Generator.bit_generator.state`` in full, with its ``bit_generator`` key.
    :param report: The last training run, if the map has been trained.
    :param version: Version of this package.
    :return: The JSON text.
    """
    return json.dumps(
        {
            "format_version": FORMAT_VERSION,
            "python_som_version": version,
            "numpy_version": np.__version__,
            "config": asdict(config),
            "rng": {"seed": seed, "state": dict(rng_state)},
            "report": asdict(report) if report is not None else None,
        },
        indent=2,
        sort_keys=True,
    )


def load_arrays(path: str | os.PathLike[str]) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
    """Read an artifact into its models and its metadata, validating both.

    Everything that could make a hostile or corrupt file dangerous or confusing is checked here
    rather than in the caller: pickle is refused, the member list is pinned, the format version is
    compared, and the weights are converted to a known dtype.

    :param path: File to read.
    :return: The models, and the parsed metadata.
    :raises ArtifactError: If the file is not a readable python-som artifact.
    """
    try:
        # allow_pickle=False is the security boundary: with it, a crafted object array is refused by
        # NumPy instead of being unpickled. Passed explicitly rather than relying on the default,
        # because a default is what a future NumPy could change.
        with np.load(path, allow_pickle=False) as archive:
            present = tuple(archive.files)
            arrays = {name: archive[name] for name in present if name in _MEMBERS}
    except (OSError, ValueError) as exc:
        msg = f"{path} could not be read as a python-som artifact: {exc}"
        raise ArtifactError(msg) from exc

    if set(present) != set(_MEMBERS):
        msg = (
            f"{path} is not a python-som artifact: expected members {list(_MEMBERS)}, "
            f"found {list(present)}"
        )
        raise ArtifactError(msg)

    weights = np.asarray(arrays["weights"], dtype=float)

    try:
        metadata = json.loads(str(arrays["metadata"]))
    except json.JSONDecodeError as exc:
        msg = f"{path} has metadata that is not valid JSON: {exc}"
        raise ArtifactError(msg) from exc

    written_by = metadata.get("format_version")
    if written_by != FORMAT_VERSION:
        msg = (
            f"{path} was written in artifact format {written_by!r}, and this version of python-som "
            f"reads format {FORMAT_VERSION}. Install the version that wrote it "
            f"({metadata.get('python_som_version', 'unknown')}) to read it."
        )
        raise ArtifactError(msg)

    if weights.ndim != _WEIGHT_DIMENSIONS:
        msg = f"{path} holds weights of shape {weights.shape}, expected three dimensions"
        raise ArtifactError(msg)

    return weights, metadata


def config_from(metadata: Mapping[str, Any], path: str | os.PathLike[str]) -> SOMConfig:
    """Rebuild a :class:`SOMConfig` from parsed metadata.

    JSON has no tuples, so the two tuple fields are converted back explicitly. Missing keys are
    reported as a bad artifact rather than raising ``KeyError`` from somewhere further in.

    :param metadata: Parsed metadata.
    :param path: File it came from, for the error message.
    :return: The configuration.
    :raises ArtifactError: If the metadata does not describe a configuration.
    """
    raw = metadata.get("config")
    if not isinstance(raw, dict):
        msg = f"{path} has no 'config' block"
        raise ArtifactError(msg)

    expected = {f.name for f in fields(SOMConfig)}
    missing = expected - set(raw)
    if missing:
        msg = f"{path} has a 'config' block missing {sorted(missing)}"
        raise ArtifactError(msg)

    values = {key: raw[key] for key in expected}
    values["shape"] = (int(values["shape"][0]), int(values["shape"][1]))
    values["cyclic"] = (bool(values["cyclic"][0]), bool(values["cyclic"][1]))
    return SOMConfig(**values)


def report_from(metadata: Mapping[str, Any]) -> TrainingReport | None:
    """Rebuild a :class:`TrainingReport` from parsed metadata, if the map was trained.

    :param metadata: Parsed metadata.
    :return: The report, or None if the saved map had never been trained.
    """
    raw = metadata.get("report")
    if not isinstance(raw, dict):
        return None
    known = {f.name for f in fields(TrainingReport)}
    return TrainingReport(**{key: value for key, value in raw.items() if key in known})


@dataclass(frozen=True, slots=True)
class Strategies:
    """The four callables a map needs, resolved and individually typed.

    A typed record rather than a ``dict[str, Any]`` so the constructor call in ``load_npz`` is
    checked. With a dict, mypy accepts passing the distance where the decay belongs.
    """

    #: Either a registered name or the caller's own callable; the constructor takes either.
    neighborhood_function: Any
    #: Decay applied to the learning rate.
    learning_rate_decay: DecayFunction
    #: Decay applied to the neighborhood radius.
    neighborhood_radius_decay: DecayFunction
    #: Dissimilarity between an input and the models.
    distance_function: DistanceFunction


def _as_neighborhood(name: str) -> Neighborhood | str:
    """Turn a stored neighborhood name into the enum member it denotes.

    Returns the name unchanged when it names nothing this package knows, so the constructor still
    raises the error a corrupt artifact deserves rather than this function swallowing it.

    :param name: Name as recorded in the artifact.
    :return: The matching member, or the name unchanged.
    """
    if name in _LEGACY_NEIGHBORHOOD_ALIASES:
        return _LEGACY_NEIGHBORHOOD_ALIASES[name]
    try:
        return Neighborhood(name)
    except ValueError:
        return name


def resolve_strategies(config: SOMConfig, overrides: Mapping[str, Any]) -> Strategies:
    """Turn the saved strategy names back into callables, honouring any the caller supplied.

    :param config: The saved configuration.
    :param overrides: Caller-supplied callables, keyed by config field name. Only entries that are
        not None are used.
    :return: The four resolved strategies.
    :raises ArtifactError: If a strategy cannot be resolved and was not supplied.
    """
    supplied = {key: value for key, value in overrides.items() if value is not None}
    unresolvable = {key: name for key, name in config.unresolvable().items() if key not in supplied}
    if unresolvable:
        details = ", ".join(f"{key}={name!r}" for key, name in sorted(unresolvable.items()))
        arguments = ", ".join(f"{key}=..." for key in sorted(unresolvable))
        msg = (
            f"this map was saved with custom function(s) that cannot be restored from a name "
            f"({details}). A name is all that was saved, because a callable cannot be written to a "
            f"file without pickle. Pass them back explicitly: load_npz(path, {arguments})"
        )
        raise ArtifactError(msg)

    # Resolved only when not supplied. `supplied.get(key, resolve(...))` would look the wrong way
    # round: Python evaluates the default eagerly, so it would resolve a name the caller had already
    # overridden -- and raise on exactly the custom function this argument exists to accept.
    def either(key: str, resolve: Callable[[str], Any], name: str) -> Any:  # noqa: ANN401
        """Return the caller's callable if given, otherwise resolve the saved name.

        :param key: Config field name.
        :param resolve: Registry lookup for this kind of strategy.
        :param name: The saved name.
        :return: The callable to use.
        """
        return supplied[key] if key in supplied else resolve(name)

    # The constructor takes the neighborhood by name as well as by callable. The saved name is a
    # *string*, because JSON has no enums, and handing it straight to the constructor would emit the
    # plain-string DeprecationWarning for something the caller never wrote. So convert it to the
    # member it denotes. The legacy "mexicanhat" spelling has no member of its own and is an alias
    # for the same function, so it maps to MEXICAN_HAT.
    return Strategies(
        neighborhood_function=supplied.get(
            "neighborhood_function", _as_neighborhood(config.neighborhood_function)
        ),
        learning_rate_decay=either(
            "learning_rate_decay", resolve_decay, config.learning_rate_decay
        ),
        neighborhood_radius_decay=either(
            "neighborhood_radius_decay", resolve_decay, config.neighborhood_radius_decay
        ),
        distance_function=either("distance_function", resolve_distance, config.distance_function),
    )
