"""Saving a map, loading it back, and the ways a file can be wrong.

The claim under test is stronger than "the weights come back". A map is a stochastic process, so a
reloaded map is only *the same map* if continuing to train it produces what never stopping would
have. That is what the saved generator state is for, and it is the assertion that would catch its
absence -- comparing weights alone would pass with the RNG state thrown away.

The rest of the file is the unhappy paths, because a loader is mostly error handling: a custom
callable that cannot be resolved from a name, a file from the future, a file that is not ours, and a
file crafted to smuggle a pickle.
"""

from __future__ import annotations

import io
import json
import pathlib
import pickle
import re
import warnings
import zipfile
from typing import Any

import numpy as np
import pytest

import python_som
from python_som import (
    ArtifactError,
    Neighborhood,
    SOMConfig,
    TrainingMode,
    TrainingReport,
    WeightInit,
)
from python_som._artifact import FORMAT_VERSION
from python_som._core._decay import DECAY_FUNCTIONS
from python_som._core._distance import DISTANCE_FUNCTIONS

#: Independent of the fixture seeds; this file is about persistence, not about training quality.
SEED = 20260801


def _data(n_samples: int = 60, n_features: int = 4) -> np.ndarray:
    """Build a reproducible dataset.

    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :return: The dataset.
    """
    return np.random.default_rng(SEED).normal(size=(n_samples, n_features))


def _trained(**kwargs: Any) -> python_som.SOM:  # noqa: ANN401
    """Build and train a small map.

    :param kwargs: Passed to the constructor.
    :return: The trained map.
    """
    data = _data()
    som = python_som.SOM(x=7, y=5, input_len=4, random_seed=11, **kwargs)
    som.weight_initialization(mode=WeightInit.LINEAR, data=data)
    som.train(data, n_iteration=20, mode=TrainingMode.BATCH)
    return som


# ---------------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------------


def test_weights_survive_exactly(tmp_path: pathlib.Path) -> None:
    """Not to a tolerance. The weights are written and read as float64."""
    som = _trained()
    path = tmp_path / "map.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path)
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0


def test_continuing_to_train_a_loaded_map_matches_never_having_stopped(
    tmp_path: pathlib.Path,
) -> None:
    """The real test of a saved map, and the only one that proves the RNG state is kept.

    Two runs: one trains 20 iterations then another 20; the other trains 20, is saved, reloaded, and
    trains 20 more. Stepwise mode is deliberate -- it draws samples from the generator, so a lost
    RNG state changes which samples arrive and the weights diverge. Batch mode would pass this test
    with the state discarded, since it presents the whole dataset every iteration.
    """
    data = _data()

    uninterrupted = python_som.SOM(x=6, y=5, input_len=4, random_seed=3)
    uninterrupted.weight_initialization(mode=WeightInit.LINEAR, data=data)
    uninterrupted.train(data, n_iteration=20, mode=TrainingMode.RANDOM)
    uninterrupted.train(data, n_iteration=20, mode=TrainingMode.RANDOM)

    interrupted = python_som.SOM(x=6, y=5, input_len=4, random_seed=3)
    interrupted.weight_initialization(mode=WeightInit.LINEAR, data=data)
    interrupted.train(data, n_iteration=20, mode=TrainingMode.RANDOM)
    path = tmp_path / "checkpoint.npz"
    interrupted.save_npz(path)
    resumed = python_som.SOM.load_npz(path)
    resumed.train(data, n_iteration=20, mode=TrainingMode.RANDOM)

    difference = np.abs(resumed.get_weights() - uninterrupted.get_weights()).max()
    assert difference == 0.0, (
        f"a resumed map diverged from an uninterrupted one by {difference}; the generator "
        "state was probably not restored"
    )


def test_reseeding_instead_of_restoring_state_would_diverge(tmp_path: pathlib.Path) -> None:
    """Guard the test above against passing for the wrong reason.

    If re-seeding happened to give the same stream, the previous test would pass with no state saved
    at all and would be worthless. This shows the two really do differ.
    """
    data = _data()
    som = python_som.SOM(x=6, y=5, input_len=4, random_seed=3)
    som.weight_initialization(mode=WeightInit.LINEAR, data=data)
    som.train(data, n_iteration=20, mode=TrainingMode.RANDOM)
    path = tmp_path / "map.npz"
    som.save_npz(path)

    resumed = python_som.SOM.load_npz(path)
    reseeded = python_som.SOM.load_npz(path)
    reseeded._rng = np.random.default_rng(som.get_random_seed())

    resumed.train(data, n_iteration=20, mode=TrainingMode.RANDOM)
    reseeded.train(data, n_iteration=20, mode=TrainingMode.RANDOM)
    assert np.abs(resumed.get_weights() - reseeded.get_weights()).max() > 0.0


def test_the_configuration_survives(tmp_path: pathlib.Path) -> None:
    """Every argument that shapes a map, compared as a whole rather than field by field."""
    som = _trained(
        neighborhood_function=Neighborhood.BUBBLE,
        learning_rate=0.25,
        neighborhood_radius=2.5,
        cyclic_x=True,
        min_neighborhood_radius=0.75,
        learning_rate_decay=python_som.linear_decay,
        neighborhood_radius_decay=python_som.inverse_decay,
    )
    path = tmp_path / "map.npz"
    som.save_npz(path)
    assert python_som.SOM.load_npz(path).config() == som.config()


def test_the_training_report_survives(tmp_path: pathlib.Path) -> None:
    """Including across the wall-time field, which is excluded from equality by design."""
    som = _trained()
    path = tmp_path / "map.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path)
    assert loaded.last_report == som.last_report
    assert loaded.last_report is not None
    assert loaded.last_report.quantization_error == pytest.approx(som.quantization_error(_data()))


def test_an_untrained_map_round_trips_with_no_report(tmp_path: pathlib.Path) -> None:
    """None, not a report full of zeros: the map has not been trained, and should say so."""
    som = python_som.SOM(x=4, y=4, input_len=3, random_seed=5)
    assert som.last_report is None
    path = tmp_path / "fresh.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path)
    assert loaded.last_report is None
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0


@pytest.mark.parametrize("decay", sorted(DECAY_FUNCTIONS))
def test_every_registered_decay_round_trips(decay: str, tmp_path: pathlib.Path) -> None:
    """A registry entry that cannot be resolved back is worse than no registry entry."""
    function = DECAY_FUNCTIONS[decay]
    som = _trained(learning_rate_decay=function, neighborhood_radius_decay=function)
    path = tmp_path / f"{decay}.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path)
    assert loaded.config().learning_rate_decay == decay
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0


@pytest.mark.parametrize("name", sorted(DISTANCE_FUNCTIONS))
def test_every_registered_distance_round_trips(name: str, tmp_path: pathlib.Path) -> None:
    som = _trained(distance_function=DISTANCE_FUNCTIONS[name])
    path = tmp_path / f"{name}.npz"
    som.save_npz(path)
    assert python_som.SOM.load_npz(path).config().distance_function == name


@pytest.mark.parametrize("neighborhood", list(Neighborhood))
def test_every_neighborhood_round_trips(neighborhood: Neighborhood, tmp_path: pathlib.Path) -> None:
    """The mexican hat included, which cannot be trained in batch but can certainly be saved."""
    data = _data()
    som = python_som.SOM(x=6, y=4, input_len=4, neighborhood_function=neighborhood, random_seed=9)
    som.weight_initialization(mode=WeightInit.LINEAR, data=data)
    path = tmp_path / f"{neighborhood.value}.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path)
    assert loaded.config().neighborhood_function == neighborhood.value
    np.testing.assert_array_equal(loaded.neighborhood((3, 2), 1.5), som.neighborhood((3, 2), 1.5))


# ---------------------------------------------------------------------------------------------
# Custom callables: recorded, refused, and recoverable
# ---------------------------------------------------------------------------------------------


def _cosine(a: Any, b: Any) -> np.ndarray:  # noqa: ANN401
    """Return one minus the cosine similarity: a distance this package does not ship.

    :param a: Input vector.
    :param b: Models.
    :return: One minus the cosine similarity.
    """
    a_array, b_array = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = np.sum(a_array * b_array, axis=-1)
    norms = np.linalg.norm(a_array, axis=-1) * np.linalg.norm(b_array, axis=-1)
    return np.asarray(1.0 - dot / np.where(norms == 0, 1.0, norms))


def test_a_custom_callable_is_recorded_by_name(tmp_path: pathlib.Path) -> None:
    """Provenance first: the file should say what was used, even though it cannot restore it."""
    som = _trained(distance_function=_cosine)
    path = tmp_path / "custom.npz"
    som.save_npz(path)

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert metadata["config"]["distance_function"] == "_cosine"


def test_loading_a_custom_callable_raises_and_says_how_to_fix_it(tmp_path: pathlib.Path) -> None:
    """The error has to name the argument, because that is the only way out of it."""
    som = _trained(distance_function=_cosine)
    path = tmp_path / "custom.npz"
    som.save_npz(path)

    with pytest.raises(ArtifactError, match="distance_function") as excinfo:
        python_som.SOM.load_npz(path)
    assert "load_npz(path, distance_function=...)" in str(excinfo.value)


def test_passing_the_callable_back_makes_it_load(tmp_path: pathlib.Path) -> None:
    """The other half: an error whose suggested fix does not work is worse than no suggestion."""
    som = _trained(distance_function=_cosine)
    path = tmp_path / "custom.npz"
    som.save_npz(path)

    loaded = python_som.SOM.load_npz(path, distance_function=_cosine)
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0
    np.testing.assert_array_equal(loaded.activate(_data()[0]), som.activate(_data()[0]))


def test_a_partial_is_treated_as_custom(tmp_path: pathlib.Path) -> None:
    """``partial(exponential_decay, factor=3.0)`` must not silently reload as ``factor=2.0``.

    A partial has no ``__name__``, so it cannot be resolved, which is the correct outcome. Resolving
    it by the wrapped function's name would restore a different decay from the one that trained the
    map, and nothing would say so.
    """
    import functools  # noqa: PLC0415

    steeper = functools.partial(python_som.exponential_decay, factor=3.0)
    som = _trained(learning_rate_decay=steeper)
    path = tmp_path / "partial.npz"
    som.save_npz(path)

    with pytest.raises(ArtifactError, match="learning_rate_decay"):
        python_som.SOM.load_npz(path)


# ---------------------------------------------------------------------------------------------
# Bad files
# ---------------------------------------------------------------------------------------------


def test_a_real_pickle_payload_does_not_execute(tmp_path: pathlib.Path) -> None:
    """The security claim, demonstrated rather than asserted.

    An object array is how a pickle rides inside an ``.npz``, and ``__reduce__`` is how a pickle
    executes on load. This proves the payload is live, then shows the loader refusing a file that
    contains it, with nothing created.

    Proving liveness first is what matters: without it, this test would pass just as well against an
    inert payload and would be evidence of nothing. Liveness is shown through ``pickle`` directly
    rather than by asking NumPy to unpickle it, which keeps the enabled form of NumPy's pickle flag
    out of this repository entirely. The architecture profile forbids that flag at the artifact
    boundary, and a test that had to switch the rule off in order to run would be the worse test.
    """
    marker = tmp_path / "executed"

    class Payload:
        """Creates ``marker`` if it is ever unpickled."""

        def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
            """Return the call pickle should make on load."""
            return (pathlib.Path.touch, (marker,))

    # The payload is live: unpickling it really does run the call.
    pickle.loads(pickle.dumps(Payload()))  # noqa: S301  that is the point of this line
    assert marker.exists(), "the payload is inert, so this test would prove nothing"
    marker.unlink()

    # Build an .npz whose `weights` member is that pickled object array. np.save pickles object
    # arrays by default, which is exactly the hostile file a user might be handed.
    buffer = io.BytesIO()
    np.save(buffer, np.array([Payload()], dtype=object))
    metadata = io.BytesIO()
    np.save(metadata, np.array(json.dumps({"format_version": FORMAT_VERSION, "config": {}})))

    hostile = tmp_path / "hostile.npz"
    with zipfile.ZipFile(hostile, "w") as archive:
        archive.writestr("weights.npy", buffer.getvalue())
        archive.writestr("metadata.npy", metadata.getvalue())

    with pytest.raises(ArtifactError):
        python_som.SOM.load_npz(hostile)
    assert not marker.exists(), "load_npz executed a pickled payload"


def test_a_file_containing_an_object_array_is_refused(tmp_path: pathlib.Path) -> None:
    """The security boundary, exercised rather than asserted about.

    An object array is how a pickle rides inside an ``.npz``. Because the loader passes
    ``allow_pickle=False``, NumPy refuses it instead of unpickling, and the loader reports it as a
    bad artifact rather than letting the ``ValueError`` escape unexplained.
    """
    path = tmp_path / "hostile.npz"
    np.savez(
        path,
        weights=np.array([{"payload": "would be unpickled"}], dtype=object),
        metadata=np.array("{}"),
    )
    with pytest.raises(ArtifactError):
        python_som.SOM.load_npz(path)


def test_a_file_from_a_newer_format_is_refused_naming_the_version(tmp_path: pathlib.Path) -> None:
    """Refused rather than half-read, and the message says what wrote it."""
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))

    metadata["format_version"] = FORMAT_VERSION + 1
    metadata["python_som_version"] = "99.0.0"
    future = tmp_path / "future.npz"
    np.savez(future, weights=weights, metadata=np.array(json.dumps(metadata)))

    with pytest.raises(ArtifactError, match=re.escape("99.0.0")):
        python_som.SOM.load_npz(future)


@pytest.mark.parametrize(
    "state",
    [
        {"bit_generator": "Philox", "state": {"counter": 1}},
        {"bit_generator": "PCG64"},
        "not a mapping at all",
        {},
    ],
    ids=["wrong-generator", "missing-inner-state", "not-a-mapping", "empty"],
)
def test_an_unusable_generator_state_is_refused_as_an_artifact_error(
    state: object, tmp_path: pathlib.Path
) -> None:
    """Every way a bad file can fail should reach the caller the same way.

    NumPy raises its own ``ValueError`` here ("state must be for a PCG64 RNG"), which untranslated
    would tell the reader about bit generators rather than about their file.
    """
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))

    metadata["rng"]["state"] = state
    broken = tmp_path / "broken-rng.npz"
    np.savez(broken, weights=weights, metadata=np.array(json.dumps(metadata)))

    with pytest.raises(ArtifactError, match="unusable random generator state"):
        python_som.SOM.load_npz(broken)


def test_a_file_with_unexpected_members_is_refused(tmp_path: pathlib.Path) -> None:
    """An npz that is not ours should be reported as such, not indexed into and crash."""
    path = tmp_path / "someone-elses.npz"
    np.savez(path, something=np.zeros(3))
    with pytest.raises(ArtifactError, match="not a python-som artifact"):
        python_som.SOM.load_npz(path)


def test_metadata_that_is_not_json_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "corrupt.npz"
    np.savez(path, weights=np.zeros((2, 2, 2)), metadata=np.array("{not json"))
    with pytest.raises(ArtifactError, match="not valid JSON"):
        python_som.SOM.load_npz(path)


def test_a_missing_config_block_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "no-config.npz"
    metadata = {"format_version": FORMAT_VERSION, "python_som_version": "0.4.0"}
    np.savez(path, weights=np.zeros((2, 2, 2)), metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ArtifactError, match=r"no 'config' block"):
        python_som.SOM.load_npz(path)


def test_an_incomplete_config_block_names_what_is_missing(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "partial-config.npz"
    metadata = {
        "format_version": FORMAT_VERSION,
        "config": {"shape": [2, 2], "input_len": 2},
    }
    np.savez(path, weights=np.zeros((2, 2, 2)), metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ArtifactError, match="missing"):
        python_som.SOM.load_npz(path)


def test_weights_that_contradict_the_config_are_refused(tmp_path: pathlib.Path) -> None:
    """A shape mismatch is corruption, and constructing the map anyway would hide it."""
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        metadata = str(archive["metadata"])

    wrong = tmp_path / "wrong-shape.npz"
    np.savez(wrong, weights=np.zeros((3, 3, 3)), metadata=np.array(metadata))
    with pytest.raises(ArtifactError, match="does not match the saved configuration"):
        python_som.SOM.load_npz(wrong)


def test_two_dimensional_weights_are_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "flat.npz"
    metadata = {"format_version": FORMAT_VERSION, "config": {}}
    np.savez(path, weights=np.zeros((4, 4)), metadata=np.array(json.dumps(metadata)))
    with pytest.raises(ArtifactError, match="expected three dimensions"):
        python_som.SOM.load_npz(path)


def test_a_file_that_is_not_an_npz_at_all_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("this is not an archive")
    with pytest.raises(ArtifactError, match="could not be read"):
        python_som.SOM.load_npz(path)


def test_a_missing_file_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ArtifactError, match="could not be read"):
        python_som.SOM.load_npz(tmp_path / "absent.npz")


# ---------------------------------------------------------------------------------------------
# What is actually on disk
# ---------------------------------------------------------------------------------------------


def test_the_artifact_holds_exactly_two_members_and_no_pickle(tmp_path: pathlib.Path) -> None:
    """Read as a plain zip, so the assertion does not depend on NumPy's loader.

    ``.npy`` members only, and no ``.pkl``: the file format itself, not merely the flag we pass when
    reading it.
    """
    som = _trained()
    path = tmp_path / "map.npz"
    som.save_npz(path)

    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
    assert names == ["metadata.npy", "weights.npy"]


def test_the_metadata_is_readable_json_with_the_documented_keys(tmp_path: pathlib.Path) -> None:
    """The artifact is meant to be inspectable without this package installed."""
    som = _trained()
    path = tmp_path / "map.npz"
    som.save_npz(path)

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))

    assert set(metadata) == {
        "format_version",
        "python_som_version",
        "numpy_version",
        "config",
        "rng",
        "report",
    }
    assert metadata["numpy_version"] == np.__version__
    assert metadata["python_som_version"] == python_som.__version__
    assert metadata["rng"]["seed"] == som.get_random_seed()


def test_loading_an_artifact_from_a_different_major_version_warns(tmp_path: pathlib.Path) -> None:
    """Allowed, because refusing would make old artifacts unreadable, but not silent.

    A major version is where this package permits numerics to change, so a map trained by one and
    reloaded under another may not reproduce the run its own report describes.
    """
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))

    metadata["python_som_version"] = "99.1.2"
    other = tmp_path / "other-major.npz"
    np.savez(other, weights=weights, metadata=np.array(json.dumps(metadata)))

    with pytest.warns(UserWarning, match="Major versions"):
        python_som.SOM.load_npz(other)


# ---------------------------------------------------------------------------------------------
# The value types
# ---------------------------------------------------------------------------------------------


def test_config_and_report_are_frozen() -> None:
    """Both describe something that already happened, so neither should be editable after."""
    som = _trained()
    with pytest.raises((AttributeError, TypeError)):
        som.config().input_len = 99  # type: ignore[misc]
    report = som.last_report
    assert report is not None
    with pytest.raises((AttributeError, TypeError)):
        report.quantization_error = 0.0  # type: ignore[misc]


def test_wall_time_is_excluded_from_report_equality() -> None:
    """Two identical runs should compare equal, and wall time is the field that will not match."""
    first, second = _trained().last_report, _trained().last_report
    assert first is not None
    assert second is not None
    assert first == second
    assert first.wall_time_seconds > 0.0


def test_unresolvable_reports_only_what_cannot_be_resolved() -> None:
    """A map built entirely from shipped functions has nothing to report."""
    assert _trained().config().unresolvable() == {}
    assert set(_trained(distance_function=_cosine).config().unresolvable()) == {"distance_function"}


def test_batch_training_reports_no_learning_rate() -> None:
    """Eq. (8) is a weighted mean: there is no step size, so None rather than an unused number."""
    report = _trained().last_report
    assert report is not None
    assert report.mode == "batch"
    assert report.final_learning_rate is None


def test_stepwise_training_reports_the_rate_it_finished_on() -> None:
    """Not the rate it started with, which is what a naive report would record."""
    data = _data()
    som = python_som.SOM(x=5, y=5, input_len=4, learning_rate=0.5, random_seed=4)
    som.train(data, n_iteration=50, mode=TrainingMode.RANDOM)
    report = som.last_report
    assert report is not None
    assert report.final_learning_rate is not None
    assert report.final_learning_rate < 0.5
    assert report.final_neighborhood_radius >= 0.5  # the documented floor


def test_a_config_can_be_built_by_hand() -> None:
    """It is public API, so it has to be constructible without a SOM to hand."""
    config = SOMConfig(
        shape=(3, 4),
        input_len=2,
        learning_rate=0.5,
        neighborhood_radius=1.0,
        min_neighborhood_radius=0.5,
        cyclic=(False, True),
        neighborhood_function=Neighborhood.GAUSSIAN,
        learning_rate_decay="asymptotic_decay",
        neighborhood_radius_decay="asymptotic_decay",
        distance_function="euclidean_distance",
    )
    assert config.unresolvable() == {}
    assert isinstance(TrainingReport, type)


# ---------------------------------------------------------------------------------------------
# The registries
# ---------------------------------------------------------------------------------------------


def test_resolve_decay_rejects_an_unknown_name() -> None:
    """The error lists the valid names, as the neighborhood resolver does."""
    from python_som._core._decay import resolve_decay  # noqa: PLC0415

    with pytest.raises(ValueError, match="Unknown decay function") as excinfo:
        resolve_decay("cosine_annealing")
    assert "asymptotic_decay" in str(excinfo.value)


def test_resolve_distance_rejects_an_unknown_name() -> None:
    from python_som._core._distance import resolve_distance  # noqa: PLC0415

    with pytest.raises(ValueError, match="Unknown distance function") as excinfo:
        resolve_distance("manhattan")
    assert "euclidean_distance" in str(excinfo.value)


def test_every_registry_key_is_what_save_would_record() -> None:
    """The invariant that makes a round trip work, stated as the two halves meeting.

    ``save_npz`` records a strategy through ``_name_of``; ``load_npz`` looks the result up in a
    registry. So the registry key must be exactly what ``_name_of`` produces for that function --
    checked against the real accessor rather than against ``__name__``, because ``_name_of`` is what
    runs and it has a fallback the protocols do not promise.

    A mismatch would restore a *different* function while the file looked correct.
    """
    from python_som._som import _name_of  # noqa: PLC0415

    for registry in (DECAY_FUNCTIONS, DISTANCE_FUNCTIONS):
        for name, function in registry.items():
            assert _name_of(function) == name, f"{name} is recorded as {_name_of(function)!r}"


def test_an_artifact_with_no_recorded_version_does_not_warn(tmp_path: pathlib.Path) -> None:
    """A missing version is not a mismatch, so it should pass quietly rather than guess."""
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))

    del metadata["python_som_version"]
    quiet = tmp_path / "no-version.npz"
    np.savez(quiet, weights=weights, metadata=np.array(json.dumps(metadata)))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python_som.SOM.load_npz(quiet)


def test_an_artifact_with_no_rng_state_still_loads(tmp_path: pathlib.Path) -> None:
    """Older or hand-built artifacts may carry a seed and no state; the weights still matter."""
    som = _trained()
    good = tmp_path / "good.npz"
    som.save_npz(good)
    with np.load(good, allow_pickle=False) as archive:
        weights = archive["weights"]
        metadata = json.loads(str(archive["metadata"]))

    metadata["rng"] = {"seed": 11}
    stateless = tmp_path / "no-state.npz"
    np.savez(stateless, weights=weights, metadata=np.array(json.dumps(metadata)))

    loaded = python_som.SOM.load_npz(stateless)
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0
    assert loaded.get_random_seed() == 11


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("gaussian", Neighborhood.GAUSSIAN),
        ("mexican_hat", Neighborhood.MEXICAN_HAT),
        ("mexicanhat", Neighborhood.MEXICAN_HAT),
        ("sombrero", "sombrero"),
    ],
    ids=["member", "canonical-alias-target", "legacy-spelling", "unknown"],
)
def test_a_stored_neighborhood_name_becomes_the_member_it_denotes(
    stored: str, expected: object
) -> None:
    """Loading a map must not warn the caller about a string they never wrote.

    The name in an artifact is a serialisation detail: JSON has no enums. Handing it straight to the
    constructor would emit the plain-string ``DeprecationWarning`` for something the file chose, not
    the caller. ``mexicanhat`` has no member of its own and is an alias for the same function, so it
    maps to ``MEXICAN_HAT``.

    An unrecognised name is returned unchanged, so a corrupt artifact still gets the constructor's
    error rather than having it swallowed here.
    """
    from python_som._artifact import _as_neighborhood  # noqa: PLC0415

    assert _as_neighborhood(stored) == expected


def test_loading_a_map_does_not_warn_about_deprecated_spellings(tmp_path: pathlib.Path) -> None:
    """The behaviour the conversion above exists for, asserted end to end."""
    # Stepwise, because the mexican hat is signed and batch rejects it.
    data = _data()
    som = python_som.SOM(
        x=6, y=5, input_len=4, neighborhood_function=Neighborhood.MEXICAN_HAT, random_seed=11
    )
    som.weight_initialization(mode=WeightInit.LINEAR, data=data)
    som.train(data, n_iteration=10, mode=TrainingMode.RANDOM)
    path = tmp_path / "m.npz"
    som.save_npz(path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = python_som.SOM.load_npz(path)
    assert np.abs(loaded.get_weights() - som.get_weights()).max() == 0.0
