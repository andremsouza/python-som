"""The properties that make the functional core a core, rather than just a folder.

These are architecture tests. They fail if a later change reintroduces a dependency, an invented
tolerance, or an in-place update, none of which a linter would catch on its own.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil

import numpy as np
import pytest

import python_som
from python_som import WeightInit
from python_som._core import _update
from python_som._core._neighborhood import axis_matrix, gaussian, gaussian_axis_profile
from tests.conftest import MODEL_SEED, make_som

#: The core package on disk, scanned rather than imported.
_CORE = pathlib.Path(python_som.__file__).parent / "_core"


# ---------------------------------------------------------------------------------------------
# The dependency boundary
# ---------------------------------------------------------------------------------------------


def _imports_of(path: pathlib.Path) -> set[str]:
    """Return the top-level package names imported by one module, from its AST.

    :param path: Module to inspect.
    :return: Top-level names of everything it imports.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_no_core_module_imports_pandas() -> None:
    """The dependency direction, asserted independently of ruff.

    Deliberately an AST scan rather than a check of ``sys.modules``. Importing
    ``python_som._core`` necessarily imports the parent package first, whose ``__init__`` reaches
    the shell and therefore pandas; that is correct, and a runtime check would flag it. What matters
    is that no module *inside* the core names pandas, which is what this measures.

    ruff's ``TID251`` enforces the same rule, so this is belt and braces. It is worth having because
    a ``per-file-ignores`` entry added later would silence ruff without failing anything.
    """
    offenders = {
        path.name: sorted(_imports_of(path) & {"pandas"})
        for path in _CORE.glob("*.py")
        if _imports_of(path) & {"pandas"}
    }
    assert not offenders, f"the core must not import pandas: {offenders}"


def test_no_core_module_imports_sklearn() -> None:
    """0.4.0 replaced the one sklearn use with ``np.linalg.svd``, so the ban is now absolute.

    Through 0.3.0 this asserted the opposite -- that ``_linalg`` was the *only* module reaching
    sklearn -- which is what kept the removal a single-file change rather than a hunt.
    """
    offenders = sorted(p.name for p in _CORE.glob("*.py") if "sklearn" in _imports_of(p))
    assert not offenders, f"the core must not import scikit-learn: {offenders}"


def test_the_whole_package_is_numpy_only_at_runtime() -> None:
    """The shell, not just the core. This is what the 264 MB saving actually rests on.

    Scans every module under ``src/`` rather than only ``_core/``, because after 0.4.0 no module on
    the import path of ``python_som`` reaches pandas or scikit-learn: the port is ``np.asarray`` and
    the PCA is ``np.linalg.svd``. A `per-file-ignores` entry restoring either would pass ruff
    silently.

    ``sklearn.py`` is exempt and is the *only* exemption. It is the adapter whose entire job is to
    import scikit-learn, it is never imported by anything else in the package, and
    ``tests/test_sklearn_adapter.py`` asserts that importing ``python_som`` does not pull it in. An
    adapter may depend on the thing it adapts; that is what makes it an adapter rather than a
    dependency.
    """
    package = pathlib.Path(python_som.__file__).parent
    banned = {"pandas", "sklearn", "scipy"}
    exempt = {"sklearn.py"}
    offenders = {
        str(path.relative_to(package)): sorted(_imports_of(path) & banned)
        for path in package.rglob("*.py")
        if path.name not in exempt and _imports_of(path) & banned
    }
    assert not offenders, f"runtime modules must not import {banned}: {offenders}"


def test_the_only_module_allowed_to_import_sklearn_is_the_adapter() -> None:
    """State the exemption positively, so it cannot quietly grow.

    The test above skips ``sklearn.py`` by name. This one asserts that the skip covers exactly one
    module, so adding a second scikit-learn import anywhere would need a deliberate edit here.
    """
    package = pathlib.Path(python_som.__file__).parent
    users = sorted(
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if "sklearn" in _imports_of(path)
    )
    assert users == ["sklearn.py"], users


def test_every_core_module_is_importable() -> None:
    """A module that no longer imports is a broken boundary, not a missing test."""
    package = python_som._core
    found = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
    assert len(found) >= 8, f"expected the full core, found {found}"
    for name in found:
        importlib.import_module(f"python_som._core.{name}")


# ---------------------------------------------------------------------------------------------
# The batch denominator: no invented tolerance
# ---------------------------------------------------------------------------------------------


def test_batch_denominator_needs_no_epsilon() -> None:
    """Regression for an invented constant that 0.3.0 shipped.

    The guard was ``abs(denominator) > 1e-12``, a number with no source. It is unnecessary: every
    term of ``sum_j n_j h_ji`` is non-negative, because batch training rejects signed neighborhoods
    and a caller cannot supply an arbitrary one. A sum of non-negative floats admits no
    cancellation, so it is zero exactly when every term is zero.

    This asserts the mathematical premise the simplification rests on, so that if a future change
    lets a signed neighborhood reach the batch update, this fails rather than the update silently
    dividing by noise.
    """
    assert "TOLERANCE" not in dir(_update)
    assert not any("1e-12" in str(v) for v in vars(_update).values() if isinstance(v, float))

    shape = (9, 7)
    for radius in (0.5, 1.0, 3.0, 10.0):
        h = gaussian(shape, (4, 3), radius, (False, False))
        assert (h >= 0).all(), "the gaussian must be non-negative for the premise to hold"


def test_batch_update_leaves_unreached_models_untouched() -> None:
    """A denominator of exactly zero means no data in reach, so the model is kept."""
    shape = (4, 4)
    weights = np.arange(4 * 4 * 2, dtype=float).reshape((*shape, 2))
    sums = np.zeros((*shape, 2))
    counts = np.zeros(shape)  # no data anywhere
    hx = axis_matrix(shape[0], 1.0, cyclic=False, profile=gaussian_axis_profile)
    hy = axis_matrix(shape[1], 1.0, cyclic=False, profile=gaussian_axis_profile)
    result = _update.batch_update(weights, sums, counts, hx, hy)
    np.testing.assert_array_equal(result, weights)


def test_batch_update_returns_a_new_array() -> None:
    """The update is pure: the caller's array must be untouched."""
    shape = (3, 3)
    weights = np.ones((*shape, 2))
    original = weights.copy()
    counts = np.ones(shape)
    sums = np.full((*shape, 2), 5.0)
    hx = axis_matrix(shape[0], 1.0, cyclic=False, profile=gaussian_axis_profile)
    hy = axis_matrix(shape[1], 1.0, cyclic=False, profile=gaussian_axis_profile)
    result = _update.batch_update(weights, sums, counts, hx, hy)
    assert result is not weights
    np.testing.assert_array_equal(weights, original)


# ---------------------------------------------------------------------------------------------
# The stepwise update: pure, and identical to the in-place form it replaced
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 3), (12, 8), (20, 20)])
@pytest.mark.parametrize("alpha", [0.05, 0.5, 1.0])
def test_pure_stepwise_update_equals_the_in_place_form_exactly(
    shape: tuple[int, int], alpha: float
) -> None:
    """Equality at 0.0, not a tolerance: the expression is the same, only the destination differs.

    0.3.0 wrote ``weights += alpha * h[..., None] * (sample - weights)``. If the pure form differed
    at all, every trained map would shift.
    """
    rng = np.random.default_rng(MODEL_SEED)
    weights = rng.normal(size=(*shape, 4))
    sample = rng.normal(size=4)
    h = gaussian(shape, (shape[0] // 2, shape[1] // 2), 2.0, (False, False))

    in_place = weights.copy()
    in_place += alpha * h[..., None] * (sample - in_place)
    pure = _update.stepwise_update(weights, sample, h, alpha)

    assert np.abs(pure - in_place).max() == 0.0


def test_stepwise_update_returns_a_new_array() -> None:
    rng = np.random.default_rng(MODEL_SEED)
    weights = rng.normal(size=(4, 4, 2))
    original = weights.copy()
    h = gaussian((4, 4), (2, 2), 1.0, (False, False))
    result = _update.stepwise_update(weights, rng.normal(size=2), h, 0.3)
    assert result is not weights
    np.testing.assert_array_equal(weights, original)


def test_a_signed_neighborhood_moves_models_away() -> None:
    """The inhibitory half of the mexican hat, asserted directly rather than inferred."""
    shape = (3, 3)
    weights = np.zeros((*shape, 1))
    sample = np.array([10.0])
    h = np.full(shape, -0.5)  # wholly inhibitory
    result = _update.stepwise_update(weights, sample, h, 1.0)
    assert (result < 0).all(), "a negative neighborhood must push models away from the sample"


# ---------------------------------------------------------------------------------------------
# The public surface
# ---------------------------------------------------------------------------------------------

#: Everything 0.3.0 exported. Nothing here may disappear before 1.0.0.
_SHIPPED_IN_0_3_0 = frozenset(
    {
        "NEIGHBORHOOD_FUNCTIONS",
        "SIGNED_NEIGHBORHOODS",
        "SOM",
        "asymptotic_decay",
        "bubble",
        "euclidean_distance",
        "exponential_decay",
        "gaussian",
        "inverse_decay",
        "linear_decay",
        "mexican_hat",
    }
)


def test_nothing_that_0_3_0_exported_has_been_removed() -> None:
    """The compatibility promise, stated separately from the current surface.

    Removing a name is a breaking change and belongs in 1.0.0. Adding one is not, which is why this
    is a subset check rather than an equality check -- the equality check lives in the test below,
    where a deliberate addition shows up as a diff to read rather than a failure to explain.
    """
    missing = _SHIPPED_IN_0_3_0 - set(python_som.__all__)
    assert not missing, f"0.3.0 exported these and they are gone: {sorted(missing)}"
    for name in sorted(_SHIPPED_IN_0_3_0):
        assert hasattr(python_som, name), f"{name} is in __all__ but not importable"


def test_the_public_surface_is_exactly_this() -> None:
    """Pin the whole surface, so growing it is a decision rather than an accident.

    0.4.0 adds the enums and their ``Literal`` counterparts, the strategy protocols, and the
    artifact types. All are additive: every existing call keeps working, and the enum members are
    ``str`` subclasses that compare equal to the strings they replace. ``__version__`` is
    deliberately absent -- it is re-exported with the ``as`` idiom, since ``__all__`` is the public
    API and a dunder is not part of it.
    """
    assert sorted(python_som.__all__) == [
        "ArtifactError",
        "DecayFunction",
        "DistanceFunction",
        "KernelFunction",
        "NEIGHBORHOOD_FUNCTIONS",
        "Neighborhood",
        "NeighborhoodFunction",
        "NeighborhoodStr",
        "SIGNED_NEIGHBORHOODS",
        "SOM",
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


#: Every public method 0.3.0 shipped. None may disappear before 1.0.0.
_METHODS_IN_0_3_0 = frozenset(
    {
        "activate",
        "activation_matrix",
        "distance_matrix",
        "get_random_seed",
        "get_shape",
        "get_weights",
        "label_map",
        "neighborhood",
        "quantization",
        "quantization_error",
        "set_learning_rate",
        "set_neighborhood_radius",
        "train",
        "weight_initialization",
        "winner",
        "winner_map",
    }
)


def _public_methods() -> list[str]:
    """Return SOM's public method names, classmethods included.

    The predicate is ``isfunction or ismethod`` rather than ``isfunction`` alone. A classmethod
    accessed through the class is a *bound method*, so ``isfunction`` is False for it -- which meant
    an earlier version of this check silently ignored ``load_npz`` and would have ignored any
    classmethod added later. A surface pin with a hole in it is worse than none.

    :return: Sorted names, excluding anything underscore-prefixed.
    """
    import inspect  # noqa: PLC0415

    return sorted(
        name
        for name, member in inspect.getmembers(python_som.SOM)
        if not name.startswith("_") and (inspect.isfunction(member) or inspect.ismethod(member))
    )


def test_no_public_method_that_0_3_0_shipped_has_been_removed() -> None:
    """The compatibility promise for methods, as a subset check.

    Adding a method is not a breaking change and does not belong in the same assertion as removing
    one, which is.
    """
    missing = _METHODS_IN_0_3_0 - set(_public_methods())
    assert not missing, f"0.3.0 had these methods and they are gone: {sorted(missing)}"


def test_the_public_methods_are_exactly_these() -> None:
    """Pin the whole set, so growing it is deliberate.

    0.6.0 adds the estimator interface: ``fit``, ``fit_transform``, ``transform``, ``predict``,
    ``score``, ``get_params``, ``set_params``. All additive, all delegating to methods that already
    existed. ``train`` remains the primary way to train.

    The properties the same release adds (``weights_``, ``n_features_in_``, ``quantization_error_``,
    ``last_report``) are not methods and so do not appear here; ``tests/test_estimator.py`` covers
    them.
    """
    assert _public_methods() == [
        "activate",
        "activation_matrix",
        "config",
        "distance_matrix",
        "fit",
        "fit_transform",
        "get_params",
        "get_random_seed",
        "get_shape",
        "get_weights",
        "label_map",
        "load_npz",
        "neighborhood",
        "predict",
        "quantization",
        "quantization_error",
        "save_npz",
        "score",
        "set_learning_rate",
        "set_neighborhood_radius",
        "set_params",
        "train",
        "transform",
        "weight_initialization",
        "winner",
        "winner_map",
    ]


# ---------------------------------------------------------------------------------------------
# Initialization errors name the argument, not a private function
# ---------------------------------------------------------------------------------------------


def test_missing_data_argument_names_the_argument() -> None:
    """0.3.0 raised ``TypeError: SOM._init_random() got an unexpected keyword argument``.

    That leaked a private method name and did not say what the caller should do.
    """
    som = make_som(x=4, y=4)
    for mode in (WeightInit.LINEAR, WeightInit.SAMPLE):
        with pytest.raises(ValueError, match=f"{mode.value!r} initialization requires"):
            som.weight_initialization(mode=mode)


def test_unexpected_argument_names_the_argument() -> None:
    som = make_som(x=4, y=4)
    with pytest.raises(ValueError, match="Unexpected argument"):
        som.weight_initialization(mode=WeightInit.RANDOM, data=np.zeros((4, 2)))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"mode": WeightInit.LINEAR}, "initialization requires"),
        ({"mode": WeightInit.RANDOM, "nonsense": 1}, "Unexpected argument"),
        ({"mode": "spectral"}, "Invalid value for 'mode' parameter"),
    ],
    ids=["missing-data", "unexpected-kwarg", "unknown-mode"],
)
def test_error_messages_never_leak_a_private_name(kwargs: dict[str, object], expected: str) -> None:
    """A caller should never be shown ``_init_random`` or ``_som``.

    Each case pins the message it expects as well, so this cannot pass by raising the wrong error
    for the right reason.
    """
    som = make_som(x=4, y=4)
    with pytest.raises(ValueError, match=expected) as excinfo:
        som.weight_initialization(**kwargs)  # type: ignore[arg-type]
    message = str(excinfo.value)
    assert "_init_" not in message, message
    assert "_som" not in message, message
