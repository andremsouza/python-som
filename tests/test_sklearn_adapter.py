"""The scikit-learn adapter, tested by handing it to scikit-learn rather than by asserting about it.

The earlier plan for this claimed that duck typing was enough and that inheriting ``BaseEstimator``
was unnecessary. That was wrong: since scikit-learn 1.7, ``Pipeline.predict``, ``GridSearchCV`` and
``cross_val_score`` reach for ``__sklearn_tags__`` and raise ``AttributeError`` without it. That
claim had been reasoned about rather than run.

So every test here calls the real scikit-learn. The five integration points that failed under duck
typing each get one, because those are the claims that were wrong, and a claim that was wrong once
is worth checking on every run rather than describing again.

Skipped wholesale when scikit-learn is absent, since the adapter is behind an optional extra.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import python_som
from python_som import Neighborhood, TrainingMode
from python_som.sklearn import SOMEstimator

#: Fixed so a search's choice is reproducible.
SEED = 20260801


def _data(n_samples: int = 60, n_features: int = 4) -> np.ndarray:
    """Build a reproducible dataset.

    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :return: The dataset.
    """
    return np.random.default_rng(SEED).normal(size=(n_samples, n_features))


# ---------------------------------------------------------------------------------------------
# The five points that duck typing could not reach
# ---------------------------------------------------------------------------------------------


def test_clone_reproduces_the_parameters() -> None:
    """``clone`` rebuilds from ``get_params`` and requires the result to compare equal.

    This is why ``__init__`` stores every argument unmodified: validating or deriving anything there
    would make the rebuilt estimator differ from the original and break cloning, which is what
    ``GridSearchCV`` does to every candidate.
    """
    original = SOMEstimator(x=5, y=4, neighborhood_radius=2.5, random_seed=7)
    copied = clone(original)
    assert copied.get_params() == original.get_params()
    assert copied is not original


def test_pipeline_fit_and_predict() -> None:
    """``Pipeline.predict`` is the first thing duck typing failed."""
    data = _data()
    pipeline = Pipeline([("scale", StandardScaler()), ("som", SOMEstimator(x=5, y=4))])
    pipeline.fit(data)

    labels = pipeline.predict(data)
    assert labels.shape == (len(data),)
    assert labels.max() < 5 * 4


def test_pipeline_transform() -> None:
    """One column per node, after the scaler has had its way with the data."""
    data = _data()
    pipeline = Pipeline([("scale", StandardScaler()), ("som", SOMEstimator(x=5, y=4))]).fit(data)
    assert pipeline.transform(data).shape == (len(data), 5 * 4)


def test_grid_search_selects_the_better_map() -> None:
    """A search must tell a good map from a bad one, which is what the sign on ``score`` is for.

    A 2x2 grid cannot represent this data as well as a 6x6 one, so the search should prefer the
    larger. If ``score`` were not negated this test would fail, and it is what would catch that.
    """
    search = GridSearchCV(SOMEstimator(), {"x": [2, 6], "y": [2, 6]}, cv=3)
    search.fit(_data())
    assert search.best_params_ == {"x": 6, "y": 6}


def test_cross_val_score_runs_and_returns_one_score_per_fold() -> None:
    scores = cross_val_score(SOMEstimator(x=4, y=4), _data(), cv=3)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert (scores < 0).all(), "the score is a negated error, so every fold should be negative"


def test_fit_predict_comes_from_the_cluster_mixin() -> None:
    """Inheriting ``ClusterMixin`` is what provides it, and it must agree with fit-then-predict."""
    data = _data()
    combined = SOMEstimator(x=5, y=4, random_seed=1).fit_predict(data)
    separate = SOMEstimator(x=5, y=4, random_seed=1).fit(data).predict(data)
    np.testing.assert_array_equal(combined, separate)


# ---------------------------------------------------------------------------------------------
# The estimator contract
# ---------------------------------------------------------------------------------------------


def test_fitted_attributes_follow_the_convention() -> None:
    """Trailing underscore, and set only by ``fit``."""
    estimator = SOMEstimator(x=5, y=4)
    assert not hasattr(estimator, "weights_")

    estimator.fit(_data())
    assert estimator.weights_.shape == (5, 4, 4)
    assert estimator.n_features_in_ == 4
    assert estimator.quantization_error_ > 0
    assert estimator.labels_.shape == (60,)
    assert isinstance(estimator.som_, python_som.SOM)


def test_input_len_is_inferred_from_the_data() -> None:
    """scikit-learn infers the feature count, so it is not a constructor argument.

    A caller who changes their feature count should not have to remember to change a second number
    that has to agree with it.
    """
    assert "input_len" not in SOMEstimator().get_params()
    for n_features in (2, 7):
        fitted = SOMEstimator(x=4, y=4).fit(_data(n_features=n_features))
        assert fitted.n_features_in_ == n_features
        assert fitted.weights_.shape == (4, 4, n_features)


def test_refitting_starts_over_rather_than_continuing() -> None:
    """The opposite of ``SOM.fit``, and deliberately so.

    ``GridSearchCV`` fits one cloned estimator on fold after fold. If weights carried over, every
    fold would inherit the previous one and the scores would be quietly meaningless.
    """
    data = _data()
    estimator = SOMEstimator(x=5, y=4, random_seed=2)
    first = estimator.fit(data).weights_.copy()
    second = estimator.fit(data).weights_
    np.testing.assert_array_equal(first, second)


def test_the_report_is_none_before_fitting() -> None:
    """None rather than absent, so reading it before fitting is not an AttributeError."""
    assert SOMEstimator(x=4, y=4).report_ is None


def test_the_report_is_available_after_fitting() -> None:
    """Separate from the test above so neither narrows the other's type for a checker."""
    report = SOMEstimator(x=4, y=4, n_iteration=12).fit(_data()).report_
    assert report is not None
    assert report.n_iteration == 12


def test_every_neighborhood_and_mode_combination_that_is_legal_works() -> None:
    """The adapter must not narrow what the core accepts."""
    data = _data()
    for neighborhood in Neighborhood:
        for mode in TrainingMode:
            if neighborhood is Neighborhood.MEXICAN_HAT and mode is TrainingMode.BATCH:
                continue  # rejected by the core, and the core's error is the right one
            fitted = SOMEstimator(
                x=4, y=4, neighborhood_function=neighborhood, mode=mode, n_iteration=5
            ).fit(data)
            assert np.isfinite(fitted.weights_).all()


def test_a_plain_string_option_works_here_too() -> None:
    """Strings are permanent, and the adapter is not a place they stop being accepted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = SOMEstimator(x=4, y=4, neighborhood_function="bubble", mode="batch").fit(_data())
    assert fitted.n_features_in_ == 4


def test_the_core_error_surfaces_unchanged() -> None:
    """The adapter adds no validation of its own, so the core's message is what a caller sees."""
    with pytest.raises(ValueError, match="cannot be used with the 'batch' training mode"):
        SOMEstimator(
            x=4, y=4, neighborhood_function=Neighborhood.MEXICAN_HAT, mode=TrainingMode.BATCH
        ).fit(_data())


# ---------------------------------------------------------------------------------------------
# The boundary: the core must not depend on this
# ---------------------------------------------------------------------------------------------


def test_importing_python_som_does_not_import_sklearn() -> None:
    """The whole point of putting the adapter in its own module.

    A subprocess, because scikit-learn is certainly already imported in this one.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = (
        "import sys, python_som; "
        "assert 'sklearn' not in sys.modules, sorted(m for m in sys.modules if 'sklearn' in m); "
        "print('clean')"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


def test_the_adapter_is_not_reachable_from_the_package_root() -> None:
    """``python_som.sklearn`` is an explicit import, never something you get by accident.

    If it were re-exported from ``__init__``, importing the package would import scikit-learn and
    the numpy-only promise would be gone.
    """
    assert "sklearn" not in python_som.__all__
    assert not hasattr(python_som, "SOMEstimator")
