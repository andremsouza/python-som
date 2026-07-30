"""The estimator interface: fit, transform, predict, score, get_params, set_params.

These methods add no behaviour. They delegate to ``train``, ``activate`` and ``winner``, which by
Ousterhout's definition makes them pass-through methods and ordinarily a design smell. They exist
because their value is conformance to an external convention: scikit-learn's ``Pipeline``,
``GridSearchCV`` and ``cross_val_score`` call these exact names, and ``KMeans`` is the precedent a
SOM should follow, being a topologically-constrained k-means.

So the tests here are mostly about *agreement*: that ``fit`` does exactly what ``train`` does, that
``predict`` agrees with ``winner``, that ``transform`` agrees with ``activate``. A pass-through that
has drifted from what it passes through is the failure mode worth guarding.

Full scikit-learn integration needs more than these names and lives in ``python_som.sklearn``; its
tests are in ``tests/test_sklearn_adapter.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import python_som
from python_som import TrainingMode, WeightInit

#: Fixed so every comparison below is between two runs that should be identical.
SEED = 20260801


def _data(n_samples: int = 50, n_features: int = 4) -> np.ndarray:
    """Build a reproducible dataset.

    :param n_samples: Number of samples.
    :param n_features: Number of features.
    :return: The dataset.
    """
    return np.random.default_rng(SEED).normal(size=(n_samples, n_features))


def _som(**kwargs: object) -> python_som.SOM:
    """Build a small map with a fixed seed.

    :param kwargs: Passed to the constructor.
    :return: The map.
    """
    return python_som.SOM(x=6, y=5, input_len=4, random_seed=3, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# fit agrees with train
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(TrainingMode))
def test_fit_and_train_produce_identical_weights(mode: TrainingMode) -> None:
    """The invariant that keeps ``fit`` honest.

    ``fit`` is a one-line delegation, so the two must agree at exactly ``0.0``. If they ever differ,
    one of them has grown behaviour the other lacks, which is the whole risk of having two names for
    one operation.
    """
    data = _data()
    trained = _som()
    trained.weight_initialization(mode=WeightInit.LINEAR, data=data)
    trained.train(data, n_iteration=20, mode=mode)

    fitted = _som()
    fitted.weight_initialization(mode=WeightInit.LINEAR, data=data)
    fitted.fit(data, n_iteration=20, mode=mode)

    assert np.abs(fitted.get_weights() - trained.get_weights()).max() == 0.0


def test_fit_returns_the_same_object_not_a_copy() -> None:
    """``fit`` returns ``self`` so calls chain, which the convention requires."""
    som = _som()
    assert som.fit(_data(), n_iteration=5) is som


def test_fit_accepts_and_ignores_y() -> None:
    """Unsupervised estimators take ``y`` anyway, so Pipeline can call every step the same way."""
    data = _data()
    with_y = _som().fit(data, np.zeros(len(data)), n_iteration=10)
    without = _som().fit(data, n_iteration=10)
    assert np.abs(with_y.get_weights() - without.get_weights()).max() == 0.0


def test_fit_can_be_called_twice_and_continues() -> None:
    """Refitting is not forbidden, and it continues rather than resetting.

    Worth pinning because scikit-learn estimators conventionally *reset* on refit. This one does
    not, and that difference should be on the record rather than a surprise: a SOM's models are its
    state, and ``train`` has always continued from wherever they were.
    """
    data = _data()
    som = _som()
    som.fit(data, n_iteration=10)
    once = som.get_weights().copy()
    som.fit(data, n_iteration=10)
    assert np.abs(som.get_weights() - once).max() > 0.0


# ---------------------------------------------------------------------------------------------
# transform and predict agree with what they delegate to
# ---------------------------------------------------------------------------------------------


def test_transform_is_activate_per_sample() -> None:
    """One row per sample, one column per node, flattened in C order."""
    data = _data()
    som = _som().fit(data, n_iteration=10)
    transformed = som.transform(data)

    assert transformed.shape == (len(data), 6 * 5)
    for row, sample in zip(transformed, data, strict=True):
        np.testing.assert_array_equal(row, som.activate(sample).ravel())


def test_predict_is_winner_flattened() -> None:
    """A flat index, and ``unravel_index`` must recover exactly what ``winner`` returns."""
    data = _data()
    som = _som().fit(data, n_iteration=10)
    labels = som.predict(data)

    assert labels.shape == (len(data),)
    assert labels.dtype.kind == "i"
    rows, columns = np.unravel_index(labels, som.get_shape())
    for row, column, sample in zip(rows, columns, data, strict=True):
        assert (int(row), int(column)) == som.winner(sample)


def test_predict_stays_inside_the_grid() -> None:
    """A label outside ``x * y`` would silently break any confusion matrix built from it."""
    data = _data()
    som = _som().fit(data, n_iteration=10)
    labels = som.predict(data)
    assert labels.min() >= 0
    assert labels.max() < 6 * 5


def test_transform_and_predict_agree_with_each_other() -> None:
    """The nearest node by ``transform`` must be the node ``predict`` names."""
    data = _data()
    som = _som().fit(data, n_iteration=10)
    np.testing.assert_array_equal(som.transform(data).argmin(axis=1), som.predict(data))


def test_fit_transform_equals_fit_then_transform() -> None:
    data = _data()
    combined = _som().fit_transform(data, n_iteration=10)
    separate = _som().fit(data, n_iteration=10).transform(data)
    np.testing.assert_array_equal(combined, separate)


def test_the_estimator_methods_accept_a_dataframe() -> None:
    """They go through the same port as everything else, so anything array-like works."""
    pd = pytest.importorskip("pandas")
    data = _data()
    frame = pd.DataFrame(data, columns=[f"f{i}" for i in range(data.shape[1])])

    from_frame = _som().fit(frame, n_iteration=10)
    from_array = _som().fit(data, n_iteration=10)
    np.testing.assert_array_equal(from_frame.get_weights(), from_array.get_weights())
    np.testing.assert_array_equal(from_frame.predict(frame), from_array.predict(data))


# ---------------------------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------------------------


def test_score_is_the_negated_quantization_error() -> None:
    """The sign is load-bearing, not cosmetic.

    Every scikit-learn scorer treats larger as better and ``GridSearchCV`` maximises. Quantization
    error is something to minimise, so without the negation a parameter search would confidently
    select the *worst* map.
    """
    data = _data()
    som = _som().fit(data, n_iteration=10)
    assert som.score(data) == pytest.approx(-som.quantization_error(data))
    assert som.score(data) < 0


def test_a_better_map_scores_higher() -> None:
    """The property the sign exists to give, checked rather than assumed."""
    data = _data()
    trained = _som().fit(data, n_iteration=60, mode=TrainingMode.BATCH)
    untrained = _som()
    assert trained.score(data) > untrained.score(data)


# ---------------------------------------------------------------------------------------------
# get_params / set_params
# ---------------------------------------------------------------------------------------------


def test_get_params_reconstructs_an_equivalent_map() -> None:
    """What ``clone`` relies on: the returned dict must be accepted by the constructor."""
    som = _som(learning_rate=0.25, neighborhood_radius=2.5, cyclic_x=True)
    rebuilt = python_som.SOM(**som.get_params())

    assert rebuilt.get_params() == som.get_params()
    assert rebuilt.get_shape() == som.get_shape()
    np.testing.assert_array_equal(rebuilt.get_weights(), som.get_weights())


def test_get_params_covers_every_constructor_argument_that_shapes_a_map() -> None:
    """A missing key would make ``clone`` silently drop a setting.

    ``data`` is deliberately absent: it is a constructor argument only for automatic sizing, and the
    shape it produced is already reported as ``x`` and ``y``.
    """
    import inspect  # noqa: PLC0415

    accepted = set(inspect.signature(python_som.SOM.__init__).parameters) - {"self", "data"}
    assert set(_som().get_params()) == accepted


def test_set_params_changes_what_it_can_and_returns_self() -> None:
    som = _som()
    assert som.set_params(learning_rate=0.1, neighborhood_radius=3.0) is som
    assert som.get_params()["learning_rate"] == pytest.approx(0.1)
    assert som.get_params()["neighborhood_radius"] == pytest.approx(3.0)


def test_set_params_refuses_to_change_the_shape() -> None:
    """Changing ``x`` would leave a map whose models do not match its own description.

    Raising beats silently rebuilding: a caller who wants a different grid wants a different map,
    and quietly discarding trained weights is the kind of help nobody asks for.
    """
    som = _som().fit(_data(), n_iteration=5)
    for parameter in ("x", "y", "input_len"):
        with pytest.raises(ValueError, match="cannot be changed after construction"):
            som.set_params(**{parameter: 99})


def test_set_params_rejects_an_unknown_parameter() -> None:
    """A typo must not be accepted in silence, which plain attribute assignment would do."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        _som().set_params(learnign_rate=0.1)


def test_set_params_still_validates_the_learning_rate() -> None:
    """The constructor rejects a non-positive rate, so this must not be a way around that."""
    with pytest.raises(ValueError, match="'learning_rate' must be a finite positive number"):
        _som().set_params(learning_rate=-1.0)


def test_set_params_agrees_with_the_older_setters() -> None:
    """``set_learning_rate`` and ``set_neighborhood_radius`` become this. Both still work."""
    old, new = _som(), _som()
    old.set_learning_rate(0.3)
    old.set_neighborhood_radius(2.0)
    new.set_params(learning_rate=0.3, neighborhood_radius=2.0)
    assert old.get_params() == new.get_params()


# ---------------------------------------------------------------------------------------------
# The fitted-attribute convention
# ---------------------------------------------------------------------------------------------


def test_the_trailing_underscore_attributes_mirror_kmeans() -> None:
    """``weights_`` for ``cluster_centers_``, ``quantization_error_`` for ``inertia_``."""
    data = _data()
    som = _som().fit(data, n_iteration=15, mode=TrainingMode.BATCH)

    np.testing.assert_array_equal(som.weights_, som.get_weights())
    assert som.n_features_in_ == 4
    assert som.quantization_error_ == pytest.approx(som.quantization_error(data))


def test_quantization_error_is_none_before_training() -> None:
    """None rather than a number, so an untrained map cannot be mistaken for a bad one."""
    assert _som().quantization_error_ is None


def test_weights_is_available_before_training() -> None:
    """Unlike scikit-learn, the models exist from construction, since initialization is separate."""
    assert _som().weights_.shape == (6, 5, 4)
