"""The packaging contract: typing marker, public surface, version and metadata."""

from __future__ import annotations

import pathlib
import sys
from importlib.metadata import metadata, version
from typing import Any

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import python_som

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    """Return the parsed ``pyproject.toml``."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_py_typed_marker_is_present() -> None:
    """Without this file, PEP 561 says downstream type checkers must ignore our annotations."""
    marker = pathlib.Path(python_som.__file__).parent / "py.typed"
    assert marker.is_file()


def test_installed_version_matches_pyproject(pyproject: dict[str, Any]) -> None:
    assert version("python-som") == pyproject["project"]["version"]


def test_dunder_version_matches_pyproject(pyproject: dict[str, Any]) -> None:
    assert python_som.__version__ == pyproject["project"]["version"]


def test_cli_extra_is_declared(pyproject: dict[str, Any]) -> None:
    """The ``cli`` extra shipped in 0.1.3 on PyPI but was never committed.

    Releasing from the repository without it would silently break
    ``pip install python-som[cli]`` for existing users.
    """
    assert "cli" in pyproject["project"]["optional-dependencies"]
    assert any(d.startswith("tqdm") for d in pyproject["project"]["optional-dependencies"]["cli"])


def test_requires_python_is_honest(pyproject: dict[str, Any]) -> None:
    """The package uses PEP 604 unions in runtime annotations, so 3.10 is the true floor."""
    assert pyproject["project"]["requires-python"] == ">=3.10"


def test_metadata_declares_typing_support() -> None:
    classifiers = metadata("python-som").get_all("Classifier") or []
    assert "Typing :: Typed" in classifiers


def test_no_deprecated_license_classifier(pyproject: dict[str, Any]) -> None:
    """PEP 639 replaces the classifier with the SPDX ``license`` field; having both conflicts."""
    classifiers = pyproject["project"]["classifiers"]
    assert not any(c.startswith("License ::") for c in classifiers)
    assert pyproject["project"]["license"] == "MIT"


def test_public_surface_is_importable() -> None:
    for name in python_som.__all__:
        assert hasattr(python_som, name), name


def test_som_is_constructible_from_the_package_root() -> None:
    som = python_som.SOM(x=4, y=4, input_len=2, random_seed=0)
    assert som.get_shape() == (4, 4)


@pytest.mark.parametrize(
    "name",
    [
        "_asymptotic_decay",
        "_linear_decay",
        "_exponential_decay",
        "_inverse_decay",
        "_euclidean_distance",
    ],
)
def test_pre_0_3_private_aliases_still_resolve(name: str) -> None:
    """These were underscore-prefixed but reachable, and the README listed them.

    Keeping them working avoids breaking imports that the old README effectively advertised.
    """
    assert callable(getattr(python_som, name))


def test_dev_dependencies_are_pinned_exactly(pyproject: dict[str, Any]) -> None:
    """Reproducible tooling: a floating dev pin means CI can break without a commit."""
    for group in ("dev", "docs"):
        for spec in pyproject["project"]["optional-dependencies"][group]:
            requirement = spec.split(";")[0].strip()
            assert "==" in requirement, f"{group} dependency is not pinned: {spec}"
