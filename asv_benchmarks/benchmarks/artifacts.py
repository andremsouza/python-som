"""Saving and loading a trained map.

Not a hot path, and tracked anyway: this is the boundary where a map becomes a file someone keeps,
and the format deliberately refuses pickle. A change that made loading slow enough to discourage
saving would quietly push people back toward ``pickle.dump``, which is arbitrary code execution on
load. Cheap to measure, so it is measured.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import python_som

from .common import FEATURES, SHAPES, som


class Artifacts:
    """One save and one load per timed run, on a real temporary file."""

    params = (SHAPES, FEATURES)
    param_names = ("shape", "n_features")

    som: python_som.SOM
    directory: Path
    path: Path

    def setup(self, shape: tuple[int, int], n_features: int) -> None:
        """Build a map and a temporary directory, and write one file to load from.

        :param shape: Grid shape.
        :param n_features: Number of features.
        """
        self.som = som(shape, n_features)
        self.directory = Path(tempfile.mkdtemp(prefix="som-asv-"))
        self.path = self.directory / "map.npz"
        self.som.save_npz(self.path)

    def teardown(self, shape: tuple[int, int], n_features: int) -> None:
        """Remove the temporary directory.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        shutil.rmtree(self.directory, ignore_errors=True)

    def time_save(self, shape: tuple[int, int], n_features: int) -> None:
        """Time writing a map, models and metadata together.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        self.som.save_npz(self.directory / "written.npz")

    def time_load(self, shape: tuple[int, int], n_features: int) -> None:
        """Time reading one back, including the metadata validation.

        :param shape: Unused.
        :param n_features: Unused.
        """
        del shape, n_features
        python_som.SOM.load_npz(self.path)
