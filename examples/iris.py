"""Train a self-organizing map on the Iris dataset and plot its U-matrix.

Run with the ``examples`` extra installed::

    uv sync --all-extras
    uv run python examples/iris.py

Writes ``docs/assets/iris.png``, the image the README and the documentation display.
"""

from __future__ import annotations

import logging
import pathlib

import matplotlib as mpl

mpl.use("Agg")  # render to a file; no display needed

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]

import python_som

OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "assets" / "iris.png"
MARKERS = ("o", "s", "D")
COLORS = ("C0", "C1", "C2")


def main() -> None:
    """Train the map and write the plot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    iris = sns.load_dataset("iris")
    target = iris.iloc[:, -1].to_numpy()
    features = iris.iloc[:, :-1].to_numpy()

    # y is chosen from the data: Kohonen Section 3.5 asks for side lengths matching the lengths of
    # the two largest principal components.
    som = python_som.SOM(
        x=20,
        y=None,
        input_len=features.shape[1],
        learning_rate=0.5,
        neighborhood_radius=1.0,
        neighborhood_function="gaussian",
        cyclic_x=True,
        cyclic_y=True,
        data=features,
        random_seed=42,
    )
    print(f"map shape: {som.get_shape()}")

    # Linear initialization is the one Kohonen recommends, and the only deterministic one.
    som.weight_initialization(mode="linear", data=features)
    error = som.train(features, n_iteration=len(features), mode="batch", verbose=True)
    print(f"quantization error: {error:.4f}")

    umatrix = som.distance_matrix().T
    codes = {name: i for i, name in enumerate(np.unique(target))}

    # get_shape() returns plain ints, so it can be handed straight to figsize.
    plt.figure(figsize=som.get_shape())
    plt.pcolor(umatrix, cmap="bone_r")
    for sample, label in zip(features, target, strict=True):
        w = som.winner(sample)
        plt.plot(
            w[0] + 0.5,
            w[1] + 0.5,
            MARKERS[codes[label]],
            markerfacecolor="None",
            markeredgecolor=COLORS[codes[label]],
            markersize=12,
            markeredgewidth=2,
        )
    plt.axis((0, som.get_shape()[0], 0, som.get_shape()[1]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
