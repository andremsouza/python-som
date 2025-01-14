"""This module is an example of how to use the python_som package to train a SOM model with Iris.

The Iris dataset is loaded from seaborn library and the labels are transformed into numeric codes.
The SOM is trained with the Iris dataset and the U-matrix is plotted with matplotlib and seaborn.
The plot is saved to a .png file.
"""

# Import python_som
# Import libraries for plotting results
import matplotlib.pyplot as plt

# Import NumPy and Pandas for storing data
import numpy as np
import seaborn as sns  # type: ignore

import python_som

# Load Iris dataset and columns of features and labels
iris = sns.load_dataset("iris")
target = iris.iloc[:, -1].to_numpy()
iris = iris.iloc[:, :-1].to_numpy()
# Transform labels into numeric codes for plotting
tg = np.zeros(len(target), dtype=int)
tg[target == "setosa"] = 0
tg[target == "versicolor"] = 1
tg[target == "virginica"] = 2

# Instantiate SOM from  python_som
# Selecting shape automatically (providing dataset for constructor)
# Using default decay and distance functions
# Using gaussian neighborhood function
# Using cyclic arrays in the vertical and horizontal directions
som = python_som.SOM(
    x=20,
    y=None,
    input_len=iris.shape[1],
    learning_rate=0.5,
    neighborhood_radius=1.0,
    neighborhood_function="gaussian",
    cyclic_x=True,
    cyclic_y=True,
    data=iris,
)

# Initialize weights of the SOM with linear initialization
som.weight_initialization(mode="linear", data=iris)

# Training SOM with default number of iterations
# Using batch learning process
som.train(data=iris, n_iteration=len(iris), mode="batch", verbose=True)

# Calculating distance matrix for plotting
umatrix = som.distance_matrix().T

# Plotting U-matrix with seaborn/matplotlib
plt.figure(figsize=(float(som.get_shape()[0]), float(som.get_shape()[1])))
plt.pcolor(umatrix, cmap="bone_r")

markers = ["o", "s", "D"]
colors = ["C0", "C1", "C2"]
for cnt, xx in enumerate(iris):
    w = som.winner(xx)  # getting the winner
    plt.plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[tg[cnt]],
        markerfacecolor="None",
        markeredgecolor=colors[tg[cnt]],
        markersize=12,
        markeredgewidth=2,
    )
plt.axis((0, som.get_shape()[0], 0, som.get_shape()[1]))
# Saving the plot to .png
plt.savefig("test_output_iris.png")
