"""This module contains the implementation of the 2D self-organizing map.

Most features were implemented using NumPy, with Scikit-learn for standardization and PCA.

Features:
    - Stepwise and batch training
    - Random weight initialization
    - Random sampling weight initialization
    - Linear weight initialization (with PCA)
    - Automatic selection of map size ratio (with PCA)
    - Support for cyclic arrays, for toroidal or spherical maps
    - Gaussian, Bubble and Mexican Hat neighborhood functions
    - Support for custom decay functions
    - Support for visualization (U-matrix, activation matrix)
    - Support for supervised learning (label map)
    - Support for NumPy arrays, Pandas DataFrames and regular lists of values

Reference:
Teuvo Kohonen,
Essentials of the self-organizing map,
Neural Networks,
Volume 37,
2013,
Pages 52-65,
ISSN 0893-6080,
https://doi.org/10.1016/j.neunet.2012.09.018.
"""

# %%
from collections import Counter
from typing import Callable

import numpy as np
import pandas as pd
import sklearn  # type: ignore
import sklearn.decomposition  # type: ignore
import sklearn.preprocessing  # type: ignore

try:
    import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# %%
def _asymptotic_decay(x: float, t: int, max_t: int) -> float:
    """
    Asymptotic decay function. Can be used for both the learning_rate or the neighborhood_radius.

    :param x: float: Initial x parameter
    :param t: int: Current iteration
    :param max_t: int: Maximum number of iterations
    :return: float: Current state of x after t iterations
    """
    return x / (1 + t / (max_t / 2))


def _linear_decay(x: float, t: int, max_t: int) -> float:
    """
    Linear decay function. Can be used for both the learning_rate or the neighborhood_radius.

    :param x: float: Initial x parameter
    :param t: int: Current iteration
    :param max_t: int: Maximum number of iterations
    :return: float: Current state of x after t iterations
    """
    return x * (1.0 - t / max_t)


def _exponential_decay(x: float, t: int, max_t: int, factor: float = 2.0) -> float:
    """
    Exponential decay function. Can be used for both the learning_rate or the neighborhood_radius.

    :param x: float: Initial x parameter
    :param t: int: Current iteration
    :param max_t: int: Maximum number of iterations
    :param factor: float: Exponential decay factor. Defaults to 2.0.
    :return: float: Current state of x after t iterations
    """
    return x * (1 - (factor / max_t)) ** t


def _inverse_decay(x: float, t: int, max_t: int) -> float:
    """
    Inverse decay function. Can be used for both the learning_rate or the neighborhood_radius.

    :param x: float: Initial x parameter
    :param t: int: Current iteration
    :param max_t: int: Maximum number of iterations
    :return: float: Current state of x after t iterations
    """
    return (max_t / 100) * x / ((max_t / 100) + t)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """This function calculates the euclidean distances between the elements of the last dimension
         of a and b.

    The parameters a and b may be n-dimensional, but their shapes must be capable of broadcasting

    The shape of the output complies to the first n-1 dimensions of the parameter with the highest
        dimensionality.

    :param a: array-like: list or numpy array of values. a must not be a scalar value.
    :param b: array-like: list or numpy array of values. b must not be a scalar value.
    :return: array-like: An array of euclidean distances between a and b.
    """
    return np.linalg.norm(np.subtract(a, b), ord=2, axis=-1)


class SOM:
    """Implementation of the 2D self-organizing map, with support for NumPy arrays and
        Pandas DataFrames.

    Most features were implemented using NumPy, with Scikit-learn for standardization and
        PCA operations.

    Features:
        - Stepwise and batch training
        - Random weight initialization
        - Random sampling weight initialization
        - Linear weight initialization (with PCA)
        - Automatic selection of map size ratio (with PCA)
        - Support for cyclic arrays, for toroidal or spherical maps
        - Gaussian, Bubble and Mexican hat neighborhood functions
        - Support for custom decay functions
        - Support for visualization (U-matrix, activation matrix)
        - Support for supervised learning (label map)
        - Support for NumPy arrays, Pandas DataFrames and regular lists of values

    Reference:
    Teuvo Kohonen,
    Essentials of the self-organizing map,
    Neural Networks,
    Volume 37,
    2013,
    Pages 52-65,
    ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2012.09.018.
    """

    def __init__(
        self,
        x: int | None,
        y: int | None,
        input_len: int,
        learning_rate: float = 0.5,
        learning_rate_decay: Callable[[float, int, int], float] = _asymptotic_decay,
        neighborhood_radius: float = 1.0,
        neighborhood_radius_decay: Callable[
            [float, int, int], float
        ] = _asymptotic_decay,
        neighborhood_function: str = "gaussian",
        distance_function: Callable[
            [np.ndarray, np.ndarray],
            np.ndarray,
        ] = _euclidean_distance,
        cyclic_x: bool = False,
        cyclic_y: bool = False,
        random_seed: int | None = None,
        data: np.ndarray | pd.DataFrame | list | None = None,
    ) -> None:
        """
        Constructor for the self-organizing map class.

        :param x: int or NoneType: X dimension of the self-organizing map, i.e.,
            number of rows of the matrix of weights.
            x should be larger than 0.
            If x is None and 'data' is provided in kwargs, its value will be automatically
            selected using PCA of 'data'. Either x or y should be different than None.
        :param y: int or NoneType: Y dimension of the self-organizing map, i.e.,
            number of columns of the matrix of weights.
            y should be larger than 0.
            If y is None and 'data' is provided in kwargs, its value will be automatically
            selected using PCA of 'data'. Either x or y should be different than None.
        :param input_len: int: Number of features of the training dataset, i.e.,
            number of elements of each node of the network.
        :param learning_rate: float: Initial learning rate for the training process.
            Should be a positive floating point value.
            Defaults to 0.5.
            Note: The value of the learning_rate is irrelevant for the 'batch' training mode.
        :param learning_rate_decay: function: Decay function for the learning_rate variable.
            May be a predefined one from this package, or a custom function, with the same
            parameters and return type. Defaults to _asymptotic_decay.
        :param neighborhood_radius: float: Initial neighborhood radius for the training process.
            Defaults to 1.
        :param neighborhood_radius_decay: function: Decay function for the neighborhood_radius
            variable. May be a predefined one from this package, or a custom function, with the
            same parameters and return type. Defaults to _asymptotic_decay
        :param neighborhood_function: str: Neighborhood function name for the training process.
            May be either 'gaussian', 'bubble' or 'mexicanhat'.
        :param distance_function: function: Function for calculating distances/dissimilarities
            between models of the network.
            May be a predefined one from this package, or a custom function, with the same
            parameters and return type. Defaults to _euclidean_distance.
        :param cyclic_x: bool: Boolean value activate/deactivate cyclic arrays in the x direction,
            i.e, between the first and last rows of the weight matrix.
            Defaults to False.
        :param cyclic_y: bool: Boolean value activate/deactivate cyclic arrays in the y direction,
            i.e, between the first and last columns of the weight matrix.
            Defaults to False.
        :param random_seed: int or None: Seed for NumPy random value generator. Defaults to None.
        :param data: array-like: dataset for performing PCA.
            Required when either x or y is None, for determining map size.
        """
        # Verifying map dimensions (initializing automatically, if a dataset is provided)
        if (x, y) == (None, None):
            raise ValueError("At least one of the dimensions (x, y) must be specified")
        if x is None or y is None:
            # If a dataset was given through **kwargs, select missing dimension with PCA
            # The ratio of the (x, y) sizes will comply roughly with the ratio of
            # the two largest principal components
            if data is None:
                raise ValueError(
                    "If one of the dimensions is not specified,"
                    "a dataset must be provided for automatic size initialization."
                )
            # Convert data to numpy array
            if isinstance(data, pd.DataFrame):
                data_array = data.to_numpy()
            else:
                data_array = np.array(data)
            # Perform PCA w/ sklearn
            data_array = sklearn.preprocessing.StandardScaler().fit_transform(
                data_array
            )
            pca = sklearn.decomposition.PCA(n_components=2)
            pca.fit(data_array)
            ratio = pca.explained_variance_[0] / pca.explained_variance_[1]
            # Update missing size variable
            if x is None:
                x = y // ratio
            if y is None:
                y = x // ratio

        # Initializing private variables
        self._shape = (np.uint(x), np.uint(y))
        self._input_len = np.uint(input_len)
        self._learning_rate = float(learning_rate)
        self._learning_rate_decay = learning_rate_decay
        self._neighborhood_radius = float(neighborhood_radius)
        self._neighborhood_radius_decay = neighborhood_radius_decay
        neighborhood_functions = {
            "gaussian": self._gaussian,
            "bubble": self._bubble,
            "mexicanhat": self._mexicanhat,
        }
        try:
            self._neighborhood_function = neighborhood_functions[neighborhood_function]
        except KeyError as exc:
            raise ValueError(
                "Invalid value for 'neighborhood_function' parameter. Value should be in "
                + str(list(neighborhood_functions.keys()))
            ) from exc
        self._neighborhood_function_name = neighborhood_function
        self._distance_function = distance_function
        self._cyclic = (bool(cyclic_x), bool(cyclic_y))
        self._neigx, self._neigy = np.arange(self._shape[0]), np.arange(self._shape[1])

        # Seed numpy random generator
        if random_seed is None:
            self._random_seed = np.random.randint(np.iinfo(np.int32).max)
        else:
            self._random_seed = int(random_seed)
        np.random.seed(self._random_seed)

        # Random weight initialization
        self._weights = np.random.standard_normal(
            size=(self._shape[0], self._shape[1], self._input_len)
        )

    def get_shape(self) -> tuple[np.uint, np.uint]:
        """
        Gets the shape of the network.

        :return: tuple(int, int): Shape of the network.
        """
        return self._shape

    def get_weights(self) -> np.ndarray:
        """
        Gets the weight matrix of the network.

        :return: np.ndarray: Weight matrix of the network.
        """
        return self._weights

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Sets the learning_rate member of the SOM.

        :param learning_rate: float: New value for learning_rate of an instance of the SOM.
        """
        self._learning_rate = float(learning_rate)

    def set_neighborhood_radius(self, neighborhood_radius: float) -> None:
        """
        Sets the neighborhood_radius member of the SOM.

        :param neighborhood_radius: float: New value for neighborhood_radius of a SOM instance.
        """
        self._neighborhood_radius = float(neighborhood_radius)

    def activate(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates distances between an instance x and the weights of the network.

        :param x: array-like: Instance to be compared with the weights of the network.
        :return: np.ndarray: Distances between x and each weight of the network.
        """
        return self._distance_function(x, self._weights)

    def winner(self, x: np.ndarray) -> tuple[int, ...]:
        """
        Calculates the best-matching unit of the network for an instance x

        :param x: array-like: Instance to be compared with the weights of the network.
        :return: (int, int): Index of the best-matching unit of x.
        """
        activation_map = self.activate(x)
        min_index = tuple(
            map(int, np.unravel_index(activation_map.argmin(), activation_map.shape))
        )
        return min_index

    def quantization(self, data: np.ndarray | pd.DataFrame | list) -> np.ndarray:
        """
        Calculates distances from each instance of 'data' to each of the weights of the network.

        :param data: array-like: Dataset to be compared with the weights of the network.
            Expected shape is (n_samples, n_features).
        :return: np.ndarray: array of lists of distances from each instance of the dataset
            to each weight of the network.
        """
        # Convert data to numpy array
        data_array = self._data_to_numpy(data)
        return np.array(
            [
                (self._distance_function(i, self._weights[self.winner(i)]))
                for i in data_array
            ]
        )

    def quantization_error(self, data: np.ndarray | pd.DataFrame | list) -> float:
        """Calculates average distance of the weights of the network to their assigned instances.
        This error is a quality measure for the training process.

        :param data: array-like: Dataset to be compared with the weights of the network.
        :return: float: Quantization error.
        """
        quantization = self.quantization(data)
        return quantization.mean()

    def distance_matrix(self, normalize: bool = False) -> np.ndarray:
        """
        Calculates U-matrix of the current state of the network,
        i.e., the matrix of distances between each node and its neighbors.
        Has support for cyclic arrays

        :param normalize: bool: Activate to normalize the U-matrix between 0 and 1.
            Defaults to False.
        :return: np.ndarray: U-matrix of the current state of the network.
        """
        um = np.zeros(shape=self._shape)
        it = np.nditer(um, flags=["multi_index"])
        distances = np.zeros(self._shape + self._shape)
        for i in range(self._shape[0]):
            for j in range(self._shape[1]):
                distances[i, j] = self._distance_function(
                    self._weights[i, j], self._weights
                )

        while not it.finished:
            update_matrix = self._bubble(it.multi_index, 1)
            um[it.multi_index] = np.sum(update_matrix * distances[it.multi_index])
            it.iternext()
        if normalize:
            # Normalize U-matrix between 0 and 1
            um = (um - np.min(um)) / (np.max(um) - np.min(um))
        return um

    def activation_matrix(self, data: np.ndarray | pd.DataFrame | list) -> np.ndarray:
        """Calculates the activation matrix of the network for a dataset.

        I.e., for each node, the count of instances that have been assigned to in,
            in the current state.

        :param data: array-like: Dataset to be compared with the weights of the network.
        :return: np.ndarray: Activation matrix.
        """
        # Convert data to numpy array
        data_array = self._data_to_numpy(data)

        activation_matrix = np.zeros(self._shape)
        for i in data_array:
            activation_matrix[self.winner(i)] += 1
        return activation_matrix

    def winner_map(self, data: np.ndarray | pd.DataFrame | list) -> dict:
        """
        Calculates, for each node (i, j) of the network,
        the list of all instances from 'data' that has been assigned to it.

        :param data: array-like: Dataset to be compared with the weights of the network.
        :return: dict: Winner map.
        """
        # Convert data to numpy array
        data_array = self._data_to_numpy(data)
        winner_map: dict[tuple[int, ...], list] = {
            tuple(index): [] for index in np.ndindex(self._shape)
        }
        for i in data_array:
            winner_map[self.winner(i)].append(i)
        return winner_map

    def label_map(
        self,
        data: np.ndarray | pd.DataFrame | list,
        labels: np.ndarray | pd.DataFrame | list,
    ) -> dict[tuple[int, ...], Counter]:
        """Calculates, for each node (i, j) of the network, the frequency of each label...

        from 'labels' corresponding to its respective instance from 'data' that has been assigned
            to this node.

        :param data: array-like: Dataset to be compared with the weights of the network.
        :param labels: array-like: Labels corresponding to the indices of 'data'.
        :return: dict: Label map.
        """
        # Convert data to numpy array
        data_array = self._data_to_numpy(data)
        # Convert labels to numpy array
        if isinstance(labels, pd.DataFrame):
            labels = labels.to_numpy()
        else:
            labels = np.array(labels)

        winner_map: dict[tuple[int, ...], list] = {
            tuple(index): [] for index in np.ndindex(self._shape)
        }
        label_count_map: dict[tuple[int, ...], Counter] = {
            tuple(index): Counter() for index in np.ndindex(self._shape)
        }
        for i, instance in enumerate(data_array):
            winner = self.winner(instance)
            winner_map[winner].append(labels[i])
            label_count_map[winner].update([labels[i]])
        return label_count_map

    def train(
        self,
        data: np.ndarray | pd.DataFrame | list,
        n_iteration: int | None = None,
        mode: str = "random",
        verbose: bool = False,
    ) -> float:
        """
        Trains the self-organizing map, with the dataset 'data', and a certain number of iterations.

        :param data: array-like: Dataset for training.
        :param n_iteration: int or None: Number of iterations of training.
            If None, defaults to 1000 * len(data) for stepwise training modes,
            or 10 * len(data) for batch training mode.
        :param mode: str: Training mode name. May be either 'random', 'sequential', or 'batch'.
            For 'batch' mode, a much smaller number of iterations is needed,
            but a higher computation power is required for each individual iteration.
        :param verbose: bool: Activate to print useful information to the terminal/console, e.g.,
            the progress of the training process
        :return: float: Quantization error after training
        """
        # Convert data to numpy array for training
        data_array = self._data_to_numpy(data)

        # If no number of iterations is given, select automatically
        if n_iteration is None:
            n_iteration = {"random": 1000, "sequential": 1000, "batch": 10}[mode] * len(
                data_array
            )

        if verbose:
            print(
                "Training with",
                n_iteration,
                "iterations.\nTraining mode:",
                mode,
                sep=" ",
            )

        if mode == "random":
            self._train_random(data_array, n_iteration, verbose)
        elif mode == "sequential":
            self._train_sequential(data_array, n_iteration, verbose)
        elif mode == "batch":
            if self._neighborhood_function_name == "mexicanhat":
                # The batch update of Kohonen (2013), Eq. (8), is a weighted mean whose denominator
                # is sum_j n_j * h_ji. That is only well defined for a non-negative neighborhood
                # function. The mexican hat is signed by construction, so the denominator can vanish
                # or turn negative, which makes the update blow up or invert the sign of the
                # correction. Use a stepwise mode instead.
                raise ValueError(
                    "The 'mexicanhat' neighborhood function cannot be used with the 'batch'"
                    " training mode: the weighted mean of Kohonen (2013), Eq. (8), is undefined"
                    " for a neighborhood function that takes negative values, as its denominator"
                    " is not sign-definite. Use mode='random' or mode='sequential' instead."
                )
            self._train_batch(data_array, n_iteration, verbose)
        else:
            # Invalid training mode value
            raise ValueError(
                "Invalid value for 'mode' parameter. Value should be in "
                + str(["random", "sequential", "batch"])
            )

        # Compute quantization error
        q_error = self.quantization_error(data_array)
        if verbose:
            print("Quantization error:", q_error, sep=" ")
        return q_error

    def _train_random(
        self, data_array: np.ndarray, n_iteration: int, verbose: bool
    ) -> None:
        if TQDM_AVAILABLE and verbose:
            iterator: enumerate | tqdm.tqdm = tqdm.tqdm(
                np.random.choice(
                    len(data_array),
                    size=n_iteration,
                    replace=(n_iteration > len(data_array)),
                ),
                total=n_iteration,
                desc="Training",
            )
        else:
            iterator = enumerate(
                np.random.choice(
                    len(data_array),
                    size=n_iteration,
                    replace=(n_iteration > len(data_array)),
                )
            )

        for it, i in iterator:  # type: ignore
            # Calculating decaying alpha and sigma parameters for updating weights
            alpha = self._learning_rate_decay(self._learning_rate, it, n_iteration)
            sigma = self._neighborhood_radius_decay(
                self._neighborhood_radius, it, n_iteration
            )

            # Finding winner node (best-matching unit)
            winner = self.winner(data_array[i])

            # Updating weights, based on current neighborhood function
            self._weights += (
                alpha
                * self._neighborhood_function(winner, sigma)[..., None]
                * (data_array[i] - self._weights)
            )

            # Print progress, if verbose is activated and tqdm is not available
            if verbose and not TQDM_AVAILABLE:
                print(
                    "Iteration:",
                    it,
                    "/",
                    n_iteration,
                    sep=" ",
                    end="\r",
                    flush=True,
                )

    def _train_sequential(
        self, data_array: np.ndarray, n_iteration: int, verbose: bool
    ) -> None:
        if TQDM_AVAILABLE and verbose:
            iterator: enumerate | tqdm.tqdm = tqdm.tqdm(
                enumerate(data_array),
                total=n_iteration,
                desc="Training",
            )
        else:
            iterator = enumerate(data_array)

        for it, i in iterator:  # type: ignore
            # Calculating decaying alpha and sigma parameters for updating weights
            alpha = self._learning_rate_decay(self._learning_rate, it, n_iteration)
            sigma = self._neighborhood_radius_decay(
                self._neighborhood_radius, it, n_iteration
            )

            # Finding winner node (best-matching unit)
            winner = self.winner(i)

            # Updating weights, based on current neighborhood function
            self._weights += (
                alpha
                * self._neighborhood_function(winner, sigma)[..., None]
                * (i - self._weights)
            )

            # Print progress, if verbose is activated and tqdm is not available
            if verbose and not TQDM_AVAILABLE:
                print(
                    "Iteration:",
                    it,
                    "/",
                    n_iteration,
                    sep=" ",
                    end="\r",
                    flush=True,
                )

    def _train_batch(
        self, data_array: np.ndarray, n_iteration: int, verbose: bool
    ) -> None:
        if TQDM_AVAILABLE and verbose:
            iterator: range | tqdm.tqdm = tqdm.tqdm(
                range(n_iteration), total=n_iteration, desc="Training"
            )
        else:
            iterator = range(n_iteration)

        for it in iterator:
            # Calculating decaying sigma
            sigma = self._neighborhood_radius_decay(
                self._neighborhood_radius, it, n_iteration
            )

            # For each node, create a list of instances associated to it
            winner_map = self.winner_map(data_array)

            # Calculate the weighted average of all instances in the neighborhood of each node
            new_weights = np.zeros(self._weights.shape)
            for i in winner_map.keys():
                neig = self._neighborhood_function(i, sigma)
                upper, bottom = np.zeros(self._input_len), 0.0
                for j in winner_map.keys():
                    upper += neig[j] * np.sum(winner_map[j], axis=0)
                    bottom += neig[j] * len(winner_map[j])

                # Only update if there is any instance associated with the winner node
                # or its neighbors
                if bottom != 0:
                    new_weights[i] = upper / bottom

            # Update all nodes concurrently
            self._weights = new_weights

            # Print progress, if verbose is activated and tqdm is not available
            if verbose and not TQDM_AVAILABLE:
                print(
                    "Iteration:",
                    it,
                    "/",
                    n_iteration,
                    sep=" ",
                    end="\r",
                    flush=True,
                )

    def weight_initialization(
        self,
        mode: str = "random",
        **kwargs: np.ndarray | pd.DataFrame | list | str | int,
    ) -> None:
        """Function for weight initialization of the self-organizing map.

        Calls other methods for each initialization mode.

        :param mode: str: Initialization mode. May be either 'random', 'linear', or 'sample'.
            Note: Each initialization method may require multiple additional arguments in kwargs.
        :param kwargs:
            For 'random' initialization mode, 'sample_mode': str may be provided to determine
            the sampling mode. 'sample_mode' may be either 'standard_normal' (default) or 'uniform'
            For 'random' and 'sample' modes, 'random_seed': int may be provided for the random
            value generator. For 'sample' and 'linear' modes, 'data': array-like must be provided
            for sampling/PCA.
        """
        modes: dict[str, Callable[..., None]] = {
            "random": self._weight_initialization_random,
            "linear": self._weight_initialization_linear,
            "sample": self._weight_initialization_sample,
        }
        try:
            modes[mode](**kwargs)
        except KeyError as exc:
            raise ValueError(
                "Invalid value for 'mode' parameter. Value should be in "
                + str(modes.keys())
            ) from exc

    def _weight_initialization_random(
        self, sample_mode: str = "standard_normal", random_seed: int | None = None
    ) -> None:
        """Random initialization method.

        Assigns weights from a random distribution defined by 'sample_mode'.

        :param sample_mode: str: Distribution for random sampling.
            May be either 'uniform' or 'standard_normal'.
            Defaults to 'standard_normal'.
        :param random_seed: int or None: Seed for NumPy random value generator. Defaults to None.
        """
        sample_modes = {
            "uniform": np.random.random,
            "standard_normal": np.random.standard_normal,
        }

        # Seed numpy random generator
        if random_seed is None:
            random_seed = np.random.randint(np.iinfo(np.int32).max)
        else:
            random_seed = int(random_seed)
        np.random.seed(random_seed)

        # Initialize weights randomly
        try:
            self._weights = sample_modes[sample_mode](size=self._weights.shape)
        except KeyError as exc:
            raise ValueError(
                "Invalid value for 'sample_mode' parameter. Value should be in "
                + str(sample_modes.keys())
            ) from exc

    def _weight_initialization_linear(
        self, data: np.ndarray | pd.DataFrame | list
    ) -> None:
        """Linear initialization method.

        Assigns weights spanning the hyperplane formed by the two first principal
        components of 'data'.

        This is the recommended initialization method, as it may lead to a faster convergence.
        Unlike other initialization modes, this method is deterministic based on the input dataset.

        :param data: array-like: Dataset for weight initialization w/ PCA.
        """
        # Convert data to numpy array for training
        data_array = self._data_to_numpy(data)

        # Perform PCA w/ sklearn
        data_array = sklearn.preprocessing.StandardScaler().fit_transform(data_array)
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(data_array)
        # Initialize weights spanning first 2 principal components of data
        for i, c1 in enumerate(np.linspace(-1, 1, num=self._shape[0])):
            for j, c2 in enumerate(np.linspace(-1, 1, num=self._shape[1])):
                self._weights[i, j] = (
                    c1 * pca.explained_variance_[0] + c2 * pca.explained_variance_[1]
                )

    def _weight_initialization_sample(
        self,
        data: np.ndarray | pd.DataFrame | list,
        random_seed: int | None = None,
    ) -> None:
        """Initialization method. Assigns weights to random samples from an input dataset.

        :param data: Dataset for weight initialization/sampling.
        :param random_seed: int or None: Seed for NumPy random value generator. Defaults to None.
        """
        # Seed numpy random generator
        if random_seed is None:
            random_seed = np.random.randint(np.iinfo(np.int32).max)
        else:
            random_seed = int(random_seed)
        np.random.seed(random_seed)

        # Convert data to numpy array for training
        data_array = self._data_to_numpy(data)

        # Assign weights to random samples from dataset
        sample_size: np.uint = self._shape[0] * self._shape[1]
        sample = np.random.choice(
            len(data_array),
            size=int(sample_size),
            replace=bool(sample_size > len(data_array)),
        )
        self._weights = data_array[sample].reshape(self._weights.shape)

    @staticmethod
    def _axis_offsets(
        coords: np.ndarray, center: int, length: float, cyclic: bool
    ) -> np.ndarray:
        """
        Signed offsets from 'center' to each coordinate along one axis of the grid.

        When the axis is cyclic, the minimum-image convention is applied, i.e., each offset is
        folded into the interval [-length/2, length/2), so that the shortest way around the torus
        is used. Both tails must be folded: an offset of -9 on an axis of length 10 represents a
        distance of 1, not 9.

        :param coords: np.ndarray: Coordinates along the axis, i.e., np.arange(length).
        :param center: int: Coordinate of the center (winner node) along the axis.
        :param length: float: Number of nodes along the axis.
        :param cyclic: bool: Whether the axis wraps around.
        :return: np.ndarray: Signed offsets from center to each coordinate.
        """
        d = coords - center
        if cyclic:
            d = (d + length / 2) % length - length / 2
        return d

    def _sqdist(self, c: tuple[int, ...], sigma: float) -> np.ndarray:
        """
        Squared geometric distance from the center c to every node of the grid.

        This is the sqdist(c, i) term of Kohonen (2013), Eq. (5). Every neighborhood function must
        be a function of this scalar quantity: the neighborhood describes lateral interaction over
        a metric space, so it depends on the distance between two nodes and not on the individual
        axis offsets. Only the gaussian is separable into a product of per-axis terms, and that is
        a property of the exponential rather than of neighborhood functions in general.

        :param c: (int, int): Center coordinates.
        :param sigma: float: Spread variable, validated here for all neighborhood functions.
        :return: np.ndarray: Squared distances from c to each node, with the shape of the network.
        :raises ValueError: If sigma is not strictly positive.
        """
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(
                "The neighborhood radius 'sigma' must be a finite positive number, got "
                + str(sigma)
            )
        dx = self._axis_offsets(
            self._neigx, c[0], float(self._shape[0]), self._cyclic[0]
        )
        dy = self._axis_offsets(
            self._neigy, c[1], float(self._shape[1]), self._cyclic[1]
        )
        return np.add.outer(np.power(dx, 2), np.power(dy, 2))

    def _gaussian(self, c: tuple[int, ...], sigma: float) -> np.ndarray:
        """
        Gaussian neighborhood function, centered in c. Has support for cyclic arrays.

        Implements h_ci = exp(-sqdist(c, i) / (2 * sigma^2)), i.e., Eq. (5) of Kohonen (2013)
        with the learning rate factored out, so that h_cc = 1.

        :param c: (int, int): Center coordinates for gaussian function.
        :param sigma: float: Spread variable for gaussian function. Must be positive.
        :return: np.ndarray: Gaussian, centered in c, over all the weights of the network.
        :raises ValueError: If sigma is not strictly positive.
        """
        return np.exp(-self._sqdist(c, sigma) / (2 * sigma * sigma))

    def _bubble(self, c: tuple[int, ...], sigma: float) -> np.ndarray:
        """
        Bubble neighborhood function, centered in c. Has support for cyclic arrays.

        The neighbors of c are the nodes in the region of sigma positions in the vertical
        and horizontal directions around c.

        This is the truncated inner, excitatory lobe of the mexican hat lateral-interaction
        function; cf. Vrieze (1995), p. 85: "In Kohonen networks usually only the inner stimulation
        area is used, i.e., when a neuron i fires, a positive feedback takes place for all neurons
        i', whose distance to i is smaller than some given number rho."

        Unlike the other neighborhood functions, a radius of zero is admissible here: it selects
        the winner alone, which is well defined, whereas for the gaussian and the mexican hat a
        zero radius is a division by zero.

        :param c: (int, int): Center coordinates for gaussian function.
        :param sigma: float: Spread variable for gaussian function. Must be finite and
            non-negative.
        :return: np.ndarray: Neighborhood matrix, centered in c.
        :raises ValueError: If sigma is negative or not finite.
        """
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(
                "The neighborhood radius 'sigma' must be a finite non-negative number, got "
                + str(sigma)
            )
        # Convert sigma to integer
        sigma = int(np.around(sigma))

        # Calculate vertical and horizontal regions
        ax = np.logical_and(self._neigx >= c[0] - sigma, self._neigx <= c[0] + sigma)
        ay = np.logical_and(self._neigy >= c[1] - sigma, self._neigy <= c[1] + sigma)

        # Calculate cyclic regions
        if self._cyclic[0]:
            if c[0] - sigma < 0:
                ax[int(c[0] - sigma) :] = True
            if c[0] + sigma >= self._shape[0]:
                ax[: int((c[0] + sigma) % self._shape[0] + 1)] = True
        if self._cyclic[1]:
            if c[1] - sigma < 0:
                ay[int(c[1] - sigma) :] = True
            if c[1] + sigma >= self._shape[1]:
                ay[: int((c[1] + sigma) % self._shape[1] + 1)] = True

        return np.outer(ax, ay).astype(int)

    def _mexicanhat(self, c: tuple[int, ...], sigma: float) -> np.ndarray:
        """
        Mexican Hat neighborhood function, centered in c. Has support for cyclic arrays.

        Also known as the Ricker wavelet, or the Laplacian of Gaussian. This is the biologically
        motivated lateral-interaction function: nodes near the winner are excited, nodes beyond a
        certain distance are inhibited, and the inhibition vanishes as the distance grows further
        (Vrieze, 1995, Fig. 3).

        Implements the isotropic 2-D form as a function of u = sqdist(c, i) / (2 * sigma^2):

            h_ci = (1 - u) * exp(-u)

        normalized so that h_cc = 1. It crosses zero at a radius of sqrt(2) * sigma, reaches its
        minimum of -exp(-2) ~= -0.135 at a radius of 2 * sigma, and decays to zero thereafter.

        Note that this is *not* the outer product of two 1-D Ricker wavelets. Unlike the gaussian,
        the Ricker wavelet is not multiplicatively separable, and such a product is not a function
        of the distance between two nodes: it would be positive in the diagonal quadrants, where
        both 1-D factors are negative, placing a spurious excitatory lobe exactly where the
        function must inhibit. The neighborhood function must depend on the grid distance alone,
        cf. sqdist(c, i) in Kohonen (2013), Eq. (5), and the "lateral distance" abscissa of
        Vrieze (1995), Fig. 3.

        The '_bubble' neighborhood function is the truncated inner (excitatory) lobe of this same
        lateral-interaction function; cf. Vrieze (1995), p. 85.

        :param c: (int, int): Center coordinates for mexican hat function.
        :param sigma: float: Spread variable for mexican hat function. Must be positive.
        :return: np.ndarray: Mexican Hat, centered in c, over all the weights of the network.
        :raises ValueError: If sigma is not strictly positive.

        References:
        O. J. Vrieze, Kohonen network, in: Artificial Neural Networks: An Introduction to ANN
        Theory and Practice, Lecture Notes in Computer Science, vol. 931, Springer, 1995,
        pp. 83-100, https://doi.org/10.1007/BFb0027024.
        """
        u = self._sqdist(c, sigma) / (2 * sigma * sigma)
        return (1.0 - u) * np.exp(-u)

    def _data_to_numpy(self, data: np.ndarray | pd.DataFrame | list) -> np.ndarray:
        """
        Converts data to numpy array.

        :param data: array-like: Dataset for training.
        :return: np.ndarray: Numpy array of the dataset.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        return np.array(data)
