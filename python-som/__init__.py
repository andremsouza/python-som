# %%
import numpy as np
import pandas as pd


# %%
def _asymptotic_decay(x, t, max_t):
    return x / (1 + t / (max_t / 2))


def _euclidean_distance(a, b):
    return np.linalg.norm(a - b, ord=2, axis=max(len(a.shape), len(b.shape)) - 1)


def _generate_iterations(data, n):
    return np.random.choice(len(data), size=n, replace=(n > len(data)))


class SOM:
    def __init__(
            self,
            x: int,
            y: int,
            input_len: int,
            learning_rate: float = 0.5,
            learning_rate_decay=_asymptotic_decay,
            neighborhood_radius: float = 1.0,
            neighborhood_radius_decay=_asymptotic_decay,
            neighborhood_function: str = 'gaussian',
            distance_function=_euclidean_distance,
            random_seed: int = None,
    ):

        # Initializing private variables
        self._shape = (x, y)
        self._input_len = np.uint(input_len)
        self._learning_rate = np.float(learning_rate)
        self._learning_rate_decay = learning_rate_decay
        self._neighborhood_radius = np.float(neighborhood_radius)
        self._neighborhood_radius_decay = neighborhood_radius_decay
        self._neighborhood_function = {
            'gaussian': self._gaussian}[neighborhood_function]
        self._distance_function = distance_function

        # Seed numpy random generator
        if random_seed is None:
            self._random_seed = np.random.randint(
                np.random.randint(np.iinfo(np.int32).max))
        else:
            self._random_seed = np.int(self._random_seed)
        np.random.seed(self._random_seed)

        # Random weight initialization
        self._weights = np.random.standard_normal(
            size=(self._shape[0], self._shape[1], self._input_len))

    def train(self, data, n_iteration: int, mode: str = 'random', verbose: bool = False):
        # Convert data to numpy array for training
        if isinstance(data, pd.DataFrame):
            data_array = data.to_numpy()
        else:
            data_array = np.array(data)

        if verbose:
            print("Training with ", n_iteration,
                  " iterations.", "\n", "Training mode: ", mode)

        if mode == 'random':
            # Random sampling from training dataset
            for it, i in enumerate(_generate_iterations(data_array, n_iteration)):
                if verbose:
                    print("Iteration: ", it)
                # Finding winner node (best-matching unit)
                winner = self.winner(data_array[i])
                self._update_weights(data_array[i], winner, it, n_iteration)

    # def _exponential_decay(self, initial_radius, t, lambda_c):
    #     return initial_radius * np.exp(-t/lambda_c)

    def _gaussian(self, c, sigma):
        d = 2 * sigma * sigma
        hc = np.exp([- _euclidean_distance(i, c) ** 2 /
                     d for i in self._weights.reshape(np.prod(self._shape), -1)])
        return hc.reshape(self._shape)

    def winner(self, instance):
        return np.unravel_index(np.argmin(_euclidean_distance(self._weights, instance)), self._weights.shape[:-1])

    def _update_weights(self, x, winner, t, max_t):
        # Calculating decaying alpha and sigma parameters for updating weights
        alpha = self._learning_rate_decay(self._learning_rate, t, max_t)
        sigma = self._neighborhood_radius_decay(
            self._neighborhood_radius, t, max_t)

        # Updating weights, based on current neighborhood function
        self._weights += alpha * self._neighborhood_function(self._weights[winner], sigma)[..., None] * (
                x - self._weights)

    def weight_initialization(self, mode: str = 'random', **kwargs):
        modes = {'random': self._weight_initialization_random, 'linear': self._weight_initialization_linear,
                 'sample': self._weight_initialization_sample}
        if mode not in modes:
            raise ValueError("Invalid value for 'mode' parameter. Value should be in " + str(modes.keys()))
        modes[mode](**kwargs)

    def _weight_initialization_random(self, **kwargs):
        sample_modes = {'uniform': np.random.random, 'standard_normal': np.random.standard_normal}
        # Get sample_mode from kwargs, if exists
        try:
            if kwargs['sample_mode'] not in sample_modes:
                # If invalid value for sample_mode, raise exception
                raise ValueError(
                    "Invalid value for 'sample_mode' parameter. Value should be in " + str(sample_modes.keys()))
            sample_mode = sample_modes[kwargs['sample_mode']]
        except KeyError:
            sample_mode = sample_modes['standard_normal']

        # Get value for random_seed, if exists
        try:
            random_seed = kwargs['random_seed']
            np.random.seed(random_seed)
        except KeyError:
            pass

        # Initialize weights
        self._weights = sample_mode(size=self._weights.shape)

    def _weight_initialization_linear(self, **kwargs):
        # TODO: Get training dataset
        # TODO: Initialize weights spanning first N principal components
        pass

    def _weight_initialization_sample(self, **kwargs):
        # TODO: Get training dataset
        # TODO: Initialize weights as random samples from the training dataset
        pass

    def distance_matrix(self):
        # TODO: Generalize to N-dimensional maps
        um = np.zeros(shape=self._shape)
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for i in range(it.multi_index[0] - 1, it.multi_index[0] + 1 + 1):
                for j in range(it.multi_index[1] - 1, it.multi_index[1] + 1 + 1):
                    if 0 <= i < self._shape[0] and 0 <= j < self._shape[1]:
                        um[it.multi_index] += _euclidean_distance(self._weights[i, j, :], self._weights[it.multi_index])
            it.iternext()
        um /= um.max(initial=0.0)
        return um


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    iris = sns.load_dataset('iris')
    som = SOM(10, 10, iris.shape[1] - 1)
    som.train(data=iris.iloc[:, :-1], n_iteration=1000 * (len(iris) - 1), verbose=True)
    # Verify distance matrix
    um = som.distance_matrix().T
    plt.figure()
    ax = sns.heatmap(um, vmin=0.0, vmax=1.0, linewidths=0.1, cmap='BuGn_r')
    ax.invert_yaxis()
    plt.show()

# %%
