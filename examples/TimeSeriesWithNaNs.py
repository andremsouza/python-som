# %%
import matplotlib.pyplot as plt
import numpy as np

import python_som

# %%
# Generate two sets of time series data with errors modeled after a possion distributions
# appropriate for count data. The first set of time series will be constant, while the second set
# will have underlying red noise.
rng = np.random.default_rng(42)
n_steps = 30

# Constant time series
n_constant_timeseries = 1000
mean_values = rng.uniform(low=0, high=10, size=n_constant_timeseries)
sigmas = np.sqrt(mean_values)
x_const = rng.normal(loc=mean_values, scale=sigmas, size=(30, n_constant_timeseries)).T
x_const_err = np.sqrt(abs(x_const))

i = rng.integers(low=0, high=n_constant_timeseries)
plt.errorbar(np.arange(n_steps), x_const[i], yerr=x_const_err[i], fmt="o")
plt.axhline(mean_values[i], color="k", ls=":")
plt.savefig("constant_timeseries_example.pdf", dpi=300)
plt.close()
# %%
# Non-constant time series with red noise
n_non_constant_timeseries = 1000
spectrum = np.zeros(n_steps, dtype=np.complex128)
indices = np.arange(1, n_steps - 1)
spectrum[indices] = 1 / indices**2
all_spectra = spectrum[np.newaxis, :].repeat(n_non_constant_timeseries, axis=0)
all_spectra[:, 0] = 1  # rng.uniform(low=1000, high=100000, size=n_constant_timeseries)
phases = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_non_constant_timeseries, n_steps)))
n = phases * all_spectra
x_true_rednoise = np.abs(np.fft.ifft(n, axis=1)) * 100
x_true_rednoise_err = np.sqrt(abs(x_true_rednoise))
x_rednoise = rng.normal(loc=x_true_rednoise, scale=x_true_rednoise_err)
x_rednoise_err = np.sqrt(abs(x_rednoise))

i = rng.integers(low=0, high=n_constant_timeseries)
plt.plot(np.arange(n_steps), x_true_rednoise[i], color="k", ls=":")
plt.errorbar(np.arange(n_steps), x_rednoise[i], yerr=x_rednoise_err[i], fmt="o")
plt.savefig("rednoise_timeseries_example.pdf", dpi=300)
plt.close()
# %%
# Combine the two sets of time series into one dataset
data = np.concatenate([x_const, x_rednoise], axis=0)
data_err = np.concatenate([x_const_err, x_rednoise_err], axis=0)
# %%
medians = np.median(data, axis=1)[:, np.newaxis]
normed_data = data / medians
normed_data_err = data_err / np.abs(medians)
# %%
# randomly drop epochs
n_exp_missing = 10
n_missing = rng.poisson(lam=n_exp_missing, size=data.shape[0])
for i, inm in enumerate(n_missing):
    if inm > 0:
        missing_indices = rng.choice(np.arange(n_steps), size=inm, replace=False)
        normed_data[i, missing_indices] = np.nan
        normed_data_err[i, missing_indices] = np.nan
# %%
X = normed_data

# Train a self-organizing map on the time series data
somsize = (10, 10)
state = np.random.RandomState(42)
som = python_som.SOM(
    x=somsize[0],
    y=somsize[1],
    input_len=n_steps,
    learning_rate=0.5,
    neighborhood_radius=1.0,
    neighborhood_function="gaussian",
    cyclic_x=True,
    cyclic_y=True,
    data=normed_data,
    random_seed=42,
)
som.fit(X, verbose=True, mode="batch")

win_map = np.array(np.unravel_index(som.predict(X), som.get_shape())).T

fig, axs = plt.subplots(*somsize, figsize=(7, 7))
for position in np.unique(win_map, axis=0):
    mask = (win_map[:, 0] == position[0]) & (win_map[:, 1] == position[1])
    if not any(mask):
        continue
    ax = axs[somsize[0] - 1 - position[0], position[1]] if somsize[1] > 1 else axs[position[0]]
    ax.plot(np.nanmean(normed_data[mask], axis=0), c="k")
    ax.fill_between(
        np.arange(n_steps),
        *np.nanquantile(normed_data[mask], [0.05, 0.95], axis=0),
        color="gray",
        alpha=0.5,
    )
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
fig.savefig("som_timeseries.pdf", dpi=300)
plt.close()

constants_counts = np.unique(
    som.predict(normed_data[:n_constant_timeseries]), return_counts=True, axis=0
)
constants_map = np.zeros(somsize)
for p, c in zip(
    np.array(np.unravel_index(constants_counts[0], som.get_shape())).T,
    constants_counts[1],
    strict=False,
):
    constants_map[p[0], p[1]] = c

rednoise_counts = np.unique(
    som.predict(normed_data[n_constant_timeseries:]), return_counts=True, axis=0
)
rednoise_map = np.zeros(somsize)
for p, c in zip(
    np.array(np.unravel_index(rednoise_counts[0], som.get_shape())).T,
    rednoise_counts[1],
    strict=False,
):
    rednoise_map[p[0], p[1]] = c

purity_map = rednoise_map / (constants_map + rednoise_map)
recall_map = rednoise_map / rednoise_map.sum()

fig, axs = plt.subplots(ncols=4, figsize=(20, 5))
for cmap, pmap, ax in zip(
    ["Reds", "Blues", "copper", "Reds"],
    [rednoise_map, constants_map, purity_map, recall_map],
    axs,
    strict=False,
):
    mesh = ax.pcolormesh(pmap, cmap=cmap)  # plotting the distance map as background
    fig.colorbar(mesh, ax=ax)
fig.savefig("som_timeseries_purity.pdf", dpi=300)
plt.close()

rednoise = rednoise_map.flatten()
constants = constants_map.flatten()
probs = purity_map.flatten()

recall = []
precision = []
xx = np.linspace(0, 1, 100)
for i in xx:
    m = probs >= i
    precision.append(rednoise[m].sum() / (rednoise[m].sum() + constants[m].sum()))
    recall.append(rednoise[m].sum() / rednoise.sum())

fig, ax = plt.subplots()
ax.plot(xx, precision, label="Precision")
ax.plot(xx, recall, label="Recall")
ax.set_xlabel("Precision")
ax.set_ylabel("Score")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
fig.savefig("som_timeseries_f1.pdf", dpi=300)
plt.close()

# %
